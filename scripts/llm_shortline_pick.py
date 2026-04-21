"""llm_shortline_pick — 读 watchlist_signal_v2 top N 信号, 调 LLM 做短线决策.

默认 backend: DeepSeek (便宜, ~0.003¥/次), 可切:
    LLM_BACKEND=anthropic python3 scripts/llm_shortline_pick.py
    LLM_BACKEND=deepseek  python3 scripts/llm_shortline_pick.py  (默认)

输出:
    - 每只股: action(buy/watch/avoid) + 止损位 + 止盈位 + 一句总结
    - 飞书聚合推送 "🧠 LLM 短线精选"
    - CSV 落盘 /output/paper_trade/llm_shortline/<date>.csv
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

env = Path(__file__).resolve().parent.parent / ".env"
if env.exists():
    for line in env.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from llm_layer.agents import _LLMBackend
from llm_layer import xml_parser as xp
from llm_layer.prompts_shortline import SHORTLINE_PICK_PROMPT

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
PAPER = ROOT / "output" / "paper_trade"


BACKEND_MODELS = {
    "deepseek":  "deepseek-chat",
    "anthropic": "claude-haiku-4-5",
    "mock":      "mock",
}


def load_latest_v2_signal() -> pd.DataFrame:
    sig_dir = PAPER / "watchlist_signals_v2"
    csvs = sorted(sig_dir.glob("*.csv")) if sig_dir.exists() else []
    if not csvs:
        raise FileNotFoundError("无 v2 信号 CSV, 先跑 watchlist_signal_v2.py")
    latest = csvs[-1]
    print(f"  v2 信号: {latest.name}")
    df = pd.read_csv(latest, index_col=0)
    df.index = df.index.astype(str).str.zfill(6)
    return df


def load_latest_sentiment() -> dict:
    sdir = ROOT / "output" / "sentiment_cycle"
    js = sorted(sdir.glob("*.json")) if sdir.exists() else []
    if not js:
        return {
            "regime": "⚪️ 平稳", "limit_up_count": 0,
            "max_streak_up": 0, "boom_rate": 0.0,
        }
    return json.loads(js[-1].read_text(encoding="utf-8"))


def load_kline_stats(codes: list[str]) -> dict:
    """对每只股算 pct_5d / pct_20d / MA5 / MA20."""
    wl_cache = sorted(CACHE.glob("watchlist_kline_*.parquet"))
    if not wl_cache:
        return {}
    df = pd.read_parquet(wl_cache[-1])
    df["date"] = pd.to_datetime(df["date"])
    df["code"] = df["code"].astype(str).str.zfill(6)
    df = df.sort_values(["code", "date"])

    out = {}
    for c in codes:
        sub = df[df["code"] == c].tail(21)
        if len(sub) < 21:
            continue
        close = sub["close"].values
        out[c] = {
            "pct_5d":  (close[-1] / close[-6] - 1) * 100 if len(close) >= 6 else 0,
            "pct_20d": (close[-1] / close[-21] - 1) * 100,
            "ma5":     float(close[-5:].mean()),
            "ma20":    float(close[-20:].mean()),
        }
    return out


def build_prompt(row: pd.Series, stats: dict, sent: dict) -> str:
    return SHORTLINE_PICK_PROMPT.format(
        code=row.name, name=row["name"],
        price=row["latest_close"],
        alpha_z=row["alpha_z"],
        top_category=row["top_category"],
        rev_score=row.get("rev_score", 0),
        limit_score=row.get("limit_score", 0),
        seat_score=row.get("seat_score", 0),
        pct_5d=stats.get("pct_5d", 0),
        pct_20d=stats.get("pct_20d", 0),
        ma5=f"{stats.get('ma5', 0):.2f}",
        ma20=f"{stats.get('ma20', 0):.2f}",
        sentiment_regime=sent.get("regime", "⚪️ 平稳"),
        limit_up_count=sent.get("limit_up_count", 0),
        max_streak=sent.get("max_streak_up", 0),
        boom_rate=sent.get("boom_rate", 0),
    )


def _normalize_action(raw: str | None) -> str:
    """归一化 LLM 输出 action 到 buy/watch/avoid."""
    s = (raw or "").strip().lower()
    if s in ("buy", "add", "long"):
        return "buy"
    if s in ("avoid", "sell", "exit", "short"):
        return "avoid"
    return "watch"


def extract_decision(raw: str) -> dict:
    """从 LLM XML 输出抽取字段."""
    return {
        "action":      _normalize_action(xp.extract_tag(raw, "ACTION")),
        "conviction":  _to_float(xp.extract_tag(raw, "CONVICTION"), 0.0),
        "stop_loss":   _to_float(xp.extract_tag(raw, "STOP_LOSS"), None),
        "take_profit": _to_float(xp.extract_tag(raw, "TAKE_PROFIT"), None),
        "holding":     _to_int(xp.extract_tag(raw, "HOLDING_DAYS"), 2),
        "reason":      (xp.extract_tag(raw, "EXPLANATION") or "").strip(),
        "risk":        (xp.extract_tag(raw, "RISK") or "").strip(),
    }


def _to_float(s: str | None, default):
    try:
        return float(s) if s else default
    except ValueError:
        return default


def _to_int(s: str | None, default):
    try:
        return int(float(s)) if s else default
    except ValueError:
        return default


def decide_one(row: pd.Series, stats: dict, sent: dict,
                backend: _LLMBackend) -> dict:
    code = row.name
    prompt = build_prompt(row, stats, sent)
    try:
        raw = backend.chat(prompt, max_tokens=900)
        d = extract_decision(raw)
        d.update({
            "code": code, "name": row["name"],
            "price": row["latest_close"], "alpha_z": row["alpha_z"],
        })
        return d
    except Exception as e:
        return {
            "code": code, "name": row["name"],
            "price": row["latest_close"], "alpha_z": row["alpha_z"],
            "action": "error", "reason": f"LLM 失败: {e}",
        }


def build_report(decisions: list[dict], sent: dict, backend_name: str) -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    lines = [
        f"🧠 **LLM 短线精选 — {today}** (backend: {backend_name})",
        "",
        f"市场情绪: {sent.get('regime', '⚪️')}  "
        f"涨停 {sent.get('limit_up_count', 0)} 只 "
        f"最高 {sent.get('max_streak_up', 0)} 连板",
        "",
    ]
    groups = {"buy": [], "watch": [], "avoid": [], "error": []}
    for d in decisions:
        groups.setdefault(d.get("action", "watch"), []).append(d)

    if groups["buy"]:
        lines.append(f"**🟢 买入 ({len(groups['buy'])} 只)**")
        for d in sorted(groups["buy"], key=lambda x: -x.get("conviction", 0)):
            sl = f"止损 ¥{d['stop_loss']:.2f}" if d.get("stop_loss") else ""
            tp = f"止盈 ¥{d['take_profit']:.2f}" if d.get("take_profit") else ""
            lines.append(
                f"  • {d['name']} ¥{d['price']:.2f}  "
                f"z={d['alpha_z']:+.2f}  信心 {d.get('conviction', 0):.0%}  "
                f"{d.get('holding', 2)}天  {sl} {tp}"
            )
            if d.get("reason"):
                lines.append(f"    💬 {d['reason']}")
        lines.append("")

    if groups["watch"]:
        lines.append(f"**👀 观望 ({len(groups['watch'])} 只)**")
        for d in groups["watch"][:5]:
            lines.append(
                f"  • {d['name']} ¥{d['price']:.2f}  z={d['alpha_z']:+.2f}  "
                f"💬 {d.get('reason', '')}"
            )
        lines.append("")

    if groups["avoid"]:
        lines.append(f"**🚫 回避 ({len(groups['avoid'])} 只)**")
        for d in groups["avoid"][:5]:
            lines.append(
                f"  • {d['name']} ¥{d['price']:.2f}  z={d['alpha_z']:+.2f}  "
                f"💬 {d.get('reason', '')}"
            )
        lines.append("")

    if groups["error"]:
        lines.append(f"⚠️ {len(groups['error'])} 只股 LLM 调用失败")
        lines.append("")

    lines.append("⚠️ LLM 决策依赖 prompt 质量和市场语料,与量化信号互为补充. "
                 "不构成投资建议.")
    return "\n".join(lines)


def send_lark(md: str) -> bool:
    user_id = os.environ.get("LARK_USER_OPEN_ID",
                              "ou_5be0f87dc7cec796b7ea97d0a9b5302f")
    cmd = ["lark-cli", "im", "+messages-send",
           "--as", "user", "--user-id", user_id, "--markdown", md]
    try:
        rc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if rc.returncode == 0:
            print("✓ 飞书已送达")
            return True
        print(f"❌ 飞书失败: {rc.stderr[:200]}")
    except Exception as e:
        print(f"❌ 飞书异常: {e}")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=10, help="取前 N 只做 LLM 分析")
    ap.add_argument("--backend", default=os.getenv("LLM_BACKEND", "deepseek"),
                    choices=["deepseek", "anthropic", "mock"])
    ap.add_argument("--dry-run-lark", action="store_true")
    ap.add_argument("--concurrency", type=int, default=3)
    args = ap.parse_args()

    print(f"\n{'='*64}\n  🧠 LLM 短线 {datetime.now():%Y-%m-%d %H:%M} ({args.backend})\n{'='*64}")

    sig = load_latest_v2_signal()
    sent = load_latest_sentiment()
    top_df = sig.head(args.top)
    print(f"  top {len(top_df)} 只做 LLM 分析")

    stats_map = load_kline_stats(top_df.index.tolist())
    print(f"  K 线统计: {len(stats_map)} 只")

    backend = _LLMBackend(args.backend, BACKEND_MODELS[args.backend])

    # 并发调用
    decisions = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {
            ex.submit(decide_one, row, stats_map.get(code, {}), sent, backend): code
            for code, row in top_df.iterrows()
        }
        for fut in as_completed(futs):
            d = fut.result()
            decisions.append(d)
            print(f"  ✓ {d['name']:<8} → {d.get('action', '?'):<6} "
                  f"(信心 {d.get('conviction', 0):.0%})")

    # 按 alpha_z 排
    decisions.sort(key=lambda d: -d.get("alpha_z", 0))

    report = build_report(decisions, sent, args.backend)
    print(f"\n{'='*64}\n飞书消息:\n{'='*64}")
    print(report)

    out_dir = PAPER / "llm_shortline"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{datetime.now():%Y-%m-%d}.csv"
    pd.DataFrame(decisions).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n  CSV: {out}")

    if not args.dry_run_lark:
        send_lark(report)


if __name__ == "__main__":
    main()
