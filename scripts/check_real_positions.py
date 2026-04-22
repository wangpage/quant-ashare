"""check_real_positions — 监控真实资金持仓, 触发硬止损/冲高减半警报.

与 paper_trade_runner 独立 (paper 是模拟策略, 这里是真实资金追踪).

触发规则:
    🛑 硬止损1: 当前价 ≤ 成本 × (1 - stop_loss_pct)   默认 5%
    🛑 硬止损2: 当前价 < MA5                           (use_ma5=true)
    ⚠️ 冲高减半: 当前价 ≥ 成本 × (1 + take_profit_pct) 默认 5%
    📅 到期退出: 今日 ≥ target_exit (T+1 最大持有天数)

数据源: akshare stock_zh_a_spot_em (盘中实时)
     fallback: 本地 watchlist_kline 缓存最新收盘价

用法 (默认推飞书):
    python3 scripts/check_real_positions.py
    python3 scripts/check_real_positions.py --dry-run-lark

crontab 建议:
    35 9 * * 1-5   # 开盘后 5 分钟
    0 13 * * 1-5   # 午后开盘
    50 14 * * 1-5  # 尾盘最后 10 分钟 (硬止损最后窗口)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
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

ROOT = Path(__file__).resolve().parent.parent
LARK_BIN = "/opt/homebrew/bin/lark-cli"
CACHE = ROOT / "cache"
ACCOUNT = ROOT / "output" / "real_positions" / "account.json"


def load_account() -> dict:
    if not ACCOUNT.exists():
        return {"positions": {}, "history": []}
    return json.loads(ACCOUNT.read_text(encoding="utf-8"))


def save_account(acc: dict):
    ACCOUNT.write_text(
        json.dumps(acc, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def fetch_spot(codes: list[str]) -> dict[str, dict]:
    """返回 {code: {price, pct_chg, open, prev_close, source}}."""
    # 先试 akshare spot
    try:
        import akshare as ak
        df = ak.stock_zh_a_spot_em()
        df = df.rename(columns={
            "代码": "code", "最新价": "price",
            "涨跌幅": "pct_chg", "今开": "open", "昨收": "prev_close",
        })
        df["code"] = df["code"].astype(str).str.zfill(6)
        sub = df[df["code"].isin(codes)]
        return {
            r["code"]: {
                "price": float(r["price"]),
                "pct_chg": float(r["pct_chg"]) if pd.notna(r["pct_chg"]) else 0,
                "open": float(r["open"]) if pd.notna(r["open"]) else None,
                "prev_close": float(r["prev_close"]) if pd.notna(r["prev_close"]) else None,
                "source": "akshare_spot",
            }
            for _, r in sub.iterrows()
        }
    except Exception as e:
        print(f"  ⚠️ akshare spot 失败 ({e}), fallback 到本地缓存")

    # fallback: 本地最新 kline
    wl_cache = sorted(CACHE.glob("watchlist_kline_*.parquet"))
    if not wl_cache:
        return {}
    df = pd.read_parquet(wl_cache[-1])
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["date"] = pd.to_datetime(df["date"])
    latest = df[df["date"] == df["date"].max()]
    prev_date = df[df["date"] < df["date"].max()]["date"].max() if len(df) > 0 else None

    out = {}
    for c in codes:
        rec = latest[latest["code"] == c]
        if rec.empty:
            continue
        r = rec.iloc[0]
        prev_close = None
        if prev_date is not None:
            p_rec = df[(df["code"] == c) & (df["date"] == prev_date)]
            if not p_rec.empty:
                prev_close = float(p_rec.iloc[0]["close"])
        out[c] = {
            "price": float(r["close"]),
            "pct_chg": float(r["pct_chg"]) if "pct_chg" in r and pd.notna(r["pct_chg"]) else 0,
            "open": float(r["open"]),
            "prev_close": prev_close,
            "source": f"cache_{wl_cache[-1].name}",
        }
    return out


def compute_ma5(code: str) -> float | None:
    """用本地缓存算近 5 个交易日 MA5."""
    wl_cache = sorted(CACHE.glob("watchlist_kline_*.parquet"))
    if not wl_cache:
        return None
    df = pd.read_parquet(wl_cache[-1])
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["date"] = pd.to_datetime(df["date"])
    sub = df[df["code"] == code].sort_values("date").tail(5)
    if len(sub) < 5:
        return None
    return float(sub["close"].mean())


def evaluate(pos: dict, spot: dict) -> dict:
    """评估一只股的当前状态."""
    cost = pos["cost"]
    price = spot["price"]
    pnl_pct = (price - cost) / cost
    pnl_amt = (price - cost) * pos["shares"]
    mkt_value = price * pos["shares"]

    stop_loss_pct = pos.get("stop_loss_pct", 0.05)
    take_profit_pct = pos.get("take_profit_pct", 0.05)
    stop_price = cost * (1 - stop_loss_pct)
    target_price = cost * (1 + take_profit_pct)

    # MA5
    ma5 = compute_ma5(pos.get("code", "")) if pos.get("use_ma5") else None

    triggers = []
    if price <= stop_price:
        triggers.append(f"🛑 硬止损 -{stop_loss_pct:.0%}: ¥{price:.2f} ≤ ¥{stop_price:.2f}")
    if ma5 is not None and price < ma5:
        triggers.append(f"🛑 跌破 MA5: ¥{price:.2f} < ¥{ma5:.2f}")
    if price >= target_price:
        triggers.append(f"✅ 冲高目标 +{take_profit_pct:.0%}: ¥{price:.2f} ≥ ¥{target_price:.2f}")

    # 判断综合状态
    if any("🛑" in t for t in triggers):
        verdict = "🔴 触发止损 - 建议 T+1 开盘卖出"
    elif any("✅" in t for t in triggers):
        verdict = "🟢 冲高区 - 建议减一半锁利润"
    elif pnl_pct > 0.02:
        verdict = "🟢 盈利区"
    elif pnl_pct < -0.02:
        verdict = "🟡 浮亏,紧盯硬底"
    else:
        verdict = "⚪️ 持平区"

    return {
        "price": price, "cost": cost,
        "mkt_value": mkt_value,
        "pnl_pct": pnl_pct, "pnl_amt": pnl_amt,
        "stop_price": stop_price, "target_price": target_price,
        "ma5": ma5, "triggers": triggers,
        "verdict": verdict, "source": spot["source"],
    }


def build_report(acc: dict, evals: dict) -> str:
    today = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"💼 **实盘持仓监控 — {today}**",
        "",
    ]
    total_mv = 0
    total_pnl = 0
    total_cost = 0
    for code, pos in acc["positions"].items():
        e = evals.get(code)
        if not e:
            lines.append(f"• {pos['name']} {code}: 无行情数据")
            continue
        total_mv += e["mkt_value"]
        total_pnl += e["pnl_amt"]
        total_cost += pos["cost"] * pos["shares"]
        lines.append(f"**{pos['name']} {code}** — {e['verdict']}")
        lines.append(
            f"  现价 ¥{e['price']:.2f}  成本 ¥{e['cost']:.2f}  "
            f"盈亏 {e['pnl_pct']:+.2%} (¥{e['pnl_amt']:+,.0f})"
        )
        lines.append(
            f"  市值 ¥{e['mkt_value']:,.0f}  "
            f"硬底 ¥{e['stop_price']:.2f}"
            + (f"  MA5 ¥{e['ma5']:.2f}" if e['ma5'] else "")
            + f"  冲高线 ¥{e['target_price']:.2f}"
        )
        if e["triggers"]:
            for t in e["triggers"]:
                lines.append(f"  {t}")
        lines.append(f"  📅 T+1 可卖  建议最晚退出: {pos.get('target_exit', 'N/A')}")
        lines.append("")

    if total_cost > 0:
        lines.append(
            f"**汇总**: 成本 ¥{total_cost:,.0f}  市值 ¥{total_mv:,.0f}  "
            f"总盈亏 {total_pnl/total_cost:+.2%} (¥{total_pnl:+,.0f})"
        )

    sources = set(e.get("source", "") for e in evals.values())
    lines.append(f"\n数据源: {', '.join(sources)}")
    return "\n".join(lines)


def send_lark(md: str) -> bool:
    user_id = os.environ.get("LARK_USER_OPEN_ID",
                              "ou_5be0f87dc7cec796b7ea97d0a9b5302f")
    cmd = [str(LARK_BIN), "im", "+messages-send",
           "--as", "user", "--user-id", user_id, "--markdown", md]
    try:
        env = {**os.environ, "PATH": f"/opt/homebrew/bin:{os.environ.get('PATH', '/usr/local/bin:/usr/bin:/bin')}"}
        rc = subprocess.run(cmd, capture_output=True, text=True, timeout=30, env=env)
        if rc.returncode == 0:
            print("✓ 飞书已送达")
            return True
        print(f"❌ 飞书失败: {rc.stderr[:200]}")
    except Exception as e:
        print(f"❌ 飞书异常: {e}")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run-lark", action="store_true")
    ap.add_argument("--alert-only", action="store_true",
                    help="只有触发止损/冲高时才推飞书")
    args = ap.parse_args()

    print(f"\n{'='*64}\n  💼 Real Positions Check {datetime.now():%Y-%m-%d %H:%M}\n{'='*64}")

    acc = load_account()
    if not acc["positions"]:
        print("  无持仓, 退出")
        return 0
    codes = list(acc["positions"].keys())
    print(f"  持仓: {len(codes)} 只")

    # 取实时价
    spots = fetch_spot(codes)
    if not spots:
        print("❌ 拉不到任何行情")
        return 1
    print(f"  行情: {len(spots)} 只")

    # 评估
    evals = {}
    for code, pos in acc["positions"].items():
        pos["code"] = code   # 注入 code 方便 compute_ma5
        if code in spots:
            evals[code] = evaluate(pos, spots[code])

    # 报告
    report = build_report(acc, evals)
    print(f"\n{'='*64}\n{report}")

    has_trigger = any(e.get("triggers") for e in evals.values())
    if args.alert_only and not has_trigger:
        print("\n  无触发, 按 --alert-only 规则不推飞书")
    elif not args.dry_run_lark:
        send_lark(report)


if __name__ == "__main__":
    main()
