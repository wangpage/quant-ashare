"""watchlist_intraday — 73 只自选股每 5 分钟盘中综合打分 + "明日买什么" Top N.

数据源: 新浪批量 quote (一次 HTTP 拉 73 只, 稳定无限频).

综合分数 (4 类因子加权):
    base_z          昨晚 v2 日频 alpha_z (反转/打板/席位)   权重 0.40
    pct_chg_rank    今日涨跌幅截面 rank                       权重 0.20
    clv_rank        盘中收盘位置 (近日高/低之间)             权重 0.20
    vol_ratio_rank  量比 (今日已成交 / 近 5 日均量 * 时间系数)  权重 0.20

→ 所有分数 z-score 合成 → 排序 → Top 10 候选推飞书

去重策略 (降低推送频次):
    - Top 3 名单变化 → 必推
    - 整点 (10:00/10:30/11:00/13:30/14:30/14:55) → 必推
    - 其他时段 Top 10 重合度 ≥ 80% → 静默

调用:
    python3 scripts/watchlist_intraday.py                # 正常推送 (去重)
    python3 scripts/watchlist_intraday.py --force        # 强制推
    python3 scripts/watchlist_intraday.py --dry-run-lark
    python3 scripts/watchlist_intraday.py --top 10       # 推 top N
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import yaml

env = Path(__file__).resolve().parent.parent / ".env"
if env.exists():
    for line in env.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

ROOT = Path(__file__).resolve().parent.parent
LARK_BIN = "/opt/homebrew/bin/lark-cli"
CACHE = ROOT / "cache"
PAPER = ROOT / "output" / "paper_trade"
OUT_DIR = ROOT / "output" / "watchlist_intraday"
STATE = OUT_DIR / "state.json"


# ---------- 数据 ----------
def load_watchlist() -> list[dict]:
    path = ROOT / "config" / "user_watchlist.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["watchlist"]


def _sina_symbol(code: str) -> str:
    code = str(code).zfill(6)
    return f"sh{code}" if code.startswith("6") else f"sz{code}"


def fetch_sina_batch(codes: list[str]) -> pd.DataFrame:
    """sina 批量实时 quote. 支持 80+ 只一次拉."""
    syms = ",".join(_sina_symbol(c) for c in codes)
    url = f"https://hq.sinajs.cn/list={syms}"
    try:
        out = subprocess.run(
            ["curl", "-s",
             "-A", "Mozilla/5.0 (Macintosh) AppleWebKit/537.36 Chrome/120",
             "-H", "Referer: https://finance.sina.com.cn/",
             "--max-time", "15", url],
            capture_output=True, text=False, timeout=18,
        )
        raw = out.stdout.decode("gbk", errors="replace")
    except Exception as e:
        print(f"  ❌ sina 批量失败: {e}")
        return pd.DataFrame()

    rows = []
    for line in raw.splitlines():
        if not line.startswith("var hq_str_"):
            continue
        try:
            key_part, payload = line.split("=", 1)
            sym = key_part.replace("var hq_str_", "").strip()
            code = sym[2:]  # 去 sh/sz 前缀
            payload = payload.strip().strip(";").strip('"')
            parts = payload.split(",")
            if len(parts) < 32 or not parts[3] or parts[3] == "0.000":
                continue
            prev_close = float(parts[2])
            price = float(parts[3])
            open_px = float(parts[1])
            high = float(parts[4])
            low = float(parts[5])
            volume_shares = int(parts[8])    # 股
            amount = float(parts[9])
            rows.append({
                "code":       code,
                "price":      price,
                "open":       open_px,
                "prev_close": prev_close,
                "high":       high,
                "low":        low,
                "volume":     volume_shares // 100,   # sina 返回股, 转成手 (与 cache 对齐)
                "amount":     amount,
                "pct_chg":    (price / prev_close - 1) * 100 if prev_close else 0,
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


def load_v2_alpha() -> dict[str, float]:
    """读最新 v2 日频 alpha_z."""
    sig_dir = PAPER / "watchlist_signals_v2"
    csvs = sorted(sig_dir.glob("*.csv")) if sig_dir.exists() else []
    if not csvs:
        return {}
    df = pd.read_csv(csvs[-1], index_col=0)
    df.index = df.index.astype(str).str.zfill(6)
    return df["alpha_z"].to_dict()


def load_kline_stats() -> dict[str, dict]:
    """读最新 watchlist_kline, 算每只股的近 5 日均量 + MA5 + MA20."""
    wl_cache = sorted(CACHE.glob("watchlist_kline_*.parquet"))
    if not wl_cache:
        return {}
    df = pd.read_parquet(wl_cache[-1])
    df["date"] = pd.to_datetime(df["date"])
    df["code"] = df["code"].astype(str).str.zfill(6)
    out = {}
    for c, g in df.groupby("code"):
        g = g.sort_values("date").tail(20)
        if len(g) < 5:
            continue
        out[c] = {
            "avg_vol_5d": float(g.tail(5)["volume"].mean()),
            "ma5":        float(g.tail(5)["close"].mean()),
            "ma20":       float(g.tail(20)["close"].mean()),
            "prev_close": float(g.iloc[-1]["close"]),
        }
    return out


# ---------- 盘中打分 ----------
def compute_scores(quotes: pd.DataFrame, v2_alpha: dict,
                    kline_stats: dict) -> pd.DataFrame:
    """多维度截面 rank → z-score 合成."""
    df = quotes.copy().set_index("code")

    # 1. base_z (昨晚 v2 日频 alpha)
    df["base_z"] = df.index.map(lambda c: v2_alpha.get(c, 0))

    # 2. 今日涨跌幅 rank
    df["pct_chg"] = df["pct_chg"].astype(float)

    # 3. CLV (日内收盘位置) - 反映强弱
    hl = df["high"] - df["low"]
    df["clv"] = ((df["price"] - df["low"]) / hl).where(hl > 0, 0.5)

    # 4. 量比 = 今日已成交量 / 近 5 日均量, 按交易时间折算
    now = datetime.now()
    total_trading_seconds = (4 * 60) * 60   # 4 小时
    passed_seconds = _elapsed_trading_seconds(now)
    time_frac = max(passed_seconds / total_trading_seconds, 0.05)

    def _vol_ratio(row):
        c = row.name
        k = kline_stats.get(c)
        if not k or k["avg_vol_5d"] <= 0:
            return 1.0
        expected = k["avg_vol_5d"] * time_frac
        return row["volume"] / expected if expected > 0 else 1.0

    df["vol_ratio"] = df.apply(_vol_ratio, axis=1)

    # 5. MA 位置 (用于辅助判断)
    df["ma5"] = df.index.map(lambda c: kline_stats.get(c, {}).get("ma5", 0))
    df["ma20"] = df.index.map(lambda c: kline_stats.get(c, {}).get("ma20", 0))
    df["vs_ma5"] = (df["price"] / df["ma5"] - 1) * 100
    df["vs_ma20"] = (df["price"] / df["ma20"] - 1) * 100

    # rank → 中心化
    def _rank_center(s):
        r = s.rank(pct=True).fillna(0.5)
        return r - 0.5

    df["r_base"] = _rank_center(df["base_z"])
    df["r_pct"] = _rank_center(df["pct_chg"])
    df["r_clv"] = _rank_center(df["clv"])
    df["r_vol"] = _rank_center(df["vol_ratio"])

    # 综合 score
    df["score"] = (df["r_base"] * 0.40
                   + df["r_pct"] * 0.20
                   + df["r_clv"] * 0.20
                   + df["r_vol"] * 0.20)

    # z-score
    mu = df["score"].mean()
    sd = df["score"].std()
    df["composite_z"] = (df["score"] - mu) / (sd + 1e-9)

    return df.sort_values("composite_z", ascending=False)


def _elapsed_trading_seconds(now: datetime) -> int:
    """已过去的交易秒数 (9:30-11:30 + 13:00-15:00)."""
    t = now.time()
    if t < time(9, 30):
        return 0
    if t <= time(11, 30):
        start = now.replace(hour=9, minute=30, second=0, microsecond=0)
        return int((now - start).total_seconds())
    if t < time(13, 0):
        return 2 * 3600   # 午休,按上午已过
    if t <= time(15, 0):
        start = now.replace(hour=13, minute=0, second=0, microsecond=0)
        return 2 * 3600 + int((now - start).total_seconds())
    return 4 * 3600


# ---------- 去重 ----------
def load_state() -> dict:
    if not STATE.exists():
        return {}
    return json.loads(STATE.read_text(encoding="utf-8"))


def save_state(s: dict):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(s, indent=2, ensure_ascii=False, default=str),
                     encoding="utf-8")


def should_push(top_codes: list[str], state: dict,
                 force: bool = False) -> tuple[bool, str]:
    if force:
        return True, "force"

    now = datetime.now()
    last_top = state.get("last_top", [])
    last_push = state.get("last_push")

    # Top 3 变化 → 必推
    if set(top_codes[:3]) != set(last_top[:3]):
        return True, f"top3_change"

    # 整点
    key_times = [(10, 0), (10, 30), (11, 0), (13, 0),
                 (13, 30), (14, 0), (14, 30), (14, 55)]
    for h, m in key_times:
        if now.hour == h and abs(now.minute - m) <= 4:
            if last_push:
                last_t = datetime.fromisoformat(last_push)
                if (now - last_t).total_seconds() < 20 * 60:
                    continue
            return True, f"keytime_{h:02d}{m:02d}"

    # Top 10 重合度
    if last_top:
        overlap = len(set(top_codes[:10]) & set(last_top[:10])) / 10
        if overlap < 0.8:
            return True, f"top10_change_overlap_{overlap:.0%}"

    return False, "no_change"


# ---------- 推送 ----------
def build_report(sig: pd.DataFrame, name_map: dict,
                  top_n: int, state: dict) -> str:
    now = datetime.now()
    lines = [
        f"📡 **自选盘中扫描 {now:%H:%M}**",
        "",
        f"Universe: {len(sig)} 只 | 当前在盘",
    ]

    # 市场概览
    up = (sig["pct_chg"] > 0).sum()
    down = (sig["pct_chg"] < 0).sum()
    strong = (sig["pct_chg"] >= 3).sum()
    weak = (sig["pct_chg"] <= -3).sum()
    avg_pct = sig["pct_chg"].mean()
    lines.append(f"涨/跌: {up}/{down}  强势(≥3%): {strong}  弱势(≤-3%): {weak}  "
                 f"均值: {avg_pct:+.2f}%")
    lines.append("")

    # Top N
    top = sig.head(top_n)
    lines.append(f"**🎯 明日买入候选 Top {top_n} (综合 z)**")

    # 对比上次
    last_top = state.get("last_top", [])
    new_entries = [c for c in top.index[:top_n]
                   if c not in last_top[:top_n]]

    for rank, (code, r) in enumerate(top.iterrows(), 1):
        is_new = code in new_entries
        mark = " 🆕" if is_new else ""
        name = name_map.get(code, code)
        lines.append(
            f"{rank}. {name} ({code}){mark}  "
            f"¥{r['price']:.2f} ({r['pct_chg']:+.1f}%)  "
            f"z={r['composite_z']:+.2f}"
        )
        lines.append(
            f"   base={r['base_z']:+.2f}  日内={r['clv']:.0%}位  "
            f"量比 {r['vol_ratio']:.1f}x  vsMA5 {r['vs_ma5']:+.1f}%"
        )
    lines.append("")

    # 掉出 top 的
    dropped = [c for c in last_top[:top_n]
               if c not in top.index[:top_n].tolist()]
    if dropped:
        lines.append(f"📤 本轮掉出: {', '.join([name_map.get(c, c) for c in dropped[:3]])}")
        lines.append("")

    lines.append(f"⚠️ base_z 来自昨晚 v2 日频 (打板/席位/反转), "
                 f"盘中不变; 叠加今日涨跌/强弱/量能做 T+1 候选.")
    return "\n".join(lines)


def send_lark(md: str) -> bool:
    user_id = os.environ.get("LARK_USER_OPEN_ID",
                              "ou_5be0f87dc7cec796b7ea97d0a9b5302f")
    cmd = [LARK_BIN, "im", "+messages-send",
           "--as", "user", "--user-id", user_id, "--markdown", md]
    try:
        env = {**os.environ,
               "PATH": f"/opt/homebrew/bin:{os.environ.get('PATH', '')}"}
        rc = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=30, env=env)
        if rc.returncode == 0:
            print("✓ 飞书已送达")
            return True
        print(f"❌ 飞书失败: {rc.stderr[:200]}")
    except Exception as e:
        print(f"❌ 飞书异常: {e}")
    return False


def is_trading_hours() -> bool:
    now = datetime.now().time()
    return (time(9, 30) <= now <= time(11, 30)) or \
           (time(13, 0) <= now <= time(15, 0))


# ---------- 主 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run-lark", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    if not is_trading_hours() and not args.force:
        print(f"  非交易时间, 跳过 (--force 强跑)")
        return 0

    print(f"\n{'='*64}\n  📡 Watchlist Intraday {datetime.now():%H:%M}\n{'='*64}")

    wl = load_watchlist()
    codes = [w["code"].zfill(6) for w in wl]
    name_map = {w["code"].zfill(6): w["name"] for w in wl}

    # 1. 批量拉实时
    quotes = fetch_sina_batch(codes)
    if quotes.empty:
        print("❌ 实时数据空"); return 1
    quotes["code"] = quotes["code"].astype(str).str.zfill(6)
    print(f"  quote: {len(quotes)} 只")

    # 2. 昨晚 v2 alpha
    v2_alpha = load_v2_alpha()
    print(f"  v2 alpha 基准: {len(v2_alpha)} 只")

    # 3. 近 5 日均量等
    kstats = load_kline_stats()
    print(f"  K 线统计: {len(kstats)} 只")

    # 4. 打分
    sig = compute_scores(quotes, v2_alpha, kstats)

    # 5. 展示 Top
    top_codes = sig.head(args.top).index.tolist()
    print(f"\n  Top {args.top}:")
    for rank, (code, r) in enumerate(sig.head(args.top).iterrows(), 1):
        print(f"    {rank:2}. {name_map.get(code, code):<8} "
              f"¥{r['price']:>7.2f}  {r['pct_chg']:+5.1f}%  "
              f"z={r['composite_z']:+.2f}")

    # 6. 去重 + 推送
    state = load_state()
    push, reason = should_push(top_codes, state, force=args.force)
    print(f"\n  推送判断: {'推' if push else '跳过'} ({reason})")

    if push:
        report = build_report(sig, name_map, args.top, state)
        if not args.dry_run_lark:
            send_lark(report)
        else:
            print("\n" + report)
        # 更新状态
        state["last_top"] = top_codes
        state["last_push"] = datetime.now().isoformat()
        save_state(state)

    # 7. 落盘每次快照 CSV
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    snap = sig[["price", "pct_chg", "clv", "vol_ratio",
                "base_z", "composite_z", "vs_ma5", "vs_ma20"]].copy()
    snap.insert(0, "name", snap.index.map(name_map))
    snap.to_csv(OUT_DIR / f"{datetime.now():%Y%m%d_%H%M}.csv",
                encoding="utf-8-sig")


if __name__ == "__main__":
    main()
