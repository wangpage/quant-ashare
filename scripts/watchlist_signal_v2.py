"""watchlist_signal v2 — 短线复合 alpha 信号.

v1 vs v2:
    v1: 7 个反转/低波因子
    v2: 7 反转 + 6 打板情绪 + 5 龙虎榜席位 = 18 因子复合

因子大类权重:
    反转/低波   0.40  (原 v1 保留)
    打板/情绪   0.40  (来自 alpha_limit.compute_limit_alpha)
    龙虎榜席位  0.20  (来自 seat_network.compute_seat_alpha)

每类内部因子极性:
    打板: STREAK_UP(+), STREAK_MAX_30(+), LU_COUNT_20(+), LU_NEXT_GAP_AVG(+),
          AMT_BREAKOUT(+), BOOM_RATE_20(-)
    席位: SEAT_FLAG_10(+), SEAT_NET_JIGOU_60(+), SEAT_INST_LEAD_20(+),
          SEAT_DIVERSITY_60(+), SEAT_PURE_HOTMONEY_20(-)

输出: 同 v1 风格, 额外标注"主导因子大类"(反转/打板/席位).
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

import numpy as np
import pandas as pd
import yaml

env = Path(__file__).resolve().parent.parent / ".env"
if env.exists():
    for line in env.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from data_adapter.em_direct import bulk_fetch_daily
from data_adapter.minute_kline import bulk_fetch_minute
from factors.alpha_limit import compute_limit_alpha
from factors.seat_network import compute_seat_alpha
from factors.alpha_intraday import compute_real_intraday_alpha

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
PAPER = ROOT / "output" / "paper_trade"

# ---------- 因子分类 + 权重 ----------
REVERSAL_WEIGHTS = {
    "REV_5":       0.20,
    "REV_20":      0.15,
    "MOM_126_21":  0.10,
    "LOW_VOL_60":  0.15,
    "TURN_Z_60":  -0.15,
    "MAX_RET_5":  -0.15,
    "AMIHUD_60":   0.10,
}
LIMIT_WEIGHTS = {
    "STREAK_UP":        1.0,
    "STREAK_MAX_30":    0.8,
    "LU_COUNT_20":      1.0,
    "LU_NEXT_GAP_AVG":  1.2,
    "AMT_BREAKOUT":     0.8,
    "BOOM_RATE_20":    -1.2,
}
SEAT_WEIGHTS = {
    "SEAT_FLAG_10":           1.0,
    "SEAT_NET_JIGOU_60":      1.0,
    "SEAT_INST_LEAD_20":      1.0,
    "SEAT_DIVERSITY_60":      0.8,
    "SEAT_PURE_HOTMONEY_20": -1.0,
}
INTRADAY_WEIGHTS = {
    # 日内高点停留 = 强势, + ; 低点停留 = 弱势, -
    "R_HIGH_DWELL":   1.0,
    "R_LOW_DWELL":   -1.0,
    # 尾盘拉高 = 明日看涨
    "R_TAIL_RET":     1.2,
    # 开盘巨量 = 抢筹 / 出货 (视 alpha 方向决定, 这里中性 +)
    "R_VOL_HEAD":     0.5,
    # 日中穿越次数越多 = 多空焦灼, 中性偏负
    "R_MID_CROSS":   -0.3,
}
CATEGORY_WEIGHTS_BASE = {
    "reversal": 0.40,
    "limit":    0.40,
    "seat":     0.20,
}
CATEGORY_WEIGHTS_WITH_INTRADAY = {
    "reversal": 0.30,
    "limit":    0.35,
    "seat":     0.20,
    "intraday": 0.15,
}


def load_watchlist() -> list[dict]:
    path = ROOT / "config" / "user_watchlist.yaml"
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["watchlist"]


def fetch_watchlist_kline(codes: list[str], days_back: int = 180) -> pd.DataFrame:
    today = pd.Timestamp.today().normalize()
    start = (today - pd.Timedelta(days=int(days_back * 1.5))).strftime("%Y%m%d")
    end = today.strftime("%Y%m%d")

    cache_path = CACHE / f"watchlist_kline_{end}.parquet"
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        df["date"] = pd.to_datetime(df["date"])
        cached = set(df["code"].unique())
        missing = [c for c in codes if c not in cached]
        if not missing:
            return df[df["code"].isin(codes)]
        extra = bulk_fetch_daily(missing, start, end, sleep_ms=80, progress=False)
        if not extra.empty:
            df = pd.concat([df, extra], ignore_index=True)
            df.to_parquet(cache_path)
        return df[df["code"].isin(codes)]

    print(f"  拉 {len(codes)} 只日 K ({start}~{end})...")
    df = bulk_fetch_daily(codes, start, end, sleep_ms=80, progress=False)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path)
    return df


def load_lhb() -> pd.DataFrame | None:
    """找 cache 里最新的 lhb parquet."""
    lhbs = sorted([p for p in CACHE.glob("lhb_2*.parquet")
                   if "taxonomy" not in p.name])
    if not lhbs:
        return None
    df = pd.read_parquet(lhbs[-1])
    print(f"  龙虎榜数据: {lhbs[-1].name} ({len(df)} 条)")
    return df


# ---------- 反转因子 (同 v1) ----------
def compute_reversal_per_stock(g: pd.DataFrame) -> pd.Series:
    g = g.sort_values("date").copy()
    close, high, low = g["close"], g["high"], g["low"]
    vol = g["volume"].astype(float)
    amt = g.get("amount", vol * close).astype(float)
    if (amt == 0).all():
        amt = vol * close
    ret1 = close.pct_change()

    out = {}
    if len(close) >= 6:
        out["REV_5"] = -(close.iloc[-1] / close.iloc[-6] - 1)
    if len(close) >= 21:
        out["REV_20"] = -(close.iloc[-1] / close.iloc[-21] - 1)
    if len(close) >= 127:
        ret126 = close.iloc[-1] / close.iloc[-127] - 1
        ret21 = close.iloc[-1] / close.iloc[-22] - 1
        out["MOM_126_21"] = ret126 - ret21
    if len(ret1) >= 60:
        out["LOW_VOL_60"] = -ret1.tail(60).std()
    if len(vol) >= 60:
        v_recent = vol.iloc[-1]
        v_mu = vol.tail(60).mean()
        v_sd = vol.tail(60).std()
        if v_sd > 0:
            out["TURN_Z_60"] = (v_recent - v_mu) / v_sd
    if len(ret1) >= 5:
        out["MAX_RET_5"] = ret1.tail(5).max()
    if len(ret1) >= 60:
        amihud = (ret1.abs() / (amt + 1)).tail(60).mean()
        out["AMIHUD_60"] = amihud
    return pd.Series(out)


def compute_reversal_panel(daily_df: pd.DataFrame) -> pd.DataFrame:
    """对每只股算最新一日的反转因子."""
    pieces = []
    for code, g in daily_df.groupby("code"):
        if len(g) < 60:
            continue
        s = compute_reversal_per_stock(g)
        s["code"] = code
        s["latest_close"] = float(g.sort_values("date")["close"].iloc[-1])
        s["latest_date"] = g["date"].max()
        pieces.append(s)
    if not pieces:
        return pd.DataFrame()
    return pd.DataFrame(pieces).set_index("code")


# ---------- 因子面板合成 ----------
def latest_slice(panel: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    """从 (date, code) MultiIndex 面板取 asof 当日截面."""
    if panel.empty:
        return pd.DataFrame()
    dates = panel.index.get_level_values("date").unique()
    use_date = dates[dates <= asof].max() if (dates <= asof).any() else dates.max()
    try:
        return panel.xs(use_date, level="date", drop_level=True)
    except KeyError:
        return pd.DataFrame()


def weighted_rank_score(df: pd.DataFrame, weights: dict) -> pd.Series:
    """每因子 rank(pct) 中心化 → 加权求和."""
    score = pd.Series(0.0, index=df.index)
    used = 0
    for f, w in weights.items():
        if f not in df.columns:
            continue
        col = df[f]
        if col.notna().sum() < 3:
            continue
        r = col.rank(pct=True).fillna(0.5) - 0.5
        score += r * w
        used += 1
    return score, used


def synthesize_composite(rev_df: pd.DataFrame,
                          limit_slice: pd.DataFrame,
                          seat_slice: pd.DataFrame,
                          intraday_slice: pd.DataFrame | None = None
                          ) -> pd.DataFrame:
    """复合 alpha_z + 主导大类. intraday_slice 可选 (启用 --use-minute 时)."""
    codes = rev_df.index
    out = rev_df.copy()

    use_intraday = intraday_slice is not None and not intraday_slice.empty
    cat_weights = (CATEGORY_WEIGHTS_WITH_INTRADAY if use_intraday
                   else CATEGORY_WEIGHTS_BASE)

    rev_score, _ = weighted_rank_score(rev_df, REVERSAL_WEIGHTS)
    out["rev_score"] = rev_score

    if not limit_slice.empty:
        limit_aligned = limit_slice.reindex(codes)
        limit_score, _ = weighted_rank_score(limit_aligned, LIMIT_WEIGHTS)
        for f in LIMIT_WEIGHTS:
            if f in limit_aligned.columns:
                out[f] = limit_aligned[f]
    else:
        limit_score = pd.Series(0.0, index=codes)
    out["limit_score"] = limit_score

    if not seat_slice.empty:
        seat_aligned = seat_slice.reindex(codes)
        seat_score, _ = weighted_rank_score(seat_aligned, SEAT_WEIGHTS)
        for f in SEAT_WEIGHTS:
            if f in seat_aligned.columns:
                out[f] = seat_aligned[f]
    else:
        seat_score = pd.Series(0.0, index=codes)
    out["seat_score"] = seat_score

    if use_intraday:
        intra_aligned = intraday_slice.reindex(codes)
        intra_score, _ = weighted_rank_score(intra_aligned, INTRADAY_WEIGHTS)
        for f in INTRADAY_WEIGHTS:
            if f in intra_aligned.columns:
                out[f] = intra_aligned[f]
        out["intraday_score"] = intra_score
    else:
        intra_score = None

    def _z(s):
        mu, sd = s.mean(), s.std()
        return (s - mu) / (sd + 1e-9)

    rev_z = _z(rev_score)
    lim_z = _z(limit_score)
    seat_z = _z(seat_score)

    composite = (rev_z * cat_weights["reversal"]
                 + lim_z * cat_weights["limit"]
                 + seat_z * cat_weights["seat"])

    if use_intraday:
        intra_z = _z(intra_score)
        composite = composite + intra_z * cat_weights["intraday"]

    out["alpha_z"] = _z(composite)

    contrib_cols = {
        "反转": rev_z * cat_weights["reversal"],
        "打板": lim_z * cat_weights["limit"],
        "席位": seat_z * cat_weights["seat"],
    }
    if use_intraday:
        contrib_cols["日内"] = intra_z * cat_weights["intraday"]
    contrib = pd.DataFrame(contrib_cols, index=codes)
    out["top_category"] = contrib.abs().idxmax(axis=1)
    out["cat_sign"] = contrib.apply(
        lambda row: "+" if row[row.abs().idxmax()] > 0 else "-", axis=1
    )

    return out.sort_values("alpha_z", ascending=False)


# ---------- 分级 / 仓位 / 消息 ----------
SIGNAL_TIERS = [
    ("🟩 强买入", 1.5, 99),
    ("🟢 买入",   0.75, 1.5),
    ("⚪️ 持有",   -0.75, 0.75),
    ("🟡 减仓",   -1.5, -0.75),
    ("🟥 清仓",   -99, -1.5),
]


def suggest_position(z: float, price: float, capital: float = 1_000_000) -> tuple[float, int]:
    if price <= 0:
        return 0.0, 0
    if z >= 1.5:
        pct = min(0.05, 0.04 + (z - 1.5) * 0.02)
    elif z >= 1.0:
        pct = 0.03
    elif z >= 0.75:
        pct = 0.02
    else:
        return 0.0, 0
    shares = int(capital * pct / price / 100) * 100
    return pct, max(0, shares)


def build_report(sig: pd.DataFrame, name_map: dict,
                  capital: float = 1_000_000) -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    z_med = sig["alpha_z"].median()
    up = (sig["alpha_z"] >= 1.0).sum()
    dn = (sig["alpha_z"] <= -1.0).sum()

    lines = [
        f"⚡ **复合 Alpha v2 — {today}**",
        "",
        f"Universe: {len(sig)} 只 | 参考资金 ¥{capital/1e4:.0f} 万",
        f"因子: 7 反转(0.40) + 6 打板(0.40) + 5 席位(0.20) = 18 因子",
        f"截面: z 中位 {z_med:+.2f}  强正 {up}  强负 {dn}",
        "",
    ]

    strong_buy = sig[sig["alpha_z"] >= 1.5]
    if len(strong_buy):
        lines.append("**🟩 强买入 (z ≥ 1.5)**")
        for code, r in strong_buy.head(5).iterrows():
            pct, sh = suggest_position(r["alpha_z"], r["latest_close"], capital)
            lines.append(f"  • {name_map.get(code, code)} ¥{r['latest_close']:.2f}  "
                         f"z={r['alpha_z']:+.2f}  建议 {pct*100:.0f}%仓 ≈ {sh} 股  "
                         f"[{r['cat_sign']}{r['top_category']}]")
        lines.append("")

    buy = sig[(sig["alpha_z"] >= 0.75) & (sig["alpha_z"] < 1.5)]
    if len(buy):
        lines.append(f"**🟢 买入 ({len(buy)} 只, 0.75 ≤ z < 1.5)**")
        for code, r in buy.head(8).iterrows():
            pct, sh = suggest_position(r["alpha_z"], r["latest_close"], capital)
            lines.append(f"  • {name_map.get(code, code)} ¥{r['latest_close']:.2f}  "
                         f"z={r['alpha_z']:+.2f}  建议 {pct*100:.0f}%仓 ≈ {sh} 股  "
                         f"[{r['cat_sign']}{r['top_category']}]")
        lines.append("")

    strong_sell = sig[sig["alpha_z"] <= -1.5]
    if len(strong_sell):
        lines.append(f"**🟥 清仓 ({len(strong_sell)} 只, z ≤ -1.5)**")
        for code, r in strong_sell.tail(5).iterrows():
            lines.append(f"  • {name_map.get(code, code)} ¥{r['latest_close']:.2f}  "
                         f"z={r['alpha_z']:+.2f}  [{r['cat_sign']}{r['top_category']}]")
        lines.append("")

    reduce_ = sig[(sig["alpha_z"] > -1.5) & (sig["alpha_z"] <= -0.75)]
    if len(reduce_):
        lines.append(f"**🟡 减仓 ({len(reduce_)} 只, -1.5 < z ≤ -0.75)**")
        for code, r in reduce_.head(5).iterrows():
            lines.append(f"  • {name_map.get(code, code)} ¥{r['latest_close']:.2f}  "
                         f"z={r['alpha_z']:+.2f}  [{r['cat_sign']}{r['top_category']}]")
        lines.append("")

    lines.append("**📊 大类主导分布**")
    lines.append(f"  反转: {(sig['top_category']=='反转').sum()} 只   "
                 f"打板: {(sig['top_category']=='打板').sum()} 只   "
                 f"席位: {(sig['top_category']=='席位').sum()} 只")
    lines.append("")
    lines.append("⚠️ v2 加入打板/情绪(STREAK_UP/BOOM_RATE/LU_COUNT) + 游资席位 "
                 "(SEAT_NET_JIGOU/SEAT_PURE_HOTMONEY). 决策参考 ≠ 买卖建议.")
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


# ---------- 主 ----------
def fetch_minute_data(codes: list[str], days_back: int = 30) -> pd.DataFrame:
    """拉 5 分钟 K (带缓存)."""
    today = pd.Timestamp.today().normalize()
    start = (today - pd.Timedelta(days=days_back)).strftime("%Y%m%d")
    end = today.strftime("%Y%m%d")
    cache_path = CACHE / f"watchlist_minute_{end}.parquet"

    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        cached = set(df["code"].unique())
        missing = [c for c in codes if c not in cached]
        if not missing:
            return df[df["code"].isin(codes)]
        extra = bulk_fetch_minute(missing, klt=5, start=start, end=end,
                                   progress=False)
        if not extra.empty:
            df = pd.concat([df, extra], ignore_index=True)
            df.to_parquet(cache_path)
        return df[df["code"].isin(codes)]

    print(f"  拉 {len(codes)} 只 5 分钟 K ({start}~{end}), 预计 1-3 分钟...")
    df = bulk_fetch_minute(codes, klt=5, start=start, end=end, progress=False)
    if df.empty:
        return df
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path)
    return df


def prepare_minute_for_intraday(minute_df: pd.DataFrame) -> pd.DataFrame:
    """把 bulk_fetch_minute 的 datetime 拆成 date/time (compute_real_intraday_alpha 需要)."""
    df = minute_df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = df["datetime"].dt.date
    df["time"] = df["datetime"].dt.time
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run-lark", action="store_true")
    ap.add_argument("--force-refetch", action="store_true")
    ap.add_argument("--use-minute", action="store_true",
                    help="启用 5 分钟级日内因子 (耗时 1-3 分钟)")
    args = ap.parse_args()

    print(f"\n{'='*64}\n  ⚡ Watchlist Signal v2 {datetime.now():%Y-%m-%d %H:%M}\n{'='*64}")

    wl = load_watchlist()
    codes = [w["code"] for w in wl]
    name_map = {w["code"]: w["name"] for w in wl}
    print(f"  watchlist: {len(codes)} 只")

    if args.force_refetch:
        for p in CACHE.glob("watchlist_kline_*.parquet"):
            p.unlink()

    daily = fetch_watchlist_kline(codes, days_back=180)
    if daily.empty:
        print("❌ 数据空"); return 1
    print(f"  日 K: {len(daily)} 行, {daily['code'].nunique()} 只")

    # 反转因子
    rev_df = compute_reversal_panel(daily)
    print(f"  反转因子: {rev_df.shape}")

    # 打板因子
    asof = daily["date"].max()
    try:
        limit_panel = compute_limit_alpha(daily)
        limit_slice = latest_slice(limit_panel, asof)
        print(f"  打板因子面板: {limit_panel.shape}, 当日截面 {limit_slice.shape}")
    except Exception as e:
        print(f"  ⚠️ 打板因子失败 ({e}), 用空面板")
        limit_slice = pd.DataFrame()

    # 席位因子
    lhb_df = load_lhb()
    if lhb_df is not None:
        try:
            trading_dates = pd.DatetimeIndex(sorted(daily["date"].unique()))
            seat_panel = compute_seat_alpha(lhb_df, trading_dates)
            seat_slice = latest_slice(seat_panel, asof)
            print(f"  席位因子面板: {seat_panel.shape}, 当日截面 {seat_slice.shape}")
        except Exception as e:
            print(f"  ⚠️ 席位因子失败 ({e}), 用空面板")
            seat_slice = pd.DataFrame()
    else:
        print("  ⚠️ 未找到龙虎榜缓存, 席位因子=0")
        seat_slice = pd.DataFrame()

    # 分钟级日内因子 (可选)
    intraday_slice = pd.DataFrame()
    if args.use_minute:
        try:
            minute_df = fetch_minute_data(codes, days_back=30)
            if minute_df.empty:
                print("  ⚠️ 分钟 K 空, 跳过日内因子")
            else:
                minute_df = prepare_minute_for_intraday(minute_df)
                print(f"  分钟 K: {len(minute_df)} 行, {minute_df['code'].nunique()} 只")
                intraday_panel = compute_real_intraday_alpha(minute_df)
                intraday_slice = latest_slice(intraday_panel, asof)
                print(f"  日内因子面板: {intraday_panel.shape}, 当日截面 {intraday_slice.shape}")
        except Exception as e:
            print(f"  ⚠️ 日内因子失败 ({e}), 跳过")
            intraday_slice = pd.DataFrame()

    # 合成
    sig = synthesize_composite(rev_df, limit_slice, seat_slice,
                                intraday_slice if args.use_minute else None)
    print(f"\n  复合 alpha_z Top 5:")
    for code, r in sig.head(5).iterrows():
        print(f"    {name_map.get(code, code):<8} ¥{r['latest_close']:>7.2f}  "
              f"z={r['alpha_z']:+.2f}  [{r['cat_sign']}{r['top_category']}]  "
              f"rev={r.get('rev_score', 0):+.2f} lim={r.get('limit_score', 0):+.2f} "
              f"seat={r.get('seat_score', 0):+.2f}")
    print(f"\n  Bottom 5:")
    for code, r in sig.tail(5).iterrows():
        print(f"    {name_map.get(code, code):<8} ¥{r['latest_close']:>7.2f}  "
              f"z={r['alpha_z']:+.2f}  [{r['cat_sign']}{r['top_category']}]")

    # 报告
    report = build_report(sig, name_map)
    print(f"\n{'='*64}\n飞书消息:\n{'='*64}")
    print(report)

    if not args.dry_run_lark:
        send_lark(report)

    # 落盘
    out_dir = PAPER / "watchlist_signals_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{datetime.now():%Y-%m-%d}.csv"
    score_cols = ["latest_close", "alpha_z", "top_category", "cat_sign",
                  "rev_score", "limit_score", "seat_score"]
    if "intraday_score" in sig.columns:
        score_cols.append("intraday_score")
    sig_out = sig[score_cols].copy()
    sig_out.insert(0, "name", sig_out.index.map(name_map))
    sig_out.to_csv(out_path)
    print(f"\n  CSV: {out_path}")


if __name__ == "__main__":
    main()
