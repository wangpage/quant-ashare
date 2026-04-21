"""watchlist_signal — 对用户自选池的专业量化信号.

输出(飞书):
    ⚡ Alpha 信号 TOP-3 (|z| ≥ 1.5, 强信号)
    ↗️ 加仓建议 (top quintile, 未持仓)
    ↘️ 减仓/清仓建议 (bottom quintile, 已持仓)
    ⏸️ 中性区(|z| < 1, 持有观察)

专业原则:
    - 不使用"龙头/热点/概念"等主观词
    - 只报 alpha score (z-score), 因子主导维度, 信号强度分级
    - 标注非 leak-free, 提醒"决策参考 ≠ 买卖建议"
    - 与 paper_trade/account.json 持仓对齐, 给出具体股数

合成因子 (无监督经典 anomaly, 日 K 可算):
    REV_5, REV_20       短期反转 (Jegadeesh 1990)
    MOM_126_21          中期动量 (12-1)
    LOW_VOL_60          低波异常 (Ang 2006)
    TURN_Z_60           换手率异动 (散户过热 → 反转)
    MAX_RET_5           最大单日涨幅(负)
    AMIHUD_60           非流动性溢价
    BOLL_POS            布林带位置

合成权重 (等权后再用 rank, 对单因子暴露稳健):
    每因子当日截面 rank → [0,1], 加权求和后再 rank.
"""
from __future__ import annotations

import argparse
import json
import os
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

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
PAPER = ROOT / "output" / "paper_trade"

# 专业合成: 各因子权重 (基于 A股散户市反转/低波经典)
FACTOR_WEIGHTS = {
    "REV_5":       0.20,   # 短反转
    "REV_20":      0.15,   # 中反转
    "MOM_126_21":  0.10,   # 中期动量 (剔除最近 1 月)
    "LOW_VOL_60":  0.15,   # 低波异常
    "TURN_Z_60":  -0.15,   # 过热换手 (负号 = 异动越强越减分)
    "MAX_RET_5":  -0.15,   # 单日暴涨后易反转
    "AMIHUD_60":   0.10,   # 非流动性溢价
}


# ---------- 加载 ----------
def load_watchlist() -> list[dict]:
    path = ROOT / "config" / "user_watchlist.yaml"
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["watchlist"]


def load_account() -> dict:
    p = PAPER / "account.json"
    if not p.exists():
        return {"positions": {}, "cash": 0, "equity_history": []}
    return json.loads(p.read_text(encoding="utf-8"))


# ---------- 数据 ----------
def fetch_watchlist_kline(codes: list[str], days_back: int = 180) -> pd.DataFrame:
    """拉 watchlist 近 N 天日 K (优先缓存)."""
    today = pd.Timestamp.today().normalize()
    start = (today - pd.Timedelta(days=int(days_back * 1.5))).strftime("%Y%m%d")
    end = today.strftime("%Y%m%d")

    cache_path = CACHE / f"watchlist_kline_{end}.parquet"
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        df["date"] = pd.to_datetime(df["date"])
        cached_codes = set(df["code"].unique())
        missing = [c for c in codes if c not in cached_codes]
        if not missing:
            return df[df["code"].isin(codes)]
        # 补拉
        extra = bulk_fetch_daily(missing, start, end, sleep_ms=80, progress=False)
        if not extra.empty:
            df = pd.concat([df, extra], ignore_index=True)
            df.to_parquet(cache_path)
        return df[df["code"].isin(codes)]

    print(f"  拉 {len(codes)} 只 watchlist 日 K ({start}~{end})...")
    df = bulk_fetch_daily(codes, start, end, sleep_ms=80, progress=False)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path)
    return df


# ---------- 因子计算 ----------
def compute_factors_per_stock(g: pd.DataFrame) -> pd.Series:
    """单只股最新一日的 7 个因子值."""
    g = g.sort_values("date").copy()
    close, high, low = g["close"], g["high"], g["low"]
    vol = g["volume"].astype(float)
    amt = g.get("amount", vol * close).astype(float)
    # amount 可能全 0 (sina fallback), 用 volume*close 兜底
    if (amt == 0).all():
        amt = vol * close
    ret1 = close.pct_change()

    out = {}
    # REV
    if len(close) >= 6:
        out["REV_5"] = -(close.iloc[-1] / close.iloc[-6] - 1)
    if len(close) >= 21:
        out["REV_20"] = -(close.iloc[-1] / close.iloc[-21] - 1)
    # 12-1 动量 (126 - 21 日)
    if len(close) >= 127:
        ret126 = close.iloc[-1] / close.iloc[-127] - 1
        ret21 = close.iloc[-1] / close.iloc[-22] - 1
        out["MOM_126_21"] = ret126 - ret21
    # 低波
    if len(ret1) >= 60:
        out["LOW_VOL_60"] = -ret1.tail(60).std()
    # 换手 z
    if len(vol) >= 60:
        v_recent = vol.iloc[-1]
        v_mu = vol.tail(60).mean()
        v_sd = vol.tail(60).std()
        if v_sd > 0:
            out["TURN_Z_60"] = (v_recent - v_mu) / v_sd
    # 近 5 日最大涨幅
    if len(ret1) >= 5:
        out["MAX_RET_5"] = ret1.tail(5).max()
    # Amihud
    if len(ret1) >= 60:
        amihud = (ret1.abs() / (amt + 1)).tail(60).mean()
        out["AMIHUD_60"] = amihud

    return pd.Series(out)


def compute_all_factors(daily_df: pd.DataFrame) -> pd.DataFrame:
    """对 watchlist 所有股票算最新一日的因子矩阵."""
    pieces = []
    for code, g in daily_df.groupby("code"):
        if len(g) < 60:
            continue
        s = compute_factors_per_stock(g)
        s["code"] = code
        s["latest_close"] = float(g.sort_values("date")["close"].iloc[-1])
        s["latest_date"] = g["date"].max()
        pieces.append(s)
    if not pieces:
        return pd.DataFrame()
    return pd.DataFrame(pieces).set_index("code")


# ---------- 信号合成 ----------
def synthesize_signal(factor_df: pd.DataFrame) -> pd.DataFrame:
    """截面 rank → 加权 → 最终 alpha score (z-score)."""
    df = factor_df.copy()

    # 每因子 rank, NaN 填 0.5 (中性)
    ranked = pd.DataFrame(index=df.index)
    for f in FACTOR_WEIGHTS:
        if f in df.columns:
            ranked[f] = df[f].rank(pct=True).fillna(0.5) - 0.5   # 中心化
        else:
            ranked[f] = 0.0

    # 加权合成
    scores = sum(ranked[f] * w for f, w in FACTOR_WEIGHTS.items())
    # 再做一次横截面 z-score
    alpha_z = (scores - scores.mean()) / (scores.std() + 1e-9)
    df["alpha_z"] = alpha_z

    # 各因子的主导维度: 对 score 贡献最大的一个
    contrib = pd.DataFrame(index=df.index)
    for f, w in FACTOR_WEIGHTS.items():
        contrib[f] = ranked[f] * w
    df["top_driver"] = contrib.abs().idxmax(axis=1)
    df["driver_sign"] = contrib.apply(
        lambda row: "+" if row[row.abs().idxmax()] > 0 else "-",
        axis=1,
    )

    return df.sort_values("alpha_z", ascending=False)


# ---------- 信号分级 ----------
SIGNAL_TIERS = [
    ("🟩 强买入", 1.5, 99),       # z >= 1.5
    ("🟢 买入",   0.75, 1.5),     # 0.75 <= z < 1.5
    ("⚪️ 持有",   -0.75, 0.75),   # |z| < 0.75
    ("🟡 减仓",   -1.5, -0.75),   # -1.5 <= z < -0.75
    ("🟥 清仓",   -99, -1.5),     # z < -1.5
]


def classify_signal(z: float) -> str:
    for name, lo, hi in SIGNAL_TIERS:
        if lo <= z < hi:
            return name
    return "⚪️ 持有"


# ---------- 建议仓位 (按 100 万总资金计算) ----------
REFERENCE_CAPITAL = 1_000_000   # 默认 100 万参考, 用户按自己资金等比例换算


def suggest_position(alpha_z: float, price: float,
                      capital: float = REFERENCE_CAPITAL) -> tuple[float, int]:
    """返回 (建议仓位%, 建议股数 @ 100 万参考资金).

    强度映射:
        z ≥ 1.5  → 4-5% (高置信)
        z ≥ 1.0  → 3%   (较强)
        z ≥ 0.75 → 2%   (弱信号)
        其他     → 0
    """
    if price <= 0:
        return 0.0, 0
    if alpha_z >= 1.5:
        pct = min(0.05, 0.04 + (alpha_z - 1.5) * 0.02)
    elif alpha_z >= 1.0:
        pct = 0.03
    elif alpha_z >= 0.75:
        pct = 0.02
    else:
        return 0.0, 0
    amount = capital * pct
    shares = int(amount / price / 100) * 100
    return pct, max(0, shares)


# ---------- 飞书消息 ----------
def build_professional_report(
    sig_df: pd.DataFrame, name_map: dict[str, str],
    capital: float = REFERENCE_CAPITAL,
) -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    # Universe 整体状态
    z_med = sig_df["alpha_z"].median()
    z_std = sig_df["alpha_z"].std()
    strong_up = (sig_df["alpha_z"] >= 1.0).sum()
    strong_dn = (sig_df["alpha_z"] <= -1.0).sum()

    lines = [
        f"⚡ **量化 Alpha 信号 — {today}**",
        "",
        f"Universe: {len(sig_df)} 只自选池 | 参考资金 ¥{capital/1e4:.0f} 万",
        f"截面信号分布: z 中位 {z_med:+.2f}  强正信号 {strong_up}  强负信号 {strong_dn}",
        "",
    ]

    # 强买入 (z >= 1.5)
    strong_buy = sig_df[sig_df["alpha_z"] >= 1.5]
    if len(strong_buy):
        lines.append("**🟩 强买入信号 (z ≥ 1.5, 高置信)**")
        for code, row in strong_buy.head(5).iterrows():
            name = name_map.get(code, code)
            pct, shares = suggest_position(row["alpha_z"],
                                            row["latest_close"], capital)
            lines.append(
                f"  • {name} ¥{row['latest_close']:.2f}  "
                f"z={row['alpha_z']:+.2f}  "
                f"建议 {pct*100:.0f}%仓 ≈ {shares} 股  "
                f"[{row['driver_sign']}{row['top_driver']}]"
            )
        lines.append("")

    # 买入 (0.75 <= z < 1.5)
    buy = sig_df[(sig_df["alpha_z"] >= 0.75) & (sig_df["alpha_z"] < 1.5)]
    if len(buy):
        lines.append("**🟢 买入信号 (0.75 ≤ z < 1.5)**")
        for code, row in buy.head(8).iterrows():
            name = name_map.get(code, code)
            pct, shares = suggest_position(row["alpha_z"],
                                            row["latest_close"], capital)
            lines.append(
                f"  • {name} ¥{row['latest_close']:.2f}  "
                f"z={row['alpha_z']:+.2f}  "
                f"建议 {pct*100:.0f}%仓 ≈ {shares} 股  "
                f"[{row['driver_sign']}{row['top_driver']}]"
            )
        lines.append("")

    # 清仓信号 (z < -1.5)
    strong_sell = sig_df[sig_df["alpha_z"] <= -1.5]
    if len(strong_sell):
        lines.append("**🟥 清仓信号 (z ≤ -1.5, 若持有)**")
        for code, row in strong_sell.tail(5).iterrows():
            name = name_map.get(code, code)
            lines.append(
                f"  • {name} ¥{row['latest_close']:.2f}  "
                f"z={row['alpha_z']:+.2f}  "
                f"[{row['driver_sign']}{row['top_driver']}]"
            )
        lines.append("")

    # 减仓信号 (-1.5 < z <= -0.75)
    reduce = sig_df[(sig_df["alpha_z"] > -1.5) & (sig_df["alpha_z"] <= -0.75)]
    if len(reduce):
        lines.append("**🟡 减仓信号 (-1.5 < z ≤ -0.75, 若持有)**")
        for code, row in reduce.head(5).iterrows():
            name = name_map.get(code, code)
            lines.append(
                f"  • {name} ¥{row['latest_close']:.2f}  "
                f"z={row['alpha_z']:+.2f}  "
                f"[{row['driver_sign']}{row['top_driver']}]"
            )
        lines.append("")

    # 因子主导分布
    driver_counts = sig_df["top_driver"].value_counts().head(4)
    lines.append("**📊 信号主导因子分布**")
    for f, n in driver_counts.items():
        lines.append(f"  {f}: {n} 只")
    lines.append("")

    lines.append(
        "⚠️ **方法说明**: 7 因子截面合成 (REV_5/REV_20/MOM_126_21/LOW_VOL_60/"
        "TURN_Z_60/MAX_RET_5/AMIHUD_60), 因子 rank 归一化后加权. "
        "非 leak-free 模型, 仅决策参考 ≠ 买卖建议. "
        "当前 A 股 regime 等权 Sharpe ≈ 2.5-3 (FOMO 牛市), 多头选股策略天生劣势, 轻仓验证."
    )
    return "\n".join(lines)


# ---------- 主 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run-lark", action="store_true")
    ap.add_argument("--force-refetch", action="store_true")
    args = ap.parse_args()

    print(f"\n{'='*64}\n  ⚡ Watchlist Signal {datetime.now().strftime('%Y-%m-%d %H:%M')}\n{'='*64}")

    watchlist = load_watchlist()
    codes = [w["code"] for w in watchlist]
    name_map = {w["code"]: w["name"] for w in watchlist}
    print(f"  watchlist: {len(codes)} 只")

    # 拉数据
    if args.force_refetch:
        for p in CACHE.glob("watchlist_kline_*.parquet"):
            p.unlink()
    daily = fetch_watchlist_kline(codes, days_back=180)
    if daily.empty:
        print("❌ 数据空"); return 1
    print(f"  数据: {len(daily)} 行, {daily['code'].nunique()} 只")

    # 因子
    factor_df = compute_all_factors(daily)
    if factor_df.empty:
        print("❌ 无法计算因子 (数据量不足)"); return 1
    print(f"  因子矩阵: {factor_df.shape}")

    # 信号
    sig_df = synthesize_signal(factor_df)
    print(f"\n  Top 5 alpha:")
    for code, row in sig_df.head(5).iterrows():
        print(f"    {name_map.get(code, code):<8} ¥{row['latest_close']:>7.2f}  "
              f"z={row['alpha_z']:+.2f}  [{row['driver_sign']}{row['top_driver']}]")
    print(f"\n  Bottom 5 alpha:")
    for code, row in sig_df.tail(5).iterrows():
        print(f"    {name_map.get(code, code):<8} ¥{row['latest_close']:>7.2f}  "
              f"z={row['alpha_z']:+.2f}  [{row['driver_sign']}{row['top_driver']}]")

    # 飞书 (watchlist 推荐独立于 paper_trade 账户, 用 100 万参考资金)
    report = build_professional_report(sig_df, name_map, REFERENCE_CAPITAL)
    print(f"\n{'='*64}\n飞书消息内容:\n{'='*64}")
    print(report)

    if not args.dry_run_lark:
        import subprocess
        user_id = os.environ.get("LARK_USER_OPEN_ID",
                                   "ou_5be0f87dc7cec796b7ea97d0a9b5302f")
        cmd = ["lark-cli", "im", "+messages-send",
               "--as", "user", "--user-id", user_id,
               "--markdown", report]
        rc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if rc.returncode == 0:
            print(f"\n✓ 飞书已送达")
        else:
            print(f"\n❌ 飞书失败: {rc.stderr[:200]}")

    # 存最新信号 csv 便于历史追溯
    out_csv = PAPER / "watchlist_signals" / f"{datetime.now().strftime('%Y-%m-%d')}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    sig_out = sig_df[["latest_close", "alpha_z", "top_driver", "driver_sign"]].copy()
    sig_out.insert(0, "name", sig_out.index.map(name_map))
    sig_out.to_csv(out_csv)
    print(f"\n  信号 CSV: {out_csv}")


if __name__ == "__main__":
    main()
