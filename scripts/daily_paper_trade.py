"""Paper trade 基础设施 - 每日自动化 V6 信号生成 + 持仓追踪.

流程 (每交易日收盘后运行):
    1. 增量拉 kline + lhb + insider 到今天
    2. 跑 V6 完整 pipeline: IC 聚类 + 自适应极性 + LightGBM
    3. 生成 top-25 今日信号 (Qwen 解释)
    4. 对比昨日持仓, 计算实际 P&L
    5. 追加到 output/paper_trade/pnl_log.csv + signals/YYYY-MM-DD.md

用法:
    python3 scripts/daily_paper_trade.py                        # 跑今天
    python3 scripts/daily_paper_trade.py --asof 2026-04-18      # 跑历史某日 (回测模式)

依赖环境变量:
    DASHSCOPE_API_KEY (可选, 没有则跳过 Qwen 解释)
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np

env = Path(__file__).resolve().parent.parent / ".env"
if env.exists():
    for line in env.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from scripts.run_real_research_v5 import (
    make_conditional_label, ic_cluster_select, _fit_model,
)
from data_adapter.lhb import (
    build_lhb_features, LHB_FACTOR_NAMES, LHB_B2_FACTOR_NAMES,
)
from data_adapter.insider import build_insider_features, INSIDER_FACTOR_NAMES
from factors.alpha_pandas import compute_pandas_alpha
from factors.alpha_reversal import compute_advanced_alpha
from factors.alpha_limit import compute_limit_alpha, LIMIT_FACTOR_NAMES
from factors.adaptive_polarity import apply_adaptive_polarity

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
PAPER = ROOT / "output" / "paper_trade"
PAPER.mkdir(parents=True, exist_ok=True)
(PAPER / "signals").mkdir(exist_ok=True)
HORIZON = 30
TOP_K = 25


def generate_signal_for_date(asof: pd.Timestamp):
    """生成指定日期的 top-K 信号."""
    start, end = "20230101", asof.strftime("%Y%m%d")

    # 加载缓存 (要求已有最新数据)
    kline_candidates = sorted(CACHE.glob(f"kline_{start}_*_n500.parquet"),
                               key=lambda p: p.stat().st_mtime)
    if not kline_candidates:
        raise FileNotFoundError("无 kline 缓存, 先跑 run_real_research_v5")
    daily = pd.read_parquet(kline_candidates[-1])
    print(f"kline {len(daily)} 行, 覆盖到 {daily['date'].max()}")

    lhb_candidates = [p for p in CACHE.glob("lhb_2*.parquet")
                       if "taxonomy" not in p.name]
    lhb_df = pd.read_parquet(sorted(lhb_candidates, key=lambda p: p.stat().st_mtime)[-1])
    ins_df = pd.read_parquet(sorted(CACHE.glob("insider_*.parquet"),
                                     key=lambda p: p.stat().st_mtime)[-1])

    # 因子
    feat_tech = compute_pandas_alpha(daily)
    feat_rev = compute_advanced_alpha(daily)
    feat_limit = compute_limit_alpha(daily)
    feat_combo = feat_tech.join(feat_rev, how="outer").join(feat_limit, how="outer")
    trading_dates = pd.DatetimeIndex(sorted(daily["date"].unique()))
    feat_lhb = build_lhb_features(lhb_df, trading_dates)
    feat_combo = feat_combo.join(feat_lhb, how="left")
    feat_ins = build_insider_features(ins_df, trading_dates)
    feat_combo = feat_combo.join(feat_ins, how="left")
    for f in (LHB_FACTOR_NAMES + LHB_B2_FACTOR_NAMES
              + INSIDER_FACTOR_NAMES + LIMIT_FACTOR_NAMES):
        if f in feat_combo.columns:
            feat_combo[f] = feat_combo[f].fillna(0)

    def _z(s):
        mu, sd = s.mean(), s.std()
        return (s - mu) / sd if sd > 0 else s * 0
    feat_z = feat_combo.groupby(level="date").transform(_z).clip(-3, 3).fillna(0)
    label = make_conditional_label(daily, horizon=HORIZON, dd_clip=0.25)

    # 只用 asof 之前的训练
    aligned = feat_z.join(label.rename("label"), how="inner")
    valid = aligned["label"].notna() & aligned.drop(columns="label").notna().all(axis=1)
    feat_valid = aligned.drop(columns="label")[valid]
    y_valid = aligned["label"][valid]

    dates = feat_valid.index.get_level_values("date").unique().sort_values()
    train_end = dates[dates < asof].max()
    if train_end is pd.NaT:
        raise ValueError(f"无 {asof.date()} 之前数据")

    # IC 聚类 - 只用训练
    tr_start = dates[max(0, dates.get_loc(train_end) - 251)]
    train_mask = (feat_valid.index.get_level_values("date") >= tr_start) & \
                  (feat_valid.index.get_level_values("date") <= train_end)
    selected = ic_cluster_select(feat_valid[train_mask], y_valid[train_mask],
                                  corr_threshold=0.6, min_ic=0.005)
    print(f"  精选因子 {len(selected)}")
    feat_sel = feat_valid[selected]

    # 自适应极性 (V6 最优参数)
    feat_adapt, weight_df = apply_adaptive_polarity(
        feat_sel, y_valid, horizon=HORIZON, window=90,
        z_threshold=0.8, z_cap=3.0, inertia=0.6, decay_lambda=0.0,
    )
    all_zero = (feat_adapt.abs().sum(axis=1) < 1e-9)
    feat_adapt = feat_adapt[~all_zero]
    y_adapt = y_valid.loc[feat_adapt.index]

    # 训练 + 预测今日截面
    X_tr = feat_adapt.loc[tr_start:train_end]
    y_tr = y_adapt.loc[X_tr.index]
    m = y_tr.notna() & X_tr.notna().all(axis=1)
    model = _fit_model(X_tr[m], y_tr[m])

    # 信号日 = asof 当日的截面 (若 asof 不在数据中, 取最近一天)
    avail_dates = feat_adapt.index.get_level_values("date").unique()
    signal_dates = avail_dates[avail_dates <= asof]
    if len(signal_dates) == 0:
        raise ValueError(f"无 ≤ {asof.date()} 的因子数据")
    sig_date = signal_dates.max()
    X_te = feat_adapt.xs(sig_date, level="date", drop_level=True)
    pred = pd.Series(model.predict(X_te.values), index=X_te.index)
    pred.index.name = "code"
    print(f"  信号日 {sig_date.date()}, 候选 {len(pred)} 只")

    # Top K
    top = pred.nlargest(TOP_K)
    return sig_date, top, daily, lhb_df


def get_stock_names() -> dict[str, str]:
    """从 sina snapshot 拉名字 (cached daily)."""
    try:
        from data_adapter.sina_universe import fetch_all_ashare
        df = fetch_all_ashare(max_pages=60)
        return df.set_index("code")["name"].to_dict()
    except Exception:
        return {}


async def qwen_explain(top: pd.Series, daily: pd.DataFrame,
                        names: dict[str, str]) -> dict[str, str]:
    """为 top 股票并发生成 Qwen 解释 (若 API 可用)."""
    if not os.environ.get("DASHSCOPE_API_KEY"):
        return {}
    try:
        from llm_layer.agents import _LLMBackend
        backend = _LLMBackend(backend="qwen", model="qwen-plus")
    except Exception:
        return {}

    SYS = """你是 A股量化研究员. 针对单只股票的模型打分 + 近期价格, 用 1 段话 (≤60 字) 说:
1) 支撑模型看好的 1-2 个理由
2) 1 个风险点
3) 仓位建议 (重仓/标配/观察/回避).
简洁不堆术语."""

    from functools import partial
    sem = asyncio.Semaphore(5)
    loop = asyncio.get_event_loop()

    async def one(code, score):
        name = names.get(code, code)
        try:
            k = daily[daily["code"] == code].sort_values("date").tail(10)
            r20 = (k["close"].iloc[-1] / k["close"].iloc[-1 if len(k) < 21 else -21]
                    - 1) if len(k) >= 2 else 0
            ctx = (f"{code} {name}, 打分 {score:+.4f}, "
                    f"最新价 {k['close'].iloc[-1]:.2f}, 近 10 日 {r20:+.2%}")
            async with sem:
                r = await loop.run_in_executor(
                    None, partial(backend.chat, f"{SYS}\n\n{ctx}", 150))
            return code, r.strip()
        except Exception as e:
            return code, f"(LLM err: {e})"

    results = await asyncio.gather(*[one(c, s) for c, s in top.items()])
    return dict(results)


def write_signal_report(sig_date, top, names, qwen_expls):
    """写 signals/YYYY-MM-DD.md."""
    lines = [f"# 🎯 Paper Trade 信号 — {sig_date.date()}\n",
              f"- 持仓规则: 月频调仓 (持 30 交易日)",
              f"- 策略: V6 B+ 自适应极性",
              f"- Top K: {TOP_K} 等权\n",
              "## 持仓清单\n",
              "| 排名 | 代码 | 名称 | 打分 | 建议 |",
              "|---|---|---|---|---|"]
    for i, (code, score) in enumerate(top.items(), 1):
        name = names.get(code, code)
        ex = qwen_expls.get(code, "").replace("|", "/").replace("\n", " ")[:80]
        lines.append(f"| {i} | {code} | {name} | {score:+.4f} | {ex} |")
    out_path = PAPER / "signals" / f"{sig_date.strftime('%Y-%m-%d')}.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def update_positions_log(sig_date, top, names):
    """更新持仓档案 - positions.csv.

    记录每次信号: (signal_date, code, name, score, rebalance_end)
    rebalance_end = signal_date + 30 交易日 (近似日历 45 天)
    """
    log_path = PAPER / "positions_log.csv"
    existing = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame()

    rows = []
    for i, (code, score) in enumerate(top.items(), 1):
        rows.append({
            "signal_date": sig_date.date(),
            "rank": i,
            "code": code,
            "name": names.get(code, code),
            "score": score,
            "target_exit": (sig_date + pd.Timedelta(days=45)).date(),
        })
    new = pd.DataFrame(rows)
    combined = pd.concat([existing, new], ignore_index=True) if len(existing) else new
    combined.to_csv(log_path, index=False)
    print(f"  持仓档案已更新: {log_path} ({len(combined)} 条历史)")


def evaluate_past_signals(daily: pd.DataFrame):
    """回看过去的 paper trade 信号, 核算实际表现."""
    log_path = PAPER / "positions_log.csv"
    if not log_path.exists():
        return

    log = pd.read_csv(log_path, parse_dates=["signal_date", "target_exit"])
    # 排除尚未到期的
    today = pd.Timestamp.today().normalize()
    closed = log[log["target_exit"] <= today]
    if closed.empty:
        print("\n  (无已到期持仓可评估)")
        return

    print(f"\n  📈 已到期持仓表现 ({len(closed)} 条):")
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily_idx = daily.set_index(["code", "date"])

    results = []
    for _, r in closed.iterrows():
        try:
            # T+1 open → T+30 close 近似 (没有 30 交易日映射, 取 signal+1 日 open, exit 日 close)
            kline = daily[daily["code"] == r["code"]]
            future = kline[kline["date"] > r["signal_date"]].head(30)
            if len(future) < 20:
                continue
            buy_p = future["open"].iloc[0]
            sell_p = future["close"].iloc[-1]
            ret = sell_p / buy_p - 1
            results.append({"signal_date": r["signal_date"].date(),
                             "code": r["code"], "return": ret,
                             "rank": r["rank"]})
        except Exception:
            continue

    if not results:
        return
    rdf = pd.DataFrame(results)
    print(f"  组合平均收益: {rdf['return'].mean():+.2%}")
    print(f"  胜率: {(rdf['return'] > 0).mean():.0%}")
    print(f"  最好/最差: {rdf['return'].max():+.2%} / {rdf['return'].min():+.2%}")


async def main_async(asof_str: str | None):
    asof = pd.Timestamp(asof_str) if asof_str else pd.Timestamp.today().normalize()
    print(f"\n{'='*64}\n  📊 Paper Trade Daily — asof {asof.date()}\n{'='*64}")

    sig_date, top, daily, lhb_df = generate_signal_for_date(asof)
    names = get_stock_names()

    print(f"\n[Qwen 解释] {len(top)} 只...")
    t0 = time.time()
    qwen_expls = await qwen_explain(top, daily, names)
    print(f"  完成 {len(qwen_expls)} 条, 耗时 {time.time()-t0:.1f}s")

    print("\n[输出报告]")
    rpt = write_signal_report(sig_date, top, names, qwen_expls)
    print(f"  信号报告: {rpt}")

    update_positions_log(sig_date, top, names)
    evaluate_past_signals(daily)

    # Top 5 终端预览
    print(f"\n--- Top 5 ({sig_date.date()}) ---")
    for i, (code, score) in enumerate(top.head(5).items(), 1):
        name = names.get(code, code)
        print(f"{i}. {code} {name}  score={score:+.4f}")
        if code in qwen_expls:
            print(f"   {qwen_expls[code][:100]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default=None, help="回测到某日, 默认今天")
    args = ap.parse_args()
    asyncio.run(main_async(args.asof))


if __name__ == "__main__":
    main()
