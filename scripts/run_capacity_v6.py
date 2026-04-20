"""V6 组合资金规模测试 (P2) - 策略容量评估.

问题: 策略在多少资金规模下还能盈利?
    - 100 万 → 滑点 ~5 bps, 基本不影响
    - 1000 万 → 滑点 ~20 bps, 轻微拖累
    - 5000 万 → 滑点 ~50 bps, 吃掉相当部分 alpha
    - 1 亿 → 滑点 ~100 bps, 可能崩盘

用 hold-out 期 (2025-01 ~ 2026-04) 测试不同资金规模下的 IR.

复用 V6 最优参数: z_threshold=0.8, window=90, inertia=0.6.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from scripts.run_real_research_v5 import (
    make_conditional_label, ic_cluster_select, _fit_model, backtest_v4,
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
HORIZON = 30
TRAIN_END = pd.Timestamp("2024-12-31")


def main():
    print(f"\n{'='*70}\n  V6 资金规模容量测试 {time.strftime('%Y-%m-%d %H:%M')}\n{'='*70}")

    start, end = "20230101", "20260420"
    daily = pd.read_parquet(CACHE / f"kline_{start}_{end}_n500.parquet")
    lhb_df = pd.read_parquet(CACHE / f"lhb_{start}_{end}.parquet")
    ins_df = pd.read_parquet(CACHE / f"insider_{start}_{end}.parquet")

    # 因子
    print("\n[1/4] 计算因子...")
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

    # IC 聚类 (仅训练集)
    print("\n[2/4] IC 聚类 (仅训练集)...")
    aligned = feat_z.join(label.rename("label"), how="inner")
    valid = aligned["label"].notna() & aligned.drop(columns="label").notna().all(axis=1)
    feat_valid = aligned.drop(columns="label")[valid]
    y_valid = aligned["label"][valid]
    train_mask = feat_valid.index.get_level_values("date") <= TRAIN_END
    selected = ic_cluster_select(feat_valid[train_mask], y_valid[train_mask],
                                  corr_threshold=0.6, min_ic=0.005)
    feat_sel = feat_valid[selected]

    # 自适应极性 (V6 最优参数)
    print("\n[3/4] 自适应极性...")
    feat_adapt, _ = apply_adaptive_polarity(
        feat_sel, y_valid, horizon=HORIZON, window=90,
        z_threshold=0.8, z_cap=3.0, inertia=0.6, decay_lambda=0.0,
    )
    all_zero = (feat_adapt.abs().sum(axis=1) < 1e-9)
    feat_adapt = feat_adapt[~all_zero]
    y_adapt = y_valid.loc[feat_adapt.index]

    # 训练 + 预测
    dates = feat_adapt.index.get_level_values("date").unique().sort_values()
    train_dates = dates[dates <= TRAIN_END]
    test_dates = dates[dates > TRAIN_END]
    tr_s = train_dates[-252]
    X_tr = feat_adapt.loc[tr_s:TRAIN_END]
    y_tr = y_adapt.loc[X_tr.index]
    m = y_tr.notna() & X_tr.notna().all(axis=1)
    model = _fit_model(X_tr[m], y_tr[m])
    X_te = feat_adapt.loc[test_dates[0]:test_dates[-1]]
    pred = pd.Series(model.predict(X_te.values), index=X_te.index)

    # 不同资金规模测试
    print("\n[4/4] 资金规模扫描...")
    capitals = [
        (1e6,   "100 万"),
        (5e6,   "500 万"),
        (1e7,   "1000 万"),
        (3e7,   "3000 万"),
        (5e7,   "5000 万"),
        (1e8,   "1 亿"),
        (3e8,   "3 亿"),
    ]

    results = []
    for cap, label_str in capitals:
        stats = backtest_v4(pred, daily, top_ratio=0.05,
                             rebalance_days=30,
                             capital_yuan=cap, vol_target=None)
        if "error" in stats:
            continue
        results.append({
            "capital": cap,
            "label": label_str,
            "sharpe": stats["sharpe"],
            "annual_return": stats["annual_return"],
            "excess_return": stats["excess_return"],
            "info_ratio": stats["info_ratio"],
            "avg_impact_bps": stats["avg_impact_bps"],
            "excess_max_dd": stats["excess_max_dd"],
        })

    # 汇总
    print("\n" + "="*70)
    print("  📊 资金规模 vs 收益对比 (hold-out 期 2025-01 ~ 2026-04)")
    print('='*70)
    df = pd.DataFrame(results)
    print(f"\n{'规模':<10} {'滑点(bps)':<12} {'年化收益':<12} {'超额年化':<12} "
          f"{'信息比率':<10} {'Sharpe':<8}")
    print("-" * 70)
    for _, r in df.iterrows():
        print(f"{r['label']:<10} {r['avg_impact_bps']:>10.1f}  "
              f"{r['annual_return']:>10.2%}  {r['excess_return']:>10.2%}  "
              f"{r['info_ratio']:>8.2f}  {r['sharpe']:>6.2f}")

    # 容量断点 (IR 跌破 0.3 的规模)
    print("\n" + "="*70)
    print("  容量评估")
    print('='*70)
    breakpoint_cap = None
    for _, r in df.iterrows():
        if r["info_ratio"] < 0.3:
            breakpoint_cap = r["label"]
            break
    if breakpoint_cap:
        print(f"  容量上限 (IR 跌破 0.3): 约 {breakpoint_cap}")
    else:
        print(f"  3 亿以下 IR 均 ≥ 0.3, 策略容量较大")

    # 保存
    out = ROOT / "output" / f"capacity_v6_{time.strftime('%Y%m%d_%H%M')}.md"
    out.parent.mkdir(exist_ok=True)
    lines = ["# V6 资金规模容量测试\n",
              "| 规模 | 滑点(bps) | 年化收益 | 超额年化 | 信息比率 | Sharpe |",
              "|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['label']} | {r['avg_impact_bps']:.1f} | "
            f"{r['annual_return']:.2%} | {r['excess_return']:.2%} | "
            f"{r['info_ratio']:.2f} | {r['sharpe']:.2f} |"
        )
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n报告: {out}")


if __name__ == "__main__":
    main()
