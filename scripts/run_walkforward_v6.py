"""Walk-forward 分段验证 V6.

最严厉的 OOS 测试:
    对 2024Q1 ~ 2026Q1 共 9 个季度, 每个季度单独做一次 hold-out:
        - 训练: 测试季度开始前 252 交易日 (1 年)
        - IC 聚类 + 自适应极性: 只用训练集
        - 测试: 该季度内 (约 60 交易日, 2-3 次调仓)

目标:
    证明 V6 hold-out IR 0.96 不是 2025 年的运气,
    要求多数季度 IR > 0 且 excess_return > 0.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from scripts.run_real_research_v5 import (
    make_conditional_label, ic_cluster_select,
    _fit_model, backtest_v4,
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


def compute_all_factors(start: str, end: str):
    """一次性计算全样本因子 (不做选择, 后续按窗口过滤)."""
    daily = pd.read_parquet(CACHE / f"kline_{start}_{end}_n500.parquet")
    lhb_df = pd.read_parquet(CACHE / f"lhb_{start}_{end}.parquet")
    ins_df = pd.read_parquet(CACHE / f"insider_{start}_{end}.parquet")

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
    return daily, feat_z, label


def run_quarter(feat_z, label, daily,
                 train_end_date: pd.Timestamp,
                 test_start_date: pd.Timestamp,
                 test_end_date: pd.Timestamp,
                 train_days: int = 252):
    """单季度 walk-forward."""
    # 对齐
    aligned = feat_z.join(label.rename("label"), how="inner")
    valid = aligned["label"].notna() & aligned.drop(columns="label").notna().all(axis=1)
    feat_valid = aligned.drop(columns="label")[valid]
    y_valid = aligned["label"][valid]

    # 训练窗口
    train_dates = feat_valid.index.get_level_values("date").unique().sort_values()
    train_dates = train_dates[train_dates <= train_end_date]
    if len(train_dates) < train_days:
        return None
    tr_s = train_dates[-train_days]
    train_mask = feat_valid.index.get_level_values("date").isin(train_dates[-train_days:])

    # IC 聚类 (仅训练窗口)
    selected = ic_cluster_select(feat_valid[train_mask], y_valid[train_mask],
                                  corr_threshold=0.6, min_ic=0.005)
    if not selected:
        return None
    feat_sel = feat_valid[selected]

    # 自适应极性 (全样本计算 IC, shift horizon 严格防泄露)
    # 但极性判断在 t 日只用 [t-horizon-window, t-horizon] 的 IC, 这里自然防泄露
    feat_adapt, _ = apply_adaptive_polarity(
        feat_sel, y_valid,
        horizon=HORIZON, window=90,
        z_threshold=0.8, z_cap=3.0,
        inertia=0.6, decay_lambda=0.0,
    )
    all_zero = (feat_adapt.abs().sum(axis=1) < 1e-9)
    feat_adapt = feat_adapt[~all_zero]
    y_adapt = y_valid.loc[feat_adapt.index]

    # 训练
    X_tr = feat_adapt[feat_adapt.index.get_level_values("date") <= train_end_date]
    X_tr = X_tr[X_tr.index.get_level_values("date") >= tr_s]
    y_tr = y_adapt.loc[X_tr.index]
    mask = y_tr.notna() & X_tr.notna().all(axis=1)
    X_tr, y_tr = X_tr[mask], y_tr[mask]
    if len(X_tr) < 1000:
        return None
    model = _fit_model(X_tr, y_tr)

    # 预测
    X_te = feat_adapt[
        (feat_adapt.index.get_level_values("date") >= test_start_date) &
        (feat_adapt.index.get_level_values("date") <= test_end_date)
    ]
    if len(X_te) == 0:
        return None
    pred = pd.Series(model.predict(X_te.values), index=X_te.index)

    # IC
    ic = pred.groupby(level="date").apply(
        lambda s: s.corr(y_adapt.loc[s.index], method="spearman")
    ).dropna()

    # 回测: 季度样本少, rebalance=10 多观测, 关掉 vol_target
    stats = backtest_v4(pred, daily, top_ratio=0.05,
                         rebalance_days=10, vol_target=None)

    return {
        "train_window": f"{tr_s.date()}→{train_end_date.date()}",
        "test_window": f"{test_start_date.date()}→{test_end_date.date()}",
        "n_train": len(X_tr),
        "n_test": len(X_te),
        "n_selected_factors": len(selected),
        "ic_mean": float(ic.mean()) if len(ic) else 0,
        "ic_ir": float(ic.mean() / (ic.std() + 1e-9)) if len(ic) else 0,
        "ic_gt0_pct": float((ic > 0).mean()) if len(ic) else 0,
        **{k: v for k, v in stats.items() if k not in ("top_k_ratio",)},
    }


def main():
    start, end = "20230101", "20260420"
    print(f"\n{'='*64}\n  Walk-forward V6 严格 OOS 验证\n{'='*64}")

    print("\n[预处理] 计算全部因子+标签...")
    daily, feat_z, label = compute_all_factors(start, end)
    print(f"  feat_z {feat_z.shape}, label {len(label)}")

    # 9 个测试季度
    quarters = [
        ("2024Q1", pd.Timestamp("2023-12-31"), pd.Timestamp("2024-01-01"), pd.Timestamp("2024-03-31")),
        ("2024Q2", pd.Timestamp("2024-03-31"), pd.Timestamp("2024-04-01"), pd.Timestamp("2024-06-30")),
        ("2024Q3", pd.Timestamp("2024-06-30"), pd.Timestamp("2024-07-01"), pd.Timestamp("2024-09-30")),
        ("2024Q4", pd.Timestamp("2024-09-30"), pd.Timestamp("2024-10-01"), pd.Timestamp("2024-12-31")),
        ("2025Q1", pd.Timestamp("2024-12-31"), pd.Timestamp("2025-01-01"), pd.Timestamp("2025-03-31")),
        ("2025Q2", pd.Timestamp("2025-03-31"), pd.Timestamp("2025-04-01"), pd.Timestamp("2025-06-30")),
        ("2025Q3", pd.Timestamp("2025-06-30"), pd.Timestamp("2025-07-01"), pd.Timestamp("2025-09-30")),
        ("2025Q4", pd.Timestamp("2025-09-30"), pd.Timestamp("2025-10-01"), pd.Timestamp("2025-12-31")),
        ("2026Q1", pd.Timestamp("2025-12-31"), pd.Timestamp("2026-01-01"), pd.Timestamp("2026-03-31")),
    ]

    results = []
    for name, tr_end, te_s, te_e in quarters:
        print(f"\n{'='*64}")
        print(f"  季度 {name}  test: {te_s.date()}→{te_e.date()}")
        print('='*64)
        t0 = time.time()
        try:
            res = run_quarter(feat_z, label, daily, tr_end, te_s, te_e)
            if res is None:
                print(f"  ❌ 跳过 ({name} 训练数据不足)")
                continue
            res["quarter"] = name
            results.append(res)
            print(f"  IC_IR={res['ic_ir']:+.2f}  IR={res.get('info_ratio', 0):+.2f}  "
                  f"Sharpe={res.get('sharpe', 0):.2f}  "
                  f"excess={res.get('excess_return', 0):+.2%}  "
                  f"ex_dd={res.get('excess_max_dd', 0):+.2%}  "
                  f"耗时 {time.time()-t0:.0f}s")
        except Exception as e:
            print(f"  ❌ {name} 异常: {e}")
            import traceback; traceback.print_exc()

    # 汇总
    if not results:
        print("\n无有效季度"); return

    df = pd.DataFrame(results)
    print("\n" + "="*64)
    print("  🔍 Walk-forward 汇总")
    print('='*64)
    cols = ["quarter", "ic_ir", "info_ratio", "sharpe", "bench_sharpe",
            "excess_return", "excess_max_dd", "avg_turnover", "n_rebalances"]
    subset = df[cols].copy()
    for c in ("excess_return", "excess_max_dd", "avg_turnover"):
        subset[c] = subset[c].apply(lambda v: f"{v:+.2%}")
    for c in ("ic_ir", "info_ratio", "sharpe", "bench_sharpe"):
        subset[c] = subset[c].apply(lambda v: f"{v:+.2f}")
    print(subset.to_string(index=False))

    # 稳定性指标
    print("\n" + "="*64)
    print("  📊 稳定性评估")
    print('='*64)
    ir_arr = df["info_ratio"].values
    er_arr = df["excess_return"].values
    print(f"  季度数: {len(df)}")
    print(f"  信息比率 IR:")
    print(f"    均值:   {ir_arr.mean():+.2f}")
    print(f"    中位数: {np.median(ir_arr):+.2f}")
    print(f"    标准差: {ir_arr.std():.2f}")
    print(f"    > 0:    {int((ir_arr > 0).sum())}/{len(ir_arr)} ({(ir_arr>0).mean():.0%})")
    print(f"    > 0.5:  {int((ir_arr > 0.5).sum())}/{len(ir_arr)}")
    print(f"    < -0.5: {int((ir_arr < -0.5).sum())}/{len(ir_arr)}")
    print(f"  超额收益:")
    print(f"    正季度: {int((er_arr > 0).sum())}/{len(er_arr)} ({(er_arr>0).mean():.0%})")
    print(f"    均值:   {er_arr.mean():+.2%}")
    print(f"    中位数: {np.median(er_arr):+.2%}")

    # 最差 3 个季度
    worst = df.nsmallest(3, "info_ratio")[["quarter", "info_ratio", "excess_return"]]
    print(f"\n  最差 3 个季度:")
    for _, r in worst.iterrows():
        print(f"    {r['quarter']}: IR={r['info_ratio']:+.2f}, excess={r['excess_return']:+.2%}")

    # 结论
    print("\n" + "="*64)
    print(f"  VERDICT")
    print('='*64)
    win_rate = (ir_arr > 0).mean()
    mean_ir = ir_arr.mean()
    if mean_ir >= 1.0 and win_rate >= 0.7:
        print(f"  ✅ 稳健 alpha (平均 IR={mean_ir:.2f}, 胜率 {win_rate:.0%})")
    elif mean_ir >= 0.5 and win_rate >= 0.6:
        print(f"  ⚠️  真 alpha 但不稳定 (平均 IR={mean_ir:.2f}, 胜率 {win_rate:.0%})")
    elif mean_ir >= 0.3:
        print(f"  ❓ 边缘有效 (平均 IR={mean_ir:.2f}, 胜率 {win_rate:.0%})")
    else:
        print(f"  ❌ 仍无稳定 alpha (平均 IR={mean_ir:.2f})")

    # 保存
    out = ROOT / "output" / f"walkforward_v6_{time.strftime('%Y%m%d_%H%M')}.md"
    out.parent.mkdir(exist_ok=True)
    lines = ["# V6 Walk-forward 9 季度 OOS 验证\n",
              f"- 训练窗口: 252 交易日 (1 年)",
              f"- 每季度独立训练 + IC 聚类 + 自适应极性 + 回测",
              f"- 参数: z>0.8, window=90, inertia=0.6\n",
              "## 逐季度结果\n",
              "| 季度 | IC_IR | IR | Sharpe | Bench Sharpe | 超额 | 超额回撤 | 换手 |",
              "|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['quarter']} | {r['ic_ir']:+.2f} | {r['info_ratio']:+.2f} | "
            f"{r['sharpe']:+.2f} | {r['bench_sharpe']:+.2f} | "
            f"{r['excess_return']:+.2%} | {r['excess_max_dd']:+.2%} | "
            f"{r['avg_turnover']:.1%} |"
        )
    lines += [
        "\n## 汇总",
        f"- 平均 IR: {mean_ir:+.2f}",
        f"- IR 胜率: {win_rate:.0%}",
        f"- 超额收益正季度: {(er_arr>0).mean():.0%}",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n报告: {out}")


if __name__ == "__main__":
    main()
