"""V6 参数敏感性扫描 - 证明 IR 0.96 不是单次运气.

扫描网格:
    z_threshold: [0.5, 0.8, 1.0, 1.2]  # 显著性过滤门槛
    window:      [60, 90, 120]          # IC 估计窗口
    共 12 组, 每组跑 6 季度 (2024Q3 ~ 2026Q1) walk-forward.

输出:
    每组 (z, window) 的 hold-out IR 中位数 / 超额收益胜率 / 最差季度.
    若大部分参数组合 IR > 0.5 且胜率 > 60%, 策略真稳.
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


def prepare_data():
    """一次性计算全样本因子+标签."""
    start, end = "20230101", "20260420"
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


def run_one_config(feat_z, label, daily,
                    z_threshold: float, window: int,
                    inertia: float = 0.6):
    """对 6 个季度做 walk-forward, 返回每季度结果."""
    aligned = feat_z.join(label.rename("label"), how="inner")
    valid = aligned["label"].notna() & aligned.drop(columns="label").notna().all(axis=1)
    feat_valid = aligned.drop(columns="label")[valid]
    y_valid = aligned["label"][valid]

    # 固定的 6 个季度 hold-out
    quarters = [
        ("2024Q3", pd.Timestamp("2024-06-30"), pd.Timestamp("2024-07-01"), pd.Timestamp("2024-09-30")),
        ("2024Q4", pd.Timestamp("2024-09-30"), pd.Timestamp("2024-10-01"), pd.Timestamp("2024-12-31")),
        ("2025Q1", pd.Timestamp("2024-12-31"), pd.Timestamp("2025-01-01"), pd.Timestamp("2025-03-31")),
        ("2025Q2", pd.Timestamp("2025-03-31"), pd.Timestamp("2025-04-01"), pd.Timestamp("2025-06-30")),
        ("2025Q3", pd.Timestamp("2025-06-30"), pd.Timestamp("2025-07-01"), pd.Timestamp("2025-09-30")),
        ("2025Q4", pd.Timestamp("2025-09-30"), pd.Timestamp("2025-10-01"), pd.Timestamp("2025-12-31")),
    ]

    quarterly_ir = []
    quarterly_er = []
    for name, tr_end, te_s, te_e in quarters:
        dates = feat_valid.index.get_level_values("date").unique().sort_values()
        train_dates = dates[dates <= tr_end]
        if len(train_dates) < 252:
            continue
        train_start = train_dates[-252]
        train_mask = feat_valid.index.get_level_values("date").isin(train_dates[-252:])

        try:
            # IC 聚类 (仅训练)
            selected = ic_cluster_select(feat_valid[train_mask], y_valid[train_mask],
                                          corr_threshold=0.6, min_ic=0.005)
            if not selected:
                continue
            feat_sel = feat_valid[selected]

            # 自适应极性 (当前参数)
            feat_adapt, _ = apply_adaptive_polarity(
                feat_sel, y_valid,
                horizon=HORIZON, window=window,
                z_threshold=z_threshold, z_cap=3.0,
                inertia=inertia, decay_lambda=0.0,
            )
            all_zero = (feat_adapt.abs().sum(axis=1) < 1e-9)
            feat_adapt = feat_adapt[~all_zero]
            y_adapt = y_valid.loc[feat_adapt.index]

            # 训练
            X_tr = feat_adapt[
                (feat_adapt.index.get_level_values("date") >= train_start) &
                (feat_adapt.index.get_level_values("date") <= tr_end)
            ]
            y_tr = y_adapt.loc[X_tr.index]
            m = y_tr.notna() & X_tr.notna().all(axis=1)
            X_tr, y_tr = X_tr[m], y_tr[m]
            if len(X_tr) < 1000:
                continue
            model = _fit_model(X_tr, y_tr)

            # 预测
            X_te = feat_adapt[
                (feat_adapt.index.get_level_values("date") >= te_s) &
                (feat_adapt.index.get_level_values("date") <= te_e)
            ]
            if len(X_te) == 0:
                continue
            pred = pd.Series(model.predict(X_te.values), index=X_te.index)

            # 回测 (rebalance=10, no vol_target)
            stats = backtest_v4(pred, daily, top_ratio=0.05,
                                 rebalance_days=10, vol_target=None)
            if "info_ratio" in stats:
                quarterly_ir.append(stats["info_ratio"])
                quarterly_er.append(stats["excess_return"])
        except Exception as e:
            print(f"      {name} err: {e}")

    if not quarterly_ir:
        return None

    ir_arr = np.array(quarterly_ir)
    er_arr = np.array(quarterly_er)
    return {
        "n_quarters": len(ir_arr),
        "ir_median": float(np.median(ir_arr)),
        "ir_mean": float(ir_arr.mean()),
        "ir_min": float(ir_arr.min()),
        "ir_max": float(ir_arr.max()),
        "er_median": float(np.median(er_arr)),
        "er_mean": float(er_arr.mean()),
        "win_rate": float((er_arr > 0).mean()),
        "worst_er": float(er_arr.min()),
    }


def main():
    print(f"\n{'='*70}\n  V6 参数敏感性扫描 {time.strftime('%Y-%m-%d %H:%M')}\n{'='*70}")
    print("\n[预处理] 计算全样本因子/标签...")
    daily, feat_z, label = prepare_data()
    print(f"  feat_z {feat_z.shape}")

    # 网格
    grid = []
    for z in [0.5, 0.8, 1.0, 1.2]:
        for w in [60, 90, 120]:
            grid.append((z, w))

    print(f"\n扫描 {len(grid)} 组参数, 每组 6 季度 walk-forward...")
    results = []
    for i, (z, w) in enumerate(grid, 1):
        print(f"\n--- [{i}/{len(grid)}] z_threshold={z}, window={w} ---")
        t0 = time.time()
        res = run_one_config(feat_z, label, daily, z_threshold=z, window=w)
        if res is None:
            print(f"  ❌ 失败")
            continue
        res["z_threshold"] = z
        res["window"] = w
        results.append(res)
        print(f"  n_q={res['n_quarters']}  IR_median={res['ir_median']:+.2f}  "
              f"win_rate={res['win_rate']:.0%}  ER_median={res['er_median']:+.2%}  "
              f"worst_ER={res['worst_er']:+.2%}  耗时 {time.time()-t0:.0f}s")

    # 汇总
    if not results:
        print("\n❌ 无结果"); return

    df = pd.DataFrame(results)
    print("\n" + "="*70)
    print("  📊 参数敏感性矩阵")
    print('='*70)
    print("\n胜率矩阵 (超额收益 > 0 的季度占比):")
    pivot_wr = df.pivot(index="z_threshold", columns="window", values="win_rate")
    print(pivot_wr.applymap(lambda v: f"{v:.0%}"))

    print("\nIR 中位数矩阵:")
    pivot_ir = df.pivot(index="z_threshold", columns="window", values="ir_median")
    print(pivot_ir.applymap(lambda v: f"{v:+.2f}"))

    print("\n超额收益中位数矩阵 (按季度):")
    pivot_er = df.pivot(index="z_threshold", columns="window", values="er_median")
    print(pivot_er.applymap(lambda v: f"{v:+.2%}"))

    print("\n最差季度超额:")
    pivot_worst = df.pivot(index="z_threshold", columns="window", values="worst_er")
    print(pivot_worst.applymap(lambda v: f"{v:+.2%}"))

    # 稳健性评估
    print("\n" + "="*70)
    print("  VERDICT")
    print('='*70)
    good = df[(df["ir_median"] >= 0.5) & (df["win_rate"] >= 0.6)]
    print(f"  稳健组合数 (IR_median≥0.5 且 win_rate≥60%): {len(good)}/{len(df)}")
    if len(good) >= len(df) * 0.6:
        print(f"  ✅ 策略对参数不敏感, 真 alpha 存在")
    elif len(good) >= len(df) * 0.3:
        print(f"  ⚠️  部分参数区间有效, 需谨慎选点")
    else:
        print(f"  ❌ 对参数极度敏感, 可能过拟合单一参数")

    # best config
    df_sorted = df.sort_values("ir_median", ascending=False)
    print(f"\nTop 3 最稳参数:")
    for _, r in df_sorted.head(3).iterrows():
        print(f"  z={r['z_threshold']} window={r['window']}: "
              f"IR={r['ir_median']:+.2f} win={r['win_rate']:.0%} ER={r['er_median']:+.2%}")

    # 保存
    out = ROOT / "output" / f"sensitivity_v6_{time.strftime('%Y%m%d_%H%M')}.md"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out.with_suffix(".csv"), index=False)
    lines = ["# V6 参数敏感性扫描\n",
              "## 胜率矩阵 (季度超额 > 0 占比)\n",
              pivot_wr.applymap(lambda v: f"{v:.0%}").to_markdown(),
              "\n## IR 中位数矩阵\n",
              pivot_ir.applymap(lambda v: f"{v:+.2f}").to_markdown(),
              "\n## 超额收益中位数\n",
              pivot_er.applymap(lambda v: f"{v:+.2%}").to_markdown(),
              "\n## 最差季度\n",
              pivot_worst.applymap(lambda v: f"{v:+.2%}").to_markdown(),
              f"\n## 稳健性\n- 12 组中 {len(good)} 组满足 IR_median≥0.5 且胜率≥60%"]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n报告: {out}")


if __name__ == "__main__":
    main()
