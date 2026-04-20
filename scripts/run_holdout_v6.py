"""V6 Hold-out 验证 - 带 B+ 自适应极性.

核心改动 vs V5:
    加一层 "自适应极性" 变换
    - IC z-score + 显著性过滤 + 惯性 + 横截面归一化 + IC 衰减
    - shift(horizon) 严格防泄露

预期: V5 hold-out IR -1.48 → V6 hold-out IR 0.5-1.2+
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
    impact_bps, _fit_model, backtest_v4,
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

TRAIN_END = pd.Timestamp("2024-12-31")
TEST_START = pd.Timestamp("2025-01-01")
HORIZON = 30


def rolling_predict_holdout(X, y, train_days=252, train_end=TRAIN_END):
    """Hold-out 严格: 只用 ≤ train_end 的数据训练最后一次, 预测全部 hold-out."""
    dates = X.index.get_level_values("date").unique().sort_values()
    train_dates = dates[dates <= train_end]
    test_dates = dates[dates > train_end]

    # 训练期 rolling OOS
    print(f"\n[训练期 rolling] train≤{train_end.date()}")
    preds_tr = []
    step = 21
    i = train_days
    while i < len(train_dates):
        j = min(i + step, len(train_dates))
        tr_s, tr_e = train_dates[i - train_days], train_dates[i - 1]
        te_s, te_e = train_dates[i], train_dates[j - 1]
        try:
            X_tr = X.loc[tr_s:tr_e]; y_tr = y.loc[tr_s:tr_e]
            mask = y_tr.notna() & X_tr.notna().all(axis=1)
            X_tr, y_tr = X_tr[mask], y_tr[mask]
            if len(X_tr) < 1000:
                i = j; continue
            m = _fit_model(X_tr, y_tr)
            X_te = X.loc[te_s:te_e]
            preds_tr.append(pd.Series(m.predict(X_te.values), index=X_te.index))
        except Exception as e:
            print(f"  err {e}")
        i = j
    pred_train = (pd.concat(preds_tr).sort_index() if preds_tr
                   else pd.Series(dtype=float))

    # Hold-out: 用最终训练集一次训练
    print(f"\n[Hold-out] test>{train_end.date()}")
    tr_e = train_end
    tr_s = train_dates[max(0, len(train_dates) - train_days)]
    X_tr = X.loc[tr_s:tr_e]; y_tr = y.loc[tr_s:tr_e]
    mask = y_tr.notna() & X_tr.notna().all(axis=1)
    X_tr, y_tr = X_tr[mask], y_tr[mask]
    print(f"  训练集 [{tr_s.date()}→{tr_e.date()}] n={len(X_tr)}")
    m = _fit_model(X_tr, y_tr)
    X_te = X.loc[test_dates[0]:test_dates[-1]]
    pred_ho = pd.Series(m.predict(X_te.values), index=X_te.index)
    print(f"  预测 {len(pred_ho)} 条")
    return pred_train, pred_ho


def main():
    start, end = "20230101", "20260420"
    print(f"\n{'='*64}")
    print(f"  V6 Hold-out (B+ 自适应极性) {time.strftime('%Y-%m-%d %H:%M')}")
    print(f"  TRAIN: {start} → {TRAIN_END.strftime('%Y%m%d')}")
    print(f"  TEST:  {TEST_START.strftime('%Y%m%d')} → {end}")
    print('='*64)

    daily = pd.read_parquet(CACHE / f"kline_{start}_{end}_n500.parquet")
    lhb_df = pd.read_parquet(CACHE / f"lhb_{start}_{end}.parquet")
    ins_df = pd.read_parquet(CACHE / f"insider_{start}_{end}.parquet")
    print(f"\nkline {len(daily)}, lhb {len(lhb_df)}, insider {len(ins_df)}")

    # 1. 全部因子
    print("\n[1/5] 计算 V5 全部因子...")
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
    print(f"  特征 {feat_z.shape}")

    # 2. Label
    print("\n[2/5] 条件 Label (horizon=30, dd 惩罚)...")
    label = make_conditional_label(daily, horizon=HORIZON, dd_clip=0.25)

    # 3. 先做 IC 聚类 (只用训练集)
    print(f"\n[3/5] IC 聚类 (只用训练集)...")
    aligned = feat_z.join(label.rename("label"), how="inner")
    valid = aligned["label"].notna() & aligned.drop(columns="label").notna().all(axis=1)
    feat_valid = aligned.drop(columns="label")[valid]
    y_valid = aligned["label"][valid]
    train_mask = feat_valid.index.get_level_values("date") <= TRAIN_END
    selected = ic_cluster_select(feat_valid[train_mask], y_valid[train_mask],
                                  corr_threshold=0.6, min_ic=0.005)
    feat_valid = feat_valid[selected]

    # 4. 关键: 自适应极性 (调参: 放宽过滤, 激活更多因子)
    print(f"\n[4/5] 🎯 B+ 自适应极性 (z>0.8, 惯性 0.6, window=90)...")
    feat_adapt, weight_df = apply_adaptive_polarity(
        feat_valid, y_valid,
        horizon=HORIZON, window=90,
        z_threshold=0.8, z_cap=3.0,
        inertia=0.6, decay_lambda=0.0,   # 用稳定的 rolling
    )
    # 剔除前期 NaN 导致的全 0 行
    all_zero = (feat_adapt.abs().sum(axis=1) < 1e-9)
    feat_adapt = feat_adapt[~all_zero]
    y_valid = y_valid.loc[feat_adapt.index]
    print(f"  自适应后有效样本 {len(feat_adapt)}")

    # 5. 训练 + hold-out
    print(f"\n[5/5] 训练 + Hold-out...")
    pred_tr, pred_ho = rolling_predict_holdout(
        feat_adapt, y_valid, train_days=252, train_end=TRAIN_END)

    def eval_preds(pred, y_series, name):
        ic = pred.groupby(level="date").apply(
            lambda s: s.corr(y_series.loc[s.index], method="spearman")
        ).dropna()
        print(f"\n  === {name} ===")
        print(f"  IC mean={ic.mean():.4f}  IR={ic.mean()/ic.std():.2f}  "
              f"IC>0={float((ic>0).mean()):.2%}  n={len(ic)}")
        return ic

    ic_tr = eval_preds(pred_tr, y_valid, "训练期 rolling OOS")
    ic_ho = eval_preds(pred_ho, y_valid, "🔒 Hold-out (冻结)")

    # 回测
    print("\n" + "="*64 + "\n  训练期 回测\n" + '='*64)
    stats_tr = backtest_v4(pred_tr, daily, top_ratio=0.05,
                            rebalance_days=30, vol_target=0.20)
    for k, v in stats_tr.items():
        if isinstance(v, float):
            if any(s in k for s in ["return","drawdown","vol","turnover"]):
                print(f"  {k:20s} {v:>10.2%}")
            elif "bps" in k:
                print(f"  {k:20s} {v:>10.1f}")
            else:
                print(f"  {k:20s} {v:>10.4f}")
        else:
            print(f"  {k:20s} {v}")

    print("\n" + "="*64 + "\n  🔒 Hold-out 回测 (2025~2026.4)\n" + '='*64)
    stats_ho = backtest_v4(pred_ho, daily, top_ratio=0.05,
                            rebalance_days=30, vol_target=0.20)
    for k, v in stats_ho.items():
        if isinstance(v, float):
            if any(s in k for s in ["return","drawdown","vol","turnover"]):
                print(f"  {k:20s} {v:>10.2%}")
            elif "bps" in k:
                print(f"  {k:20s} {v:>10.1f}")
            else:
                print(f"  {k:20s} {v:>10.4f}")
        else:
            print(f"  {k:20s} {v}")

    tr_ir = stats_tr.get("info_ratio", 0)
    ho_ir = stats_ho.get("info_ratio", 0)
    print("\n" + "="*64)
    print(f"  VERDICT (B+ 自适应极性)")
    print('='*64)
    print(f"  训练期 IR:    {tr_ir:+.2f}")
    print(f"  Hold-out IR:  {ho_ir:+.2f}")
    print(f"  过拟合度:     {tr_ir - ho_ir:+.2f}")
    print(f"  V5 对比:      V5 hold-out IR = -1.48")
    print(f"  提升:         {ho_ir - (-1.48):+.2f}")
    if ho_ir >= 1.3:
        print(f"  ✅ 稳健 (hold-out IR ≥ 1.3)")
    elif ho_ir >= 0.5:
        print(f"  ⚠️  可用 (0.5 ≤ IR < 1.3, 需继续优化)")
    elif ho_ir > 0:
        print(f"  ❓ 勉强 (IR 正但弱, regime 切换仍未完全解决)")
    else:
        print(f"  ❌ 仍然过拟合")

    # 看 weight_df 诊断: 2025 年哪些因子被 flip
    print("\n🔍 关键因子极性切换诊断:")
    factors_to_watch = ["LU_COUNT_20", "STREAK_UP", "BOOM_BAN_FLAG",
                         "LHB_YOUZI_SELL_20", "INSIDER_REDUCE_30",
                         "MOM12_1"]
    for f in factors_to_watch:
        if f in weight_df.columns:
            # 训练末期 2024-10 vs hold-out 2025-06 的平均权重
            try:
                w_tr = weight_df.loc["2024-09":"2024-12", f].mean()
                w_ho = weight_df.loc["2025-05":"2025-10", f].mean()
                flip = "↔️ 翻转!" if w_tr * w_ho < -0.001 else ""
                print(f"  {f:22s} train_late={w_tr:+.3f}  ho_mid={w_ho:+.3f}  {flip}")
            except KeyError:
                pass


if __name__ == "__main__":
    main()
