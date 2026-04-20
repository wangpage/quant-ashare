"""严格样本外 hold-out 验证 V5.

目的: 证明 IR 1.86 不是过拟合.

铁律:
    - 训练期: 2023-01-01 ~ 2024-12-31 (2 年)
    - 测试期: 2025-01-01 ~ 2026-04-20 (冻结, 完全不让模型看)

防泄露关键:
    1. IC 聚类去共线: 只用训练期数据选因子
    2. 模型训练: rolling window 内部, 但不跨越 TRAIN_END
    3. 测试集: 只做预测 + 回测, 不参与任何选择/调参

对比指标:
    - 训练期内部 IR (rolling OOS)
    - Hold-out IR (完全冻结)
    - 差距 = 过拟合程度 (差距 < 0.3 算稳)

用法:
    python3 scripts/run_holdout_v5.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

# 复用 v5 的核心函数
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

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"

TRAIN_END = pd.Timestamp("2024-12-31")
TEST_START = pd.Timestamp("2025-01-01")


def rolling_predict_holdout(X, y, train_days=252, step_days=21,
                              train_end=TRAIN_END):
    """只用 <= train_end 的数据做训练, 预测 > train_end 的数据.

    实际上是 rolling, 但 train 窗口永远不包含 train_end 之后的数据.
    为防未来函数: 测试样本 t 的训练集为 [t-train_days, t-1], 且 t-1 ≤ train_end 时
    才用真实滚动; 否则用最后一个 ≤ train_end 的窗口 "冻结" 预测.
    """
    dates = X.index.get_level_values("date").unique().sort_values()
    train_dates = dates[dates <= train_end]
    test_dates = dates[dates > train_end]

    if len(train_dates) < train_days + 10:
        raise ValueError(f"训练期数据不足 (需 ≥{train_days+10} 天)")

    # 1) 训练集内部 rolling OOS (用于对比)
    print(f"\n[训练期 rolling] train<{train_end.date()}")
    preds_tr = []
    i = train_days
    while i < len(train_dates):
        j = min(i + step_days, len(train_dates))
        tr_s = train_dates[i - train_days]
        tr_e = train_dates[i - 1]
        te_s = train_dates[i]
        te_e = train_dates[j - 1]
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
    pred_train_oos = (pd.concat(preds_tr).sort_index() if preds_tr
                       else pd.Series(dtype=float))

    # 2) Hold-out: 用最后一个"合法"训练集 (截至 train_end) 训练多个模型,
    #    每 step_days 重训但训练数据只用 ≤ 对应日前 1 天 且 ≤ train_end
    # 严格版: 不滚动, 只用 [train_end - train_days, train_end] 一次训练, 预测整个 hold-out
    # 更严版(推荐): 测试日 t 训练集 = [t-train_days, t-1], 但 t-1 必须 ≤ train_end 的上限
    #   → 这其实意味着: 对 t > train_end + step_days, 用 train_end 前最后一个窗口
    print(f"\n[Hold-out] test>{train_end.date()} ~ {test_dates[-1].date()}")

    # 我采用更严格做法: 仅训练一次, 用完整 [train_end - train_days, train_end] 窗口
    tr_e = train_end
    tr_s = train_dates[max(0, len(train_dates) - train_days)]
    X_tr = X.loc[tr_s:tr_e]; y_tr = y.loc[tr_s:tr_e]
    mask = y_tr.notna() & X_tr.notna().all(axis=1)
    X_tr, y_tr = X_tr[mask], y_tr[mask]
    print(f"  训练集: [{tr_s.date()}→{tr_e.date()}] n={len(X_tr)}")
    m = _fit_model(X_tr, y_tr)

    X_te = X.loc[test_dates[0]:test_dates[-1]]
    pred_holdout = pd.Series(m.predict(X_te.values), index=X_te.index)
    print(f"  hold-out 预测 {len(pred_holdout)} 条, 期间 {test_dates[0].date()}→{test_dates[-1].date()}")

    return pred_train_oos, pred_holdout, (tr_s, tr_e)


def main():
    start, end = "20230101", "20260420"
    print(f"\n{'='*64}")
    print(f"  V5 Hold-out 严格样本外验证 {time.strftime('%Y-%m-%d %H:%M')}")
    print(f"  TRAIN: {start} → {TRAIN_END.strftime('%Y%m%d')}")
    print(f"  TEST:  {TEST_START.strftime('%Y%m%d')} → {end}")
    print('='*64)

    # 数据
    daily = pd.read_parquet(CACHE / f"kline_{start}_{end}_n500.parquet")
    lhb_df = pd.read_parquet(CACHE / f"lhb_{start}_{end}.parquet")
    ins_df = pd.read_parquet(CACHE / f"insider_{start}_{end}.parquet")
    print(f"\nkline {len(daily)} 行, lhb {len(lhb_df)}, insider {len(ins_df)}")

    # 因子
    print("\n[1/4] 计算全部 V5 因子...")
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

    # Label
    print("\n[2/4] 条件 Label (horizon=30)...")
    label = make_conditional_label(daily, horizon=30, dd_clip=0.25)

    aligned = feat_z.join(label.rename("label"), how="inner")
    valid = aligned["label"].notna() & aligned.drop(columns="label").notna().all(axis=1)
    feat_valid = aligned.drop(columns="label")[valid]
    y_valid = aligned["label"][valid]

    # 关键: IC 聚类只用训练集
    print(f"\n[3/4] 🔒 IC 聚类【只用训练集】({start} → {TRAIN_END.date()})...")
    train_mask = feat_valid.index.get_level_values("date") <= TRAIN_END
    feat_train = feat_valid[train_mask]
    y_train = y_valid[train_mask]
    selected = ic_cluster_select(feat_train, y_train,
                                  corr_threshold=0.6, min_ic=0.005)

    X = feat_valid[selected]
    y = y_valid

    # 训练 + hold-out 预测
    print(f"\n[4/4] 训练 + Hold-out 预测...")
    pred_tr, pred_ho, (tr_s, tr_e) = rolling_predict_holdout(
        X, y, train_days=252, step_days=21, train_end=TRAIN_END)

    # 评估
    def eval_preds(pred, y_series, name):
        ic = pred.groupby(level="date").apply(
            lambda s: s.corr(y_series.loc[s.index], method="spearman")
        ).dropna()
        print(f"\n  === {name} ===")
        print(f"  IC mean={ic.mean():.4f}  IR={ic.mean()/ic.std():.2f}  "
              f"IC>0={float((ic>0).mean()):.2%}  n_days={len(ic)}")
        return ic

    ic_tr = eval_preds(pred_tr, y, "训练期 rolling OOS")
    ic_ho = eval_preds(pred_ho, y, "Hold-out (完全冻结)")

    # 回测
    print("\n" + "="*64 + "\n  训练期 回测")
    print('='*64)
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

    print("\n" + "="*64 + "\n  🔒 Hold-out 回测 (2025 ~ 2026.4)")
    print('='*64)
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

    # 汇总
    tr_ir = stats_tr.get("info_ratio", 0)
    ho_ir = stats_ho.get("info_ratio", 0)
    print("\n" + "="*64)
    print(f"  VERDICT")
    print('='*64)
    print(f"  训练期 IR: {tr_ir:.2f}")
    print(f"  Hold-out IR: {ho_ir:.2f}")
    print(f"  过拟合度(差距): {tr_ir - ho_ir:+.2f}")
    if ho_ir >= 1.3:
        print(f"  ✅ 可信 (hold-out IR ≥ 1.3)")
    elif ho_ir >= 0.8:
        print(f"  ⚠️  中等 (0.8 ≤ hold-out IR < 1.3, 有一定过拟合)")
    else:
        print(f"  ❌ 过拟合严重 (hold-out IR < 0.8)")

    # 保存
    out = ROOT / "output" / f"holdout_v5_{time.strftime('%Y%m%d_%H%M')}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(
        f"# V5 Hold-out 样本外验证\n\n"
        f"- 训练: {start} → {TRAIN_END.date()}\n"
        f"- 测试 (冻结): {TEST_START.date()} → {end}\n"
        f"- IC 聚类: 只用训练集\n"
        f"- 精选因子: {len(selected)}\n\n"
        f"## 训练期 rolling IR\n\n"
        f"- IC mean: {ic_tr.mean():.4f}, IR: {ic_tr.mean()/ic_tr.std():.2f}\n"
        f"- 策略 Sharpe: {stats_tr.get('sharpe', 0):.2f}\n"
        f"- 信息比率: {tr_ir:.2f}\n"
        f"- 超额年化: {stats_tr.get('excess_return', 0):.2%}\n"
        f"- 超额回撤: {stats_tr.get('excess_max_dd', 0):.2%}\n\n"
        f"## Hold-out 冻结期\n\n"
        f"- IC mean: {ic_ho.mean():.4f}, IR: {ic_ho.mean()/ic_ho.std():.2f}\n"
        f"- 策略 Sharpe: {stats_ho.get('sharpe', 0):.2f}\n"
        f"- 信息比率: {ho_ir:.2f}\n"
        f"- 超额年化: {stats_ho.get('excess_return', 0):.2%}\n"
        f"- 超额回撤: {stats_ho.get('excess_max_dd', 0):.2%}\n\n"
        f"## 过拟合评估\n\n"
        f"- 训练 IR → Hold-out IR: {tr_ir:.2f} → {ho_ir:.2f}\n"
        f"- 差距: {tr_ir - ho_ir:+.2f}\n",
        encoding="utf-8",
    )
    print(f"\n报告: {out}")


if __name__ == "__main__":
    main()
