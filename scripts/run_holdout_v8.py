"""V8 - 主力资金流 (大单/超大单) 驱动的策略.

核心转变:
    - 放弃"公开因子越堆越多"的 V6 思路
    - 只用 2 类核心因子:
      (1) 主力资金流 (超大单/大单/散户) - "大鳄进出" 直接信号
      (2) 精简 V6 体系 (龙虎榜 + insider + 少量反转)

严格 leak-free:
    - label cutoff = train_end - horizon
    - IC 聚类只用训练集
    - adaptive polarity shift(horizon)
    - 数据只有 120 日 -> 训练 80 / 测试 40
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
from data_adapter.fundflow import build_fundflow_features, FUNDFLOW_FACTOR_NAMES
from factors.alpha_pandas import compute_pandas_alpha
from factors.alpha_reversal import compute_advanced_alpha
from factors.alpha_limit import compute_limit_alpha, LIMIT_FACTOR_NAMES
from factors.adaptive_polarity import apply_adaptive_polarity

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"


def rolling_predict_strict(X, y, train_end, horizon, train_days=60):
    """严格 leak-free walk-forward.
    - 训练 label cutoff = train_end - horizon (交易日数)
    - 预测 > train_end
    """
    dates = X.index.get_level_values("date").unique().sort_values()
    train_dates = dates[dates <= train_end]
    test_dates = dates[dates > train_end]
    if len(train_dates) <= horizon + 20:
        raise ValueError(f"训练日不足 (need > {horizon+20}, got {len(train_dates)})")

    # label_cutoff = train_end - horizon 个交易日
    label_cutoff = train_dates[-(horizon + 1)]
    tr_start = train_dates[max(0, len(train_dates) - train_days - horizon)]
    X_tr = X.loc[tr_start:label_cutoff]
    y_tr = y.loc[tr_start:label_cutoff]
    mask = y_tr.notna() & X_tr.notna().all(axis=1)
    X_tr, y_tr = X_tr[mask], y_tr[mask]
    print(f"  🔒 训练 [{tr_start.date()}→{label_cutoff.date()}] n={len(X_tr)} "
          f"(label_cutoff=train_end-{horizon}d)")
    if len(X_tr) < 500:
        raise ValueError(f"训练样本不足 {len(X_tr)}")

    model = _fit_model(X_tr, y_tr)
    X_te = X.loc[test_dates[0]:test_dates[-1]]
    pred = pd.Series(model.predict(X_te.values), index=X_te.index)
    print(f"  预测 {len(pred)} 条 ({test_dates[0].date()}→{test_dates[-1].date()})")
    return pred


def main(horizon: int = 5, rebalance_days: int = 5):
    print(f"\n{'='*64}\n  V8 (主力资金流驱动) horizon={horizon}, reb={rebalance_days}d\n{'='*64}")

    # 加载资金流 (仅覆盖近 120 日, 决定了时间窗口)
    ff_path = CACHE / "fundflow_500.parquet"
    if not ff_path.exists():
        print(f"❌ {ff_path} 未就绪, 先跑数据拉取"); return
    ff_df = pd.read_parquet(ff_path)
    print(f"  资金流 {len(ff_df)} 行, {ff_df['code'].nunique()} 只, "
          f"日期 {ff_df['date'].min().date()} ~ {ff_df['date'].max().date()}")

    # 日 K (选 ff 覆盖的日期)
    daily = pd.read_parquet(CACHE / "kline_20230101_20260420_n500.parquet")
    daily["date"] = pd.to_datetime(daily["date"])
    ff_start = ff_df["date"].min()
    ff_end = ff_df["date"].max()
    daily = daily[(daily["date"] >= ff_start) & (daily["date"] <= ff_end)]
    # 只保留 ff_df 覆盖的股票
    common_codes = set(ff_df["code"].unique()) & set(daily["code"].unique())
    daily = daily[daily["code"].isin(common_codes)]
    print(f"  日 K 对齐: {len(daily)} 行, {daily['code'].nunique()} 只")

    # LHB / insider 对齐区间
    lhb_df = pd.read_parquet(CACHE / "lhb_20230101_20260420.parquet")
    lhb_df["TRADE_DATE"] = pd.to_datetime(lhb_df["TRADE_DATE"])
    lhb_df = lhb_df[(lhb_df["TRADE_DATE"] >= ff_start) & (lhb_df["TRADE_DATE"] <= ff_end)]
    ins_df = pd.read_parquet(CACHE / "insider_20230101_20260420.parquet")
    ins_df["CHANGE_DATE"] = pd.to_datetime(ins_df["CHANGE_DATE"])
    ins_df = ins_df[(ins_df["CHANGE_DATE"] >= ff_start) & (ins_df["CHANGE_DATE"] <= ff_end)]
    print(f"  LHB {len(lhb_df)} 条, insider {len(ins_df)} 条")

    # 因子: 3 条线
    print("\n[1/5] 计算全部因子...")
    feat_tech = compute_pandas_alpha(daily)
    feat_rev = compute_advanced_alpha(daily)
    feat_limit = compute_limit_alpha(daily)
    feat_combo = feat_tech.join(feat_rev, how="outer").join(feat_limit, how="outer")
    trading_dates = pd.DatetimeIndex(sorted(daily["date"].unique()))

    feat_lhb = build_lhb_features(lhb_df, trading_dates)
    feat_combo = feat_combo.join(feat_lhb, how="left")
    feat_ins = build_insider_features(ins_df, trading_dates)
    feat_combo = feat_combo.join(feat_ins, how="left")
    # V8 核心: 主力资金流
    feat_ff = build_fundflow_features(ff_df, trading_dates)
    feat_combo = feat_combo.join(feat_ff, how="left")
    print(f"  资金流因子 {len(FUNDFLOW_FACTOR_NAMES)}, "
          f"特征矩阵 {feat_combo.shape}")

    for f in (LHB_FACTOR_NAMES + LHB_B2_FACTOR_NAMES
              + INSIDER_FACTOR_NAMES + LIMIT_FACTOR_NAMES + FUNDFLOW_FACTOR_NAMES):
        if f in feat_combo.columns:
            feat_combo[f] = feat_combo[f].fillna(0)

    def _z(s):
        mu, sd = s.mean(), s.std()
        return (s - mu) / sd if sd > 0 else s * 0
    feat_z = feat_combo.groupby(level="date").transform(_z).clip(-3, 3).fillna(0)

    # Label
    label = make_conditional_label(daily, horizon=horizon, dd_clip=0.25)

    # 切分
    all_dates = feat_z.index.get_level_values("date").unique().sort_values()
    # 80 训 40 测 (120 - horizon - 预留)
    split_idx = max(len(all_dates) // 2, len(all_dates) - 40 - horizon)
    train_end = all_dates[split_idx]
    print(f"\n[2/5] 切分:")
    print(f"  全部日期: {all_dates[0].date()} ~ {all_dates[-1].date()} ({len(all_dates)} 日)")
    print(f"  train_end: {train_end.date()} (训练 {split_idx+1} 日, 测试 {len(all_dates)-split_idx-1} 日)")

    aligned = feat_z.join(label.rename("label"), how="inner")
    valid = aligned["label"].notna() & aligned.drop(columns="label").notna().all(axis=1)
    feat_valid = aligned.drop(columns="label")[valid]
    y_valid = aligned["label"][valid]

    # IC 聚类 - 🔒 label cutoff
    print(f"\n[3/5] IC 聚类 (label cutoff = train_end - {horizon}d)...")
    train_dates_arr = feat_valid.index.get_level_values("date").unique().sort_values()
    train_dates_arr = train_dates_arr[train_dates_arr <= train_end]
    if len(train_dates_arr) <= horizon:
        print("❌ 训练日太少"); return
    ic_cutoff = train_dates_arr[-(horizon + 1)]
    train_mask = feat_valid.index.get_level_values("date") <= ic_cutoff
    selected = ic_cluster_select(feat_valid[train_mask], y_valid[train_mask],
                                  corr_threshold=0.6, min_ic=0.005)
    if not selected:
        print("❌ 无有效因子"); return
    feat_sel = feat_valid[selected]

    # 自适应极性
    print(f"\n[4/5] 自适应极性 (z>0.8, inertia=0.6)...")
    feat_adapt, weight_df = apply_adaptive_polarity(
        feat_sel, y_valid, horizon=horizon, window=30,  # 数据少, window 缩到 30
        z_threshold=0.8, z_cap=3.0, inertia=0.6, decay_lambda=0.0,
    )
    all_zero = (feat_adapt.abs().sum(axis=1) < 1e-9)
    feat_adapt = feat_adapt[~all_zero]
    y_adapt = y_valid.loc[feat_adapt.index]

    # 训练 + hold-out 预测
    print(f"\n[5/5] 训练 + hold-out 预测...")
    try:
        pred = rolling_predict_strict(feat_adapt, y_adapt, train_end=train_end,
                                        horizon=horizon, train_days=60)
    except Exception as e:
        print(f"❌ {e}"); return

    ic = pred.groupby(level="date").apply(
        lambda s: s.corr(y_adapt.loc[s.index], method="spearman")
    ).dropna()
    print(f"\n  === Hold-out IC ===")
    print(f"  IC mean={ic.mean():+.4f}  IR={ic.mean()/ic.std():+.2f}  "
          f"IC>0={float((ic>0).mean()):.2%}  n={len(ic)}")

    # 回测
    print(f"\n  === Hold-out 回测 ===")
    stats = backtest_v4(pred, daily, top_ratio=0.05,
                         rebalance_days=rebalance_days,
                         vol_target=0.20)
    for k, v in stats.items():
        if isinstance(v, float):
            if any(s in k for s in ["return","drawdown","vol","turnover"]):
                print(f"  {k:20s} {v:>10.2%}")
            elif "bps" in k:
                print(f"  {k:20s} {v:>10.1f}")
            else:
                print(f"  {k:20s} {v:>10.4f}")
        else:
            print(f"  {k:20s} {v}")

    ho_ir = stats.get("info_ratio", 0)
    print("\n" + "="*64)
    print(f"  V8 VERDICT (主力资金流驱动)")
    print('='*64)
    print(f"  Hold-out IR:  {ho_ir:+.2f}")
    print(f"  V6 leak-free 对比: V6 IR ≈ -0.13")
    print(f"  变化: {ho_ir - (-0.13):+.2f}")
    if ho_ir >= 1.0:
        print(f"  ✅ 显著 alpha (资金流信号真的有效)")
    elif ho_ir >= 0.5:
        print(f"  ⚠️  边际有效")
    elif ho_ir > 0:
        print(f"  ❓ 弱正 (需要更多数据)")
    else:
        print(f"  ❌ 仍无效")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--reb", type=int, default=5)
    args = ap.parse_args()
    main(args.horizon, args.reb)
