"""查看 LightGBM 的 feature importance, 确认 B2 因子是否真的被使用."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np

from data_adapter.lhb import (
    build_lhb_features, LHB_FACTOR_NAMES, LHB_B2_FACTOR_NAMES,
)
from factors.alpha_pandas import compute_pandas_alpha
from factors.alpha_reversal import compute_advanced_alpha

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"

daily = pd.read_parquet(CACHE / "kline_20230101_20260420_n500.parquet")
lhb_df = pd.read_parquet(CACHE / "lhb_20230101_20260420.parquet")

feat_tech = compute_pandas_alpha(daily)
feat_rev = compute_advanced_alpha(daily)
feat_combo = feat_tech.join(feat_rev, how="outer")

trading_dates = pd.DatetimeIndex(sorted(daily["date"].unique()))
feat_lhb = build_lhb_features(lhb_df, trading_dates)
feat_combo = feat_combo.join(feat_lhb, how="left")
for f in LHB_FACTOR_NAMES + LHB_B2_FACTOR_NAMES:
    if f in feat_combo.columns:
        feat_combo[f] = feat_combo[f].fillna(0)


def _z(s):
    mu, sd = s.mean(), s.std()
    return (s - mu) / sd if sd > 0 else s * 0


feat_z = feat_combo.groupby(level="date").transform(_z).clip(-3, 3).fillna(0)

# label horizon=30
df = daily.copy()
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["code", "date"])
open_next = df.groupby("code")["open"].shift(-1)
close_fwd = df.groupby("code")["close"].shift(-30)
df["label"] = (close_fwd / open_next - 1).clip(-0.5, 0.5)
y = df.set_index(["date", "code"])["label"]

aligned = feat_z.join(y.rename("label"), how="inner")
feat_cols = [c for c in feat_z.columns if c in aligned.columns]
X = aligned[feat_cols]
y = aligned["label"]
mask = y.notna() & X.notna().all(axis=1)
X, y = X[mask], y[mask]

# 全样本训练一个模型, 只看 importance
try:
    import lightgbm as lgb
    m = lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.04,
        num_leaves=31, max_depth=6, min_child_samples=80,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1,
    )
    m.fit(X.values, y.values)
    imp = pd.Series(m.feature_importances_, index=feat_cols).sort_values(ascending=False)
except (OSError, ImportError):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.inspection import permutation_importance
    m = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.04, max_leaf_nodes=31, max_depth=6,
        min_samples_leaf=80, l2_regularization=0.1, random_state=42,
    )
    m.fit(X.values, y.values)
    # 用 permutation importance (较慢, 样本子集)
    sub = np.random.choice(len(X), size=min(20000, len(X)), replace=False)
    pi = permutation_importance(m, X.iloc[sub].values, y.iloc[sub].values,
                                 n_repeats=3, random_state=42, n_jobs=-1)
    imp = pd.Series(pi.importances_mean, index=feat_cols).sort_values(ascending=False)

print(f"\n总因子数: {len(feat_cols)}")
print(f"B2 因子总和占比: {imp[imp.index.isin(LHB_B2_FACTOR_NAMES)].sum() / imp.sum():.2%}")
print(f"B1 基础龙虎榜占比: {imp[imp.index.isin(LHB_FACTOR_NAMES)].sum() / imp.sum():.2%}")

print(f"\nTop 20 重要因子:")
for i, (name, v) in enumerate(imp.head(20).items(), 1):
    b2 = " [B2]" if name in LHB_B2_FACTOR_NAMES else ""
    b1 = " [B1]" if name in LHB_FACTOR_NAMES else ""
    print(f"  {i:2}. {name:22s} {v:>10.4f}{b2}{b1}")

print(f"\n所有 B2 因子的排名:")
for name in LHB_B2_FACTOR_NAMES:
    if name in imp.index:
        rank = imp.index.get_loc(name) + 1
        print(f"  {name:22s} rank={rank:3}  imp={imp[name]:.4f}")
