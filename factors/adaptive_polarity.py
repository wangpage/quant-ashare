"""B+ 自适应因子极性 (Regime-aware, 4 层防护).

解决的问题: 单一方向的因子会在 regime 切换时失效 (V5 hold-out IR -1.48 教训).

4 层防护:
    1. IC z-score: ic_mean / ic_std, 衡量因子有效性 **置信度**, 不是绝对值
    2. 显著性过滤: |z| < threshold → weight=0 (不用噪音因子)
    3. 惯性: w_t = inertia × w_{t-1} + (1-inertia) × w_raw
       防止 IC 轻微波动就翻多翻空, 频繁换仓
    4. 横截面归一化: Σ|w| = 1, 避免某因子权重爆炸
    5. IC 时间衰减 (可选): 近期 IC 权重大, 线性/指数衰减

严格样本外: rolling IC 用 shift(horizon) 避免 future leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_cross_sectional_ic(feat_z: pd.DataFrame,
                                 label: pd.Series) -> pd.DataFrame:
    """每天每个因子的截面 IC. 返回 DataFrame (index=date, cols=factors)."""
    dates = feat_z.index.get_level_values("date").unique().sort_values()
    ic_data = {col: [] for col in feat_z.columns}
    for dt in dates:
        try:
            f_d = feat_z.xs(dt, level="date")
            y_d = label.loc[dt]
            idx = f_d.index.intersection(y_d.index)
            y_aligned = y_d.loc[idx]
            if len(idx) < 20 or y_aligned.notna().sum() < 20:
                for col in feat_z.columns:
                    ic_data[col].append(np.nan)
                continue
            for col in feat_z.columns:
                x = f_d[col].loc[idx]
                # pandas Series.corr 自动处理 NaN
                ic_data[col].append(x.corr(y_aligned, method="spearman"))
        except KeyError:
            for col in feat_z.columns:
                ic_data[col].append(np.nan)
    return pd.DataFrame(ic_data, index=dates)


def compute_adaptive_weights(
    ic_df: pd.DataFrame,
    horizon: int = 30,
    window: int = 60,
    z_threshold: float = 1.5,
    z_cap: float = 3.0,
    inertia: float = 0.7,
    decay_lambda: float = 0.02,   # 0 = 无衰减; 0.02 = 半衰期 ~35 天
) -> pd.DataFrame:
    """4-5 层防护的极性权重.

    输入: ic_df (date, factors) - 每日因子 IC
    输出: weight_df (date, factors) - 每日每因子的带符号权重 (~[-3, 3])

    关键: shift(horizon) 确保时刻 t 只用 t-horizon 之前的 IC
    """
    # ---- Step 5 (加分): IC 指数衰减加权 ----
    if decay_lambda > 0:
        # EMA 版本的滚动均值/方差, alpha = decay_lambda
        ic_mean = ic_df.ewm(alpha=decay_lambda, min_periods=30, adjust=True).mean().shift(horizon)
        ic_std = ic_df.ewm(alpha=decay_lambda, min_periods=30, adjust=True).std().shift(horizon)
    else:
        ic_mean = ic_df.rolling(window, min_periods=30).mean().shift(horizon)
        ic_std = ic_df.rolling(window, min_periods=30).std().shift(horizon)

    # ---- Step 1: IC z-score (置信度) ----
    ic_z = ic_mean / (ic_std + 1e-6)

    # ---- Step 2: 显著性过滤 + cap ----
    raw = np.sign(ic_z) * np.minimum(np.abs(ic_z), z_cap)
    raw = raw.where(ic_z.abs() >= z_threshold, 0.0)

    # ---- Step 3: 惯性 (EMA 平滑) ----
    # w_t = inertia * w_{t-1} + (1-inertia) * raw_t
    # 对应 ewm(alpha=1-inertia)
    alpha = 1.0 - inertia
    weight = raw.ewm(alpha=alpha, adjust=False).mean()

    # ---- Step 4: 横截面归一化 (Σ|w| = 1 每天) ----
    weight_norm = weight.div(weight.abs().sum(axis=1) + 1e-6, axis=0)

    return weight_norm


def apply_adaptive_polarity(
    feat_z: pd.DataFrame,
    label: pd.Series,
    horizon: int = 30,
    window: int = 60,
    z_threshold: float = 1.5,
    z_cap: float = 3.0,
    inertia: float = 0.7,
    decay_lambda: float = 0.02,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """对因子面板应用自适应极性权重.

    Returns:
        feat_adaptive: shape 同 feat_z, 每个值 = raw_z_score × adaptive_weight
        weight_df: 每日每因子的权重 (诊断用)
    """
    print("  计算每日截面 IC...")
    ic_df = compute_cross_sectional_ic(feat_z, label)
    print(f"  IC 矩阵 {ic_df.shape}, 有效 IC 占比 {ic_df.notna().mean().mean():.1%}")

    print("  计算自适应权重 (4+1 层防护)...")
    w = compute_adaptive_weights(
        ic_df, horizon=horizon, window=window,
        z_threshold=z_threshold, z_cap=z_cap,
        inertia=inertia, decay_lambda=decay_lambda,
    )

    # 应用: 按 date 广播权重到 (date, code) panel
    dates_of_rows = feat_z.index.get_level_values("date")
    w_expanded = w.reindex(dates_of_rows).values   # (N_rows, n_factors)
    adj = feat_z.values * w_expanded
    feat_adaptive = pd.DataFrame(adj, index=feat_z.index,
                                   columns=feat_z.columns).fillna(0)

    # 诊断
    non_zero = (w.abs() > 1e-4).sum(axis=1)
    print(f"  每日活跃因子数 (非零权重): mean={non_zero.mean():.1f}, "
          f"min={non_zero.min()}, max={non_zero.max()}")
    avg_flip = (np.sign(w).diff().abs() > 0.5).sum(axis=1).mean()
    print(f"  每日平均因子翻转数: {avg_flip:.2f} (越小越稳)")

    return feat_adaptive, w
