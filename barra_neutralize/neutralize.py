"""风格 + 行业中性化 - 把 alpha 从"风格 beta" 中剥离出来."""
from __future__ import annotations

import numpy as np
import pandas as pd


def industry_dummies(industries: pd.Series) -> pd.DataFrame:
    """把行业标签转成 one-hot dummies.

    A股一级行业 (申万/中信): 约 30+ 个.
    """
    return pd.get_dummies(industries, prefix="ind", dtype=float)


def neutralize_by_regression(
    alpha: pd.Series,
    style_factors: pd.DataFrame,
    industries: pd.Series | None = None,
    weights: pd.Series | None = None,
) -> pd.Series:
    """对风格 + 行业回归取残差.

    核心公式:
        alpha_raw = β_1 × Size + β_2 × Beta + ... + β_ind × Industry + ε
        alpha_clean = ε   (真正属于你的信号)

    Args:
        alpha: 原始 alpha 值, index=stock_code
        style_factors: 风格因子 DataFrame
        industries: 行业标签 Series (可选)
        weights: WLS 加权 (通常用 sqrt(市值))

    Returns:
        中性化后 alpha (回归残差).
    """
    common = alpha.dropna().index.intersection(style_factors.dropna().index)
    if len(common) < 20:
        return alpha.copy()

    y = alpha.loc[common].values
    X_parts = [style_factors.loc[common].values]

    if industries is not None:
        ind_dum = industry_dummies(industries.loc[common])
        X_parts.append(ind_dum.values)

    X = np.hstack(X_parts)
    X = np.hstack([np.ones((X.shape[0], 1)), X])     # 加 intercept

    if weights is not None:
        w = weights.loc[common].values
        w = np.sqrt(w / w.sum())
        X_w = X * w[:, None]
        y_w = y * w
        coef, *_ = np.linalg.lstsq(X_w, y_w, rcond=None)
    else:
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)

    y_pred = X @ coef
    residual = y - y_pred
    return pd.Series(residual, index=common)


def neutralize_one_stock(
    alpha_value: float,
    stock_styles: dict[str, float],
    style_betas: dict[str, float],
) -> float:
    """对单支股票做快速中性化.

    场景: 实时交易中, 用历史拟合的 style_betas 直接对新 alpha 去风格.

    Args:
        alpha_value: 该股票今日 alpha
        stock_styles: 该股票今日的风格暴露 {'Size': 0.5, ...}
        style_betas: 历史回归得到的系数 (risk model 拟合输出)

    Returns:
        去暴露后的 alpha.
    """
    adjustment = sum(style_betas.get(k, 0) * v
                     for k, v in stock_styles.items())
    return alpha_value - adjustment


def explained_variance_by_styles(
    alpha: pd.Series, style_factors: pd.DataFrame,
) -> float:
    """衡量你的 alpha 有多少其实是风格暴露.

    > 30%: 你的 alpha 主要来自风格 beta, 不是真实 alpha, 危险.
    < 10%: 基本独立于风格, 优秀.
    """
    common = alpha.dropna().index.intersection(style_factors.dropna().index)
    if len(common) < 20:
        return 0.0
    y = alpha.loc[common].values
    X = style_factors.loc[common].values
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ coef
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)
