"""Mean-Variance 优化 + Black-Litterman (带先验信号).

MVO 标准形式:
    max_w   μ^T w - λ/2 × w^T Σ w
    s.t.    Σ w_i = 1, w_i ∈ [w_min, w_max]

Black-Litterman (1990): 把主观观点 (agent 的 view) 和
市场均衡 (CAPM 隐含收益率) 贝叶斯结合, 得到后验 μ.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def mean_variance_optimize(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_aversion: float = 2.0,
    max_weight: float = 0.10,
    min_weight: float = 0.0,
    sum_to_one: bool = True,
) -> pd.Series:
    """闭式解 MVO (支持简单约束, 无需 cvxpy).

    无约束最优: w = (1/λ) × Σ^{-1} × μ

    本实现:
        1. 求无约束最优
        2. clip 到 [min, max]
        3. 归一化

    Args:
        expected_returns: 预期收益 (alpha 值)
        cov_matrix: 协方差矩阵
        risk_aversion: λ, 越大越保守 (1-5 合理)

    Returns:
        权重 Series
    """
    common = expected_returns.dropna().index.intersection(cov_matrix.index)
    mu = expected_returns.loc[common].values
    sigma = cov_matrix.loc[common, common].values

    # 闭式解
    try:
        inv_sigma = np.linalg.inv(sigma + np.eye(len(mu)) * 1e-8)
    except np.linalg.LinAlgError:
        inv_sigma = np.linalg.pinv(sigma)

    w_raw = (1.0 / risk_aversion) * inv_sigma @ mu

    # 简单约束
    w = np.clip(w_raw, min_weight, max_weight)
    if sum_to_one and w.sum() > 0:
        w = w / w.sum()

    return pd.Series(w, index=common)


def black_litterman_posterior(
    market_weights: pd.Series,      # 市值权重 (如沪深300)
    cov_matrix: pd.DataFrame,
    views: dict[str, float],        # {code: expected_return}
    view_confidence: dict[str, float],  # {code: 0-1}
    risk_aversion: float = 2.0,
    tau: float = 0.05,
) -> pd.Series:
    """Black-Litterman 后验期望收益.

    Args:
        market_weights: 市场隐含权重 (N 只票)
        cov_matrix: N x N 协方差
        views: {code: 你的预期收益}, 可以是部分股票
        view_confidence: 每个 view 的信心 [0, 1]
        tau: 先验不确定性, 经验 0.025-0.05

    Returns:
        融合后的 μ (给 MVO 用)
    """
    codes = cov_matrix.index.tolist()
    sigma = cov_matrix.loc[codes, codes].values
    w_mkt = market_weights.reindex(codes).fillna(0).values

    # 1. CAPM 隐含收益率 (prior)
    pi = risk_aversion * sigma @ w_mkt

    # 2. Views
    view_codes = [c for c in views if c in codes]
    if not view_codes:
        return pd.Series(pi, index=codes)

    P = np.zeros((len(view_codes), len(codes)))
    q = np.zeros(len(view_codes))
    omega = np.zeros((len(view_codes), len(view_codes)))
    for i, c in enumerate(view_codes):
        P[i, codes.index(c)] = 1.0
        q[i] = views[c]
        conf = view_confidence.get(c, 0.5)
        var = tau * sigma[codes.index(c), codes.index(c)] * (1 - conf) / max(conf, 0.01)
        omega[i, i] = var

    # 3. 后验
    tau_sigma = tau * sigma
    try:
        inv1 = np.linalg.inv(tau_sigma)
        inv2 = np.linalg.inv(omega + np.eye(len(view_codes)) * 1e-8)
        posterior_cov = np.linalg.inv(inv1 + P.T @ inv2 @ P)
        posterior_mean = posterior_cov @ (inv1 @ pi + P.T @ inv2 @ q)
    except np.linalg.LinAlgError:
        return pd.Series(pi, index=codes)

    return pd.Series(posterior_mean, index=codes)
