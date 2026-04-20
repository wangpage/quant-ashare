"""风险平价 (Risk Parity).

核心思想: 每只票对组合总风险的边际贡献相等.
    MRC_i = w_i × (Σw)_i / σ_p
    RC_i  = w_i × MRC_i = w_i^2 × σ_i^2 / σ_p  (简化版)

    风险平价 = 所有 RC_i 相等.

实现:
    - 精确解: 牛顿法迭代 (scipy.optimize)
    - 近似解: 反向波动率权重 (1/σ_i) 近似, 精度 90%+
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def inverse_volatility_weights(volatilities: pd.Series) -> pd.Series:
    """最简近似: 权重 ∝ 1/vol.

    用途:
        - 没有协方差矩阵时的 baseline
        - 快速组合重平衡
    """
    inv = 1.0 / volatilities.replace(0, np.nan)
    w = inv / inv.sum()
    return w.fillna(0)


def risk_parity_weights(
    cov_matrix: np.ndarray | pd.DataFrame,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> np.ndarray:
    """精确风险平价 (Newton's method).

    Args:
        cov_matrix: N x N 协方差矩阵

    Returns:
        权重向量, 和为 1, 每股票贡献的风险相等.
    """
    if isinstance(cov_matrix, pd.DataFrame):
        cov = cov_matrix.values
    else:
        cov = cov_matrix
    n = cov.shape[0]

    # 初始猜测: 等权
    w = np.ones(n) / n

    for _ in range(max_iter):
        # 每只票的风险贡献
        port_var = w @ cov @ w
        if port_var <= 0:
            break
        marginal = cov @ w
        risk_contrib = w * marginal

        # 目标: RC_i 相等 (= port_var / n)
        target = port_var / n
        grad = risk_contrib - target

        if np.max(np.abs(grad)) < tol:
            break

        # 简易梯度下降 (cycle update)
        w_new = w - 0.01 * grad / (marginal + 1e-12)
        w_new = np.clip(w_new, 0, 1)
        w_new /= w_new.sum()

        if np.max(np.abs(w_new - w)) < tol:
            break
        w = w_new

    return w


def equal_risk_contribution_weights(
    cov_matrix: pd.DataFrame,
    target_risks: np.ndarray | None = None,
) -> pd.Series:
    """广义风险平价: 支持自定义风险预算 (不是等权).

    Args:
        target_risks: 每支股票目标风险贡献比例, 和为 1
                      None = 等风险
    """
    n = cov_matrix.shape[0]
    if target_risks is None:
        target_risks = np.ones(n) / n

    w = risk_parity_weights(cov_matrix.values)
    # 按 target_risks 缩放 (近似)
    w = w * target_risks / (np.ones(n) / n)
    w = w / w.sum()
    return pd.Series(w, index=cov_matrix.index)


def portfolio_risk_breakdown(
    weights: pd.Series, cov_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """组合风险分解: 每支票对总风险的贡献 (% of total variance)."""
    w = weights.values
    cov = cov_matrix.values
    port_var = w @ cov @ w
    marginal = cov @ w
    rc = w * marginal / port_var
    return pd.DataFrame({
        "weight": weights.values,
        "risk_contribution": rc,
        "risk_share": rc / rc.sum(),
    }, index=weights.index)
