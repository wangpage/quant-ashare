"""风格 + 行业中性化 - 把 alpha 从"风格 beta"中剥离出来.

本模块提供两套接口:

1. neutralize_by_regression (向后兼容)
   单步 WLS + 岭回归, 内置横截面 winsorize + 标准化.

2. neutralize_hierarchical (推荐, CNE5 分层结构)
   step 1 行业中性 (组内 demean)
   step 2 风格正交化 (Gram-Schmidt)
   step 3 加权岭回归, 取残差

两套方法共同的工程级细节:
    - MAD 缩尾抗极值
    - Median/MAD 稳健标准化
    - Ridge α 对抗小样本不稳定
    - 条件数诊断, > 100 发出警告
    - sqrt(mcap) 加权, 符合 Barra WLS 约定
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from utils.logger import logger

from .preprocess import (
    condition_number,
    cross_section_standardize,
    orthogonalize_factors,
    preprocess_factor,
    preprocess_factor_matrix,
    winsorize_mad,
)


_MIN_SAMPLES = 20          # 小于此截面样本不做中性化
_COND_WARN_THRESH = 100.0  # 条件数警告阈值
_DEFAULT_RIDGE_ALPHA = 1e-3


@dataclass
class NeutralizeDiagnostics:
    """单次中性化的诊断信息, 便于回测/研究期间排查."""

    n_samples: int
    n_style_factors: int
    n_industry_factors: int
    condition_number: float
    r_squared: float
    ridge_alpha: float
    residual_std: float
    warnings: list[str]


def industry_dummies(industries: pd.Series) -> pd.DataFrame:
    """行业 one-hot. 避免共线性: 不做 drop_first, 回归时与 intercept 配合.

    为了避免 "intercept + full dummies" 共线, 我们在回归中 **不加 intercept**,
    改由 dummies 自身承担截距. 这样 β_ind 可直接解释为"行业平均 alpha".
    """
    return pd.get_dummies(industries, prefix="ind", dtype=float)


def _ridge_solve(
    X: np.ndarray, y: np.ndarray, alpha: float, weights: np.ndarray | None = None,
) -> np.ndarray:
    """(X'WX + αI) β = X'Wy 的闭式解. α=0 退化为 WLS."""
    if weights is not None:
        w_sqrt = np.sqrt(weights)
        Xw = X * w_sqrt[:, None]
        yw = y * w_sqrt
    else:
        Xw, yw = X, y
    n_feat = Xw.shape[1]
    XtX = Xw.T @ Xw + alpha * np.eye(n_feat)
    Xty = Xw.T @ yw
    try:
        return np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        coef, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        return coef


def _normalize_weights(weights: pd.Series | None, index: pd.Index) -> np.ndarray | None:
    """把加权 Series 归一到均值 1, 满足 Σw = N."""
    if weights is None:
        return None
    w = weights.reindex(index).astype(float).fillna(weights.median())
    w = w.clip(lower=1e-9)
    w = w / w.mean()
    return w.values


def neutralize_by_regression(
    alpha: pd.Series,
    style_factors: pd.DataFrame,
    industries: pd.Series | None = None,
    weights: pd.Series | None = None,
    ridge_alpha: float = _DEFAULT_RIDGE_ALPHA,
    winsorize_n: float = 5.0,
    return_diagnostics: bool = False,
) -> pd.Series | tuple[pd.Series, NeutralizeDiagnostics]:
    """单步回归中性化 (向后兼容入口).

    相比旧版增加:
        - MAD winsorize + 稳健标准化 (alpha 与 style_factors)
        - Ridge α, 避免 X 接近奇异时系数爆炸
        - WLS 权重归一化, sqrt(mcap) 的常见口径
        - 条件数诊断 + 警告
        - 不再同时加 intercept + full industry dummies (共线性)

    Args:
        alpha: 原始 alpha (index=stock_code)
        style_factors: 风格因子 DataFrame
        industries: 行业标签 Series (可选)
        weights: WLS 加权 (通常 sqrt(市值))
        ridge_alpha: 岭系数, 默认 1e-3 (小样本宜用 1e-2)
        winsorize_n: 缩尾 MAD 倍数
        return_diagnostics: True 时返回 (residual, diagnostics)
    """
    warnings: list[str] = []
    common = alpha.dropna().index.intersection(style_factors.dropna(how="all").index)
    if industries is not None:
        common = common.intersection(industries.dropna().index)
    if len(common) < _MIN_SAMPLES:
        empty_diag = NeutralizeDiagnostics(
            n_samples=len(common), n_style_factors=style_factors.shape[1],
            n_industry_factors=0, condition_number=float("inf"),
            r_squared=0.0, ridge_alpha=ridge_alpha, residual_std=0.0,
            warnings=["样本不足, 原值返回"],
        )
        out = alpha.copy()
        return (out, empty_diag) if return_diagnostics else out

    # 横截面预处理 (样本内)
    ind_series = industries.loc[common] if industries is not None else None
    y_raw = alpha.loc[common]
    y = preprocess_factor(y_raw, ind_series, winsorize_n=winsorize_n)
    X_style = preprocess_factor_matrix(
        style_factors.loc[common], ind_series, winsorize_n=winsorize_n,
    )

    X_parts = [X_style.values]
    n_ind = 0
    if industries is not None:
        ind_dum = industry_dummies(ind_series)
        X_parts.append(ind_dum.values)
        n_ind = ind_dum.shape[1]
        # 已有 full dummies → 不加 intercept
        X = np.hstack(X_parts)
    else:
        X = np.hstack([np.ones((X_style.shape[0], 1)), X_style.values])

    # 诊断: 条件数
    cond = condition_number(X)
    if cond > _COND_WARN_THRESH:
        msg = (f"Barra 设计矩阵条件数 {cond:.1f} > {_COND_WARN_THRESH}, "
               f"存在多重共线性, 建议 orthogonalize_factors 或加大 ridge_alpha")
        warnings.append(msg)
        logger.warning(msg)

    w = _normalize_weights(weights, common)
    y_arr = y.values
    coef = _ridge_solve(X, y_arr, ridge_alpha, w)
    y_pred = X @ coef
    residual = y_arr - y_pred

    ss_res = float(np.sum((y_arr - y_pred) ** 2))
    ss_tot = float(np.sum((y_arr - y_arr.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    resid_series = pd.Series(residual, index=common, name=alpha.name or "alpha_resid")

    if return_diagnostics:
        diag = NeutralizeDiagnostics(
            n_samples=len(common),
            n_style_factors=int(X_style.shape[1]),
            n_industry_factors=n_ind,
            condition_number=float(cond),
            r_squared=float(r2),
            ridge_alpha=float(ridge_alpha),
            residual_std=float(resid_series.std()),
            warnings=warnings,
        )
        return resid_series, diag
    return resid_series


def neutralize_hierarchical(
    alpha: pd.Series,
    style_factors: pd.DataFrame,
    industries: pd.Series | None = None,
    weights: pd.Series | None = None,
    ridge_alpha: float = _DEFAULT_RIDGE_ALPHA,
    orthogonalize: bool = True,
    winsorize_n: float = 5.0,
    return_diagnostics: bool = False,
) -> pd.Series | tuple[pd.Series, NeutralizeDiagnostics]:
    """分层中性化 (CNE5 推荐流程):

    step1: 横截面预处理 (winsorize + robust standardize)
    step2: 行业中性 - 组内 demean, 消除行业β
    step3: 风格因子 Gram-Schmidt 正交化 (可选)
    step4: 加权岭回归, 取残差

    相比 neutralize_by_regression, 行业 β 在 step2 就被吸收, 剩余的
    风格回归维度大幅降低, 条件数显著改善.
    """
    warnings: list[str] = []
    common = alpha.dropna().index.intersection(style_factors.dropna(how="all").index)
    if industries is not None:
        common = common.intersection(industries.dropna().index)
    if len(common) < _MIN_SAMPLES:
        diag = NeutralizeDiagnostics(
            n_samples=len(common), n_style_factors=style_factors.shape[1],
            n_industry_factors=0, condition_number=float("inf"),
            r_squared=0.0, ridge_alpha=ridge_alpha, residual_std=0.0,
            warnings=["样本不足, 原值返回"],
        )
        out = alpha.copy()
        return (out, diag) if return_diagnostics else out

    ind_series = industries.loc[common] if industries is not None else None

    # step1 预处理
    y = preprocess_factor(alpha.loc[common], ind_series, winsorize_n=winsorize_n)
    X_style = preprocess_factor_matrix(
        style_factors.loc[common], ind_series, winsorize_n=winsorize_n,
    )

    # step2 行业中性 (组内 demean). 对 alpha 和 style 都做.
    n_ind = 0
    if ind_series is not None:
        y = y - y.groupby(ind_series).transform("mean")
        X_style = X_style.apply(
            lambda c: c - c.groupby(ind_series).transform("mean")
        )
        n_ind = int(ind_series.nunique())

    # step3 风格正交化 (可选, 默认 True)
    if orthogonalize and X_style.shape[1] >= 2:
        X_style = orthogonalize_factors(X_style)

    # step4 岭回归
    X = X_style.values
    cond = condition_number(X)
    if cond > _COND_WARN_THRESH:
        msg = f"分层中性化后条件数仍为 {cond:.1f}, 建议增大 ridge_alpha"
        warnings.append(msg)
        logger.warning(msg)

    w = _normalize_weights(weights, common)
    y_arr = y.values
    coef = _ridge_solve(X, y_arr, ridge_alpha, w)
    y_pred = X @ coef
    residual = y_arr - y_pred

    ss_res = float(np.sum((y_arr - y_pred) ** 2))
    ss_tot = float(np.sum((y_arr - y_arr.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    resid_series = pd.Series(residual, index=common, name=alpha.name or "alpha_resid")

    if return_diagnostics:
        diag = NeutralizeDiagnostics(
            n_samples=len(common),
            n_style_factors=int(X_style.shape[1]),
            n_industry_factors=n_ind,
            condition_number=float(cond),
            r_squared=float(r2),
            ridge_alpha=float(ridge_alpha),
            residual_std=float(resid_series.std()),
            warnings=warnings,
        )
        return resid_series, diag
    return resid_series


def neutralize_one_stock(
    alpha_value: float,
    stock_styles: dict[str, float],
    style_betas: dict[str, float],
) -> float:
    """单只股票快速中性化 (实时场景).

    用历史拟合的 style_betas 直接对新 alpha 去风格. 适用于盘中
    单票新信号 + 回测期系数固定的场景.
    """
    adjustment = sum(
        style_betas.get(k, 0.0) * v for k, v in stock_styles.items()
    )
    return alpha_value - adjustment


def explained_variance_by_styles(
    alpha: pd.Series, style_factors: pd.DataFrame,
    winsorize_n: float = 5.0,
) -> float:
    """衡量 alpha 多少来自风格 beta.

    > 30%: 危险, 主要是风格
    10%-30%: 混合, 需中性化
    < 10%: 相对独立
    """
    common = alpha.dropna().index.intersection(style_factors.dropna(how="all").index)
    if len(common) < _MIN_SAMPLES:
        return 0.0
    y = preprocess_factor(alpha.loc[common], winsorize_n=winsorize_n)
    X_style = preprocess_factor_matrix(
        style_factors.loc[common], winsorize_n=winsorize_n,
    )
    X = np.hstack([np.ones((X_style.shape[0], 1)), X_style.values])
    coef = _ridge_solve(X, y.values, _DEFAULT_RIDGE_ALPHA)
    y_pred = X @ coef
    ss_res = np.sum((y.values - y_pred) ** 2)
    ss_tot = np.sum((y.values - y.values.mean()) ** 2)
    if ss_tot <= 1e-12:
        return 0.0
    return float(1 - ss_res / ss_tot)
