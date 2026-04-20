"""因子预处理 - 去极值 / 标准化 / 缺失填充 / 正交化.

机构级 Barra 中性化前必须做的横截面清洗.

大众误区:
    直接跑 lstsq(y, X) 取残差. 但 X 含极端值时 OLS 被尾部主导,
    β 系数几乎无意义. 残差也不是真正的 alpha.

本模块提供:
    - winsorize_mad: 基于 MAD 的稳健缩尾 (比 std 抗极值)
    - robust_standardize: median/MAD z-score
    - cross_section_standardize: 传统 mean/std z-score
    - orthogonalize_factors: Gram-Schmidt 正交化, 解多重共线性
    - fill_na_by_industry: 行业均值填充
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# 1.4826 = MAD → σ 的一致性系数 (正态分布下 MAD × 1.4826 ≈ std)
_MAD_CONSISTENCY = 1.4826


def winsorize_mad(s: pd.Series, n: float = 5.0) -> pd.Series:
    """基于 MAD 的稳健缩尾.

    比 ±3σ 的"截断式 winsorize"更抗极值: 若单一离群点把 std 拉大,
    std 版本会漏掉真正异常; MAD 不受少数点影响.

    Args:
        s: 横截面因子 (index=stock_code)
        n: 缩尾倍数 (Barra 标准 = 3; 保守策略用 5)
    """
    s = s.astype(float)
    med = s.median()
    mad = (s - med).abs().median()
    if mad == 0 or not np.isfinite(mad):
        std = s.std()
        if std == 0 or not np.isfinite(std):
            return s.copy()
        return s.clip(med - n * std, med + n * std)
    bound = n * _MAD_CONSISTENCY * mad
    return s.clip(med - bound, med + bound)


def robust_standardize(s: pd.Series) -> pd.Series:
    """median/MAD 标准化 - 抗极值版 z-score."""
    s = s.astype(float)
    med = s.median()
    mad = (s - med).abs().median()
    if mad == 0 or not np.isfinite(mad):
        std = s.std()
        if std == 0 or not np.isfinite(std):
            return s * 0.0
        return (s - s.mean()) / std
    return (s - med) / (_MAD_CONSISTENCY * mad)


def cross_section_standardize(s: pd.Series) -> pd.Series:
    """传统横截面 z-score, 用在已 winsorize 过的因子上."""
    s = s.astype(float)
    mu = s.mean()
    sd = s.std()
    if sd == 0 or not np.isfinite(sd):
        return s * 0.0
    return (s - mu) / sd


def fill_na_by_industry(
    s: pd.Series, industries: pd.Series | None = None,
) -> pd.Series:
    """行业均值填充; 无行业信息时退化为全截面均值."""
    s = s.astype(float)
    if industries is None:
        return s.fillna(s.mean())
    aligned_ind = industries.reindex(s.index)
    group_mean = s.groupby(aligned_ind).transform("mean")
    filled = s.fillna(group_mean)
    # 某个行业全部缺失 → 再用全截面均值兜底
    return filled.fillna(filled.mean())


def preprocess_factor(
    s: pd.Series,
    industries: pd.Series | None = None,
    winsorize_n: float = 5.0,
    use_robust: bool = True,
) -> pd.Series:
    """一条龙: 填充 → 缩尾 → 标准化."""
    s = fill_na_by_industry(s, industries)
    s = winsorize_mad(s, n=winsorize_n)
    return robust_standardize(s) if use_robust else cross_section_standardize(s)


def preprocess_factor_matrix(
    df: pd.DataFrame,
    industries: pd.Series | None = None,
    winsorize_n: float = 5.0,
) -> pd.DataFrame:
    """整张因子矩阵的横截面预处理."""
    out = {}
    for col in df.columns:
        out[col] = preprocess_factor(df[col], industries, winsorize_n)
    return pd.DataFrame(out, index=df.index)


def orthogonalize_factors(
    df: pd.DataFrame,
    order: list[str] | None = None,
) -> pd.DataFrame:
    """Gram-Schmidt 正交化: 消除风格因子间的多重共线性.

    CNE5 顺序 (Barra 白皮书): Size → Beta → Momentum → ResidualVol
    → NonLinSize → Liquidity. 后者对前者做回归, 取残差.

    Args:
        df: 因子矩阵 (index=stock_code, cols=factor_name)
        order: 正交化顺序; 未指定则按 df.columns 顺序
    """
    cols = order or list(df.columns)
    X = df[cols].copy().astype(float).fillna(0.0)
    out = pd.DataFrame(index=X.index)
    for i, c in enumerate(cols):
        y = X[c].values
        if i == 0:
            out[c] = y
            continue
        # 对已有正交因子做回归, 取残差
        Z = out.values
        Z_aug = np.hstack([np.ones((len(y), 1)), Z])
        try:
            beta, *_ = np.linalg.lstsq(Z_aug, y, rcond=None)
            y_res = y - Z_aug @ beta
        except np.linalg.LinAlgError:
            y_res = y
        out[c] = y_res
    # 正交化后再次标准化, 保证各因子同尺度
    for c in cols:
        out[c] = cross_section_standardize(pd.Series(out[c], index=out.index))
    return out


def condition_number(X: np.ndarray) -> float:
    """矩阵条件数. > 30 多重共线性警告, > 100 严重病态."""
    try:
        _, sv, _ = np.linalg.svd(X, full_matrices=False)
        sv_min = sv[sv > 1e-12].min() if (sv > 1e-12).any() else 1e-12
        return float(sv.max() / sv_min)
    except np.linalg.LinAlgError:
        return float("inf")
