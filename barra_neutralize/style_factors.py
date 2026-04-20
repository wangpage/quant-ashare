"""Barra CNE5 风格因子计算 - 简化实现.

参考: MSCI Barra 《CNE5 Model》白皮书 (2012)

每个因子输出跨截面 Z-score 化的值 (mean=0, std=1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _zscore(s: pd.Series) -> pd.Series:
    if s.std() == 0:
        return s * 0
    return (s - s.mean()) / s.std()


def compute_size(market_cap: pd.Series) -> pd.Series:
    """Size = log(总市值)."""
    return _zscore(np.log(market_cap.clip(lower=1)))


def compute_beta(
    stock_returns: pd.DataFrame, market_return: pd.Series,
    window: int = 60, half_life: int = 20,
) -> pd.Series:
    """Beta = EW-回归(r_stock, r_market) 过去 60 日.

    使用半衰期加权, 近期数据权重高.
    """
    # 指数半衰期权重
    w = np.power(0.5, np.arange(window, 0, -1) / half_life)
    w = w / w.sum()

    betas = {}
    for code in stock_returns.columns:
        s = stock_returns[code].tail(window).values
        m = market_return.tail(window).values
        if len(s) < window or len(m) < window:
            betas[code] = np.nan
            continue
        cov = np.sum(w * (s - np.average(s, weights=w)) *
                     (m - np.average(m, weights=w)))
        var = np.sum(w * (m - np.average(m, weights=w)) ** 2)
        betas[code] = cov / var if var > 0 else np.nan
    return _zscore(pd.Series(betas).dropna())


def compute_momentum(
    returns: pd.DataFrame, horizon: int = 252, skip: int = 21,
) -> pd.Series:
    """Momentum = 过去 T-21 到 T-252 的累计收益 (跳过最近 1 月).

    跳过最近 1 月是为了避开反转效应.
    """
    mom = (1 + returns.iloc[-horizon:-skip]).prod() - 1
    return _zscore(mom)


def compute_residual_volatility(
    stock_returns: pd.DataFrame, market_return: pd.Series,
    window: int = 60,
) -> pd.Series:
    """残差波动 = 回归残差的标准差 (个股特质风险)."""
    rv = {}
    for code in stock_returns.columns:
        s = stock_returns[code].tail(window)
        m = market_return.tail(window)
        common = s.index.intersection(m.index)
        if len(common) < 20:
            rv[code] = np.nan
            continue
        cov = np.cov(s.loc[common], m.loc[common])[0, 1]
        var_m = m.loc[common].var()
        beta = cov / var_m if var_m > 0 else 0
        residual = s.loc[common] - beta * m.loc[common]
        rv[code] = residual.std()
    return _zscore(pd.Series(rv).dropna())


def compute_liquidity(
    turnover_1m: pd.Series, turnover_3m: pd.Series, turnover_12m: pd.Series,
) -> pd.Series:
    """Barra Liquidity 复合因子: 三个周期换手率的加权和.

    权重 (Barra CNE5): 0.35 × STOM + 0.35 × STOQ + 0.30 × STOA
    (短期月换手 / 季度 / 年)

    Args:
        turnover_1m, turnover_3m, turnover_12m: 均为 log(月平均换手)
    """
    stom = _zscore(np.log(turnover_1m.clip(lower=1e-6)))
    stoq = _zscore(np.log(turnover_3m.clip(lower=1e-6)))
    stoa = _zscore(np.log(turnover_12m.clip(lower=1e-6)))
    return 0.35 * stom + 0.35 * stoq + 0.30 * stoa


def compute_non_linear_size(size_factor: pd.Series) -> pd.Series:
    """Non-Linear Size = Size^3 中性化后 (捕捉"中等市值" vs 两端).

    操作:
        1. NLS = Size ^ 3
        2. 对 Size 回归, 取残差 (所以叫 'non-linear')

    退化保护: 若 Size 方差 ~0 (外层传了常数 market_cap), 返回零 Series,
    避免 polyfit 抛 SVD 异常, 上游调用 compute_all_styles 时不会整体失败.
    """
    clean = size_factor.dropna()
    if len(clean) < 3 or clean.std() < 1e-12:
        return pd.Series(0.0, index=size_factor.index)
    nls = size_factor ** 3
    # 对 Size 回归取残差
    try:
        beta = np.polyfit(clean.values, (clean ** 3).values, 1)
    except (np.linalg.LinAlgError, ValueError):
        return pd.Series(0.0, index=size_factor.index)
    residual = nls - (beta[0] * size_factor + beta[1])
    return _zscore(residual)


def compute_all_styles(
    market_cap: pd.Series,
    returns: pd.DataFrame,
    market_return: pd.Series,
    turnover_1m: pd.Series,
    turnover_3m: pd.Series,
    turnover_12m: pd.Series,
) -> pd.DataFrame:
    """一次性计算 5 个最重要的 Barra 风格因子.

    Returns:
        DataFrame 列: Size, Beta, Momentum, ResidualVol, Liquidity, NonLinSize
    """
    size = compute_size(market_cap)
    beta = compute_beta(returns, market_return)
    mom = compute_momentum(returns)
    rv = compute_residual_volatility(returns, market_return)
    liq = compute_liquidity(turnover_1m, turnover_3m, turnover_12m)
    nls = compute_non_linear_size(size)

    return pd.DataFrame({
        "Size": size, "Beta": beta, "Momentum": mom,
        "ResidualVol": rv, "Liquidity": liq, "NonLinSize": nls,
    })
