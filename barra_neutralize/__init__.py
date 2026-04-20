"""Barra 风格因子中性化 - 大众以为做了中性化, 实际没做透.

大众做法: "减去市值 + 行业 dummy"  (2 维中性化)
Barra 做法: "回归 10 个风格因子 + 行业 30+ 维 dummy, 取残差" (40+ 维)

Barra USE4 / CNE5 风格因子:
    1. Size (对数市值)
    2. Beta (60日回归 beta)
    3. Momentum (12 - 1 月收益)
    4. Residual Volatility (特质波动率)
    5. Non-Linear Size (市值的立方项)
    6. Book-to-Price (PB 倒数)
    7. Liquidity (1/3/12 月换手)
    8. Earnings Yield (PE 倒数, 预期 EPS)
    9. Growth (历史营收/利润成长)
    10. Leverage (杠杆率)

中性化意义:
    你以为找到了 alpha, 实际只是 "Size × 1" 或 "Momentum × 1",
    赚的是风格轮动的钱, 不是真 alpha.
    消掉风格暴露后, 残差才是你的真实 alpha.
"""
from .style_factors import (
    compute_size, compute_beta, compute_momentum,
    compute_residual_volatility, compute_liquidity,
    compute_all_styles,
)
from .neutralize import (
    neutralize_by_regression, neutralize_hierarchical,
    neutralize_one_stock, industry_dummies,
    explained_variance_by_styles, NeutralizeDiagnostics,
)
from .preprocess import (
    winsorize_mad, robust_standardize, cross_section_standardize,
    orthogonalize_factors, preprocess_factor, preprocess_factor_matrix,
    condition_number,
)

__all__ = [
    "compute_size", "compute_beta", "compute_momentum",
    "compute_residual_volatility", "compute_liquidity",
    "compute_all_styles",
    "neutralize_by_regression", "neutralize_hierarchical",
    "neutralize_one_stock", "industry_dummies",
    "explained_variance_by_styles", "NeutralizeDiagnostics",
    "winsorize_mad", "robust_standardize", "cross_section_standardize",
    "orthogonalize_factors", "preprocess_factor", "preprocess_factor_matrix",
    "condition_number",
]
