"""组合构建 - 从"选股"到"建仓位"的最后一英里.

大众: Top-20 等权重.
实际: 你的 alpha 只告诉方向, 仓位要靠组合优化.

三大核心:
    1) 风险平价 (Risk Parity): 每只票对组合总风险的贡献相等
    2) 波动率目标 (Vol Targeting): 组合年化波动率锚定 15%
    3) 约束优化 (Mean-Variance + 约束): 行业 / 换手 / 集中度限制

Bridgewater 全天候策略就是风险平价的代表作.
A股头部量化 (幻方/九坤) 基本都用约束 MVO + 风险预算.
"""
from .risk_parity import (
    risk_parity_weights, inverse_volatility_weights,
)
from .vol_targeting import (
    vol_target_scale, calculate_kelly_with_drawdown,
)
from .mvo import (
    mean_variance_optimize, black_litterman_posterior,
)

__all__ = [
    "risk_parity_weights", "inverse_volatility_weights",
    "vol_target_scale", "calculate_kelly_with_drawdown",
    "mean_variance_optimize", "black_litterman_posterior",
]
