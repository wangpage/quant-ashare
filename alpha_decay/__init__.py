"""因子衰减监控 - 决定你的策略能活多久.

头部私募内部铁律:
    "所有 alpha 都会衰减, 区别只是快慢."

大众量化师: 训完模型上线就不管.
实际做法:
    1) 每日跟踪 rolling_IC / all_time_IC 比值, < 0.5 立即下线
    2) Turnover 上升 + IC 下降 = 因子拥挤的铁证
    3) 多 alpha 组合分散 (低相关因子池), 单 alpha 生命周期 6-18 月
    4) 每月自动 re-rank 因子权重
"""
from .monitor import (
    rolling_ic_decay, half_life_estimate,
    alpha_health_score,
)
from .crowding import (
    factor_crowding_index, turnover_signal,
    public_strategy_overlap,
)

__all__ = [
    "rolling_ic_decay", "half_life_estimate", "alpha_health_score",
    "factor_crowding_index", "turnover_signal", "public_strategy_overlap",
]
