"""公司行为事件 - 回测里被忽视但实盘必杀的陷阱.

A股特有事件:
    1) 财报日 ±2 天: 已知信息提前交易, 回测会虚高
    2) 解禁日: 前 10 天股价率先反应, 因子失效
    3) 大宗交易折价: > 3% 折价是内部人套现, 利空信号
    4) 增发 / 可转债定增: 稀释预期, 利空
    5) 股东减持公告: 3% 以上减持 6 个月内利空显著
    6) 送股除权: 价格调整, 不做处理就会错误触发止损
    7) 商誉暴雷: 前次财报 ROE 异常下降的标

这些事件的时间分布是明确的, 建议在训练集剔除相关窗口的样本.
"""
from .event_mask import (
    EarningsMask, UnlockMask, BlockTradeMask,
    combined_event_mask,
)
from .event_factors import (
    unlock_pressure_factor, block_trade_discount_factor,
    insider_net_activity_factor,
)

__all__ = [
    "EarningsMask", "UnlockMask", "BlockTradeMask",
    "combined_event_mask",
    "unlock_pressure_factor", "block_trade_discount_factor",
    "insider_net_activity_factor",
]
