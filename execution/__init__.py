"""执行层 - 从信号到订单的最后一公里.

大众做法: 市价单全量一次性下单.
真相:
    - 开盘前 10 分钟 + 尾盘 15 分钟 冲击成本翻倍
    - 一次性下单 1 亿 Level2 会直接"吃掉"盘口 5 档
    - 不同策略需要不同执行算法: TWAP / VWAP / POV / Implementation Shortfall

本模块提供:
    - 时段避让: 最优交易时段判定
    - TWAP / VWAP 拆单
    - Implementation Shortfall (Almgren-Chriss optimal execution)
    - 基于盘口 Level2 的冲击感知路由
"""
from .time_windows import (
    OptimalTradingWindow, is_tradeable_now,
    avoid_auction_window,
)
from .slicers import (
    TWAPSlicer, VWAPSlicer, POVSlicer,
)
from .impact_router import (
    ImpactAwareRouter, split_order_by_participation,
)
from .simulator import BacktestExecutionSim

__all__ = [
    "OptimalTradingWindow", "is_tradeable_now", "avoid_auction_window",
    "TWAPSlicer", "VWAPSlicer", "POVSlicer",
    "ImpactAwareRouter", "split_order_by_participation",
    "BacktestExecutionSim",
]
