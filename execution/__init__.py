"""执行层 - 从信号到订单的最后一公里.

## 统一接口

推荐上游代码通过 ``ExecutionEngine`` 访问本模块:

    from execution import ExecutionEngine, OrderRequest, Side

    engine = ExecutionEngine.default(strategy="ImpactAware")
    plan = engine.plan(OrderRequest(
        code="600519", side=Side.BUY, total_shares=1000,
        ref_price=1680.0, adv_yuan=5e9, volatility=0.02,
    ))

所有 Slicer 都实现 ``execution.base.Slicer`` 协议.

## 子模块

- base.py         - OrderRequest / OrderSlice / ExecutionPlan / Slicer 协议
- time_windows.py - 最优交易时段
- slicers.py      - TWAP / VWAP / POV
- impact_router.py - ImpactAwareRouter (sqrt 律 + 盘口 + ADV 20 日口径)
- simulator.py    - 回测执行仿真
- engine.py       - ExecutionEngine, 统一入口
"""
from .base import (
    OrderRequest, OrderSlice, ExecutionPlan, Side, Slicer,
    clip_to_trading_session, effective_trading_minutes,
)
from .time_windows import (
    OptimalTradingWindow, WindowQuality,
    is_tradeable_now, avoid_auction_window,
)
from .slicers import (
    TWAPSlicer, VWAPSlicer, POVSlicer,
)
from .impact_router import (
    ImpactAwareRouter, RoutingPlan,
    split_order_by_participation, estimate_adv_yuan,
)
from .simulator import BacktestExecutionSim, ExecutionResult
from .engine import ExecutionEngine

__all__ = [
    "OrderRequest", "OrderSlice", "ExecutionPlan", "Side", "Slicer",
    "clip_to_trading_session", "effective_trading_minutes",
    "OptimalTradingWindow", "WindowQuality",
    "is_tradeable_now", "avoid_auction_window",
    "TWAPSlicer", "VWAPSlicer", "POVSlicer",
    "ImpactAwareRouter", "RoutingPlan",
    "split_order_by_participation", "estimate_adv_yuan",
    "BacktestExecutionSim", "ExecutionResult",
    "ExecutionEngine",
]
