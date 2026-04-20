"""ExecutionEngine - 执行层统一入口.

把所有 Slicer (TWAP/VWAP/POV/ImpactAware) + 时段避让 + 回测仿真
串成一致 API. 上游 (pipeline / daily_trading) 只认 ExecutionEngine,
无需关心细节.

使用:
    engine = ExecutionEngine.default()
    plan = engine.plan(OrderRequest(...))
    result = engine.simulate_backtest(plan)
"""
from __future__ import annotations

from dataclasses import dataclass

from utils.logger import logger

from .base import ExecutionPlan, OrderRequest, Side, Slicer
from .impact_router import ImpactAwareRouter
from .simulator import BacktestExecutionSim
from .slicers import POVSlicer, TWAPSlicer, VWAPSlicer
from .time_windows import OptimalTradingWindow, WindowQuality


_STRATEGY_TABLE: dict[str, type] = {
    "TWAP": TWAPSlicer,
    "VWAPSlicer": VWAPSlicer,
    "VWAP": VWAPSlicer,
    "POV": POVSlicer,
    "ImpactAware": ImpactAwareRouter,
}


def _pick_slicer(strategy: str, **kwargs) -> Slicer:
    cls = _STRATEGY_TABLE.get(strategy)
    if cls is None:
        raise ValueError(
            f"未知切片策略 {strategy}, 支持: {list(_STRATEGY_TABLE.keys())}"
        )
    return cls(**kwargs)


@dataclass
class ExecutionEngine:
    """统一执行入口, 负责:
        1. 按策略选 Slicer
        2. 时段质量门禁 (FORBIDDEN 时拒单)
        3. 调 BacktestExecutionSim 得到成本诊断
    """
    slicer: Slicer
    simulator: BacktestExecutionSim
    reject_forbidden_window: bool = True

    @classmethod
    def default(
        cls,
        strategy: str = "ImpactAware",
        slicer_kwargs: dict | None = None,
        sim_kwargs: dict | None = None,
        reject_forbidden_window: bool = True,
    ) -> "ExecutionEngine":
        slicer = _pick_slicer(strategy, **(slicer_kwargs or {}))
        sim = BacktestExecutionSim(**(sim_kwargs or {}))
        return cls(
            slicer=slicer, simulator=sim,
            reject_forbidden_window=reject_forbidden_window,
        )

    def plan(self, request: OrderRequest) -> ExecutionPlan:
        """产出执行计划.  FORBIDDEN 时段拒单."""
        if request.start_time is not None and self.reject_forbidden_window:
            q = OptimalTradingWindow.classify(request.start_time)
            if q == WindowQuality.FORBIDDEN:
                logger.warning(
                    f"拒单: {request.code} 在非交易时段 {request.start_time}"
                )
                return ExecutionPlan(
                    request=request, slices=[],
                    total_cost_bps=0, total_cost_yuan=0, duration_minutes=0,
                    strategy=getattr(self.slicer, "name", "unknown"),
                    notes="FORBIDDEN 时段拒单",
                )
        return self.slicer.slice(request)

    def simulate_backtest(
        self, plan: ExecutionPlan, fill_prices: dict[float, float] | None = None,
    ) -> dict:
        """回测一次性消化 plan 中所有 slice, 返回聚合成交结果.

        Args:
            fill_prices: {time_offset_min: actual_ref_price}, 若无则用 request.ref_price
        """
        if not plan.slices:
            return {"filled": 0, "avg_price": 0.0, "cost_yuan": 0.0,
                    "cost_bps": 0.0, "reason": "无切片"}

        req = plan.request
        prices = fill_prices or {0.0: req.ref_price}

        total_filled = 0
        total_cost_yuan = 0.0
        total_notional = 0.0
        for s in plan.slices:
            price = prices.get(s.time_offset_minutes, req.ref_price)
            res = self.simulator.execute(
                action=req.side.value if isinstance(req.side, Side) else str(req.side),
                ref_price=price,
                shares=s.shares,
                daily_volume=max(
                    int(req.daily_volume or
                        int((req.adv_yuan or 1e8) / max(price, 1))),
                    1,
                ),
                volatility=req.volatility,
                trade_time=s.start_time,
            )
            total_filled += res.filled_shares
            total_cost_yuan += res.total_cost_yuan
            total_notional += res.filled_shares * res.avg_fill_price
        avg_price = total_notional / max(total_filled, 1)
        return {
            "filled": int(total_filled),
            "avg_price": float(avg_price),
            "cost_yuan": float(total_cost_yuan),
            "cost_bps": float(total_cost_yuan / max(total_notional, 1) * 1e4),
            "strategy": plan.strategy,
        }
