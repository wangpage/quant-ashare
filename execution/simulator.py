"""回测执行仿真 - 可替换 qlib 默认执行器.

区别于大众固定 bps 的执行:
    - 按时段乘数 (OptimalTradingWindow.cost_multiplier)
    - 按冲击模型 (sqrt 律)
    - 按盘口深度 (若有 Level2)
    - 涨跌停 / 停牌 不可成交
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from market_microstructure.impact import square_root_impact
from .time_windows import OptimalTradingWindow


@dataclass
class ExecutionResult:
    requested_shares: int
    filled_shares: int
    avg_fill_price: float
    total_cost_yuan: float
    cost_bps: float
    slippage_bps: float           # vs 基准价
    reason_unfilled: str = ""


class BacktestExecutionSim:
    """更真实的回测执行仿真器."""

    def __init__(
        self,
        default_commission: float = 0.00025,
        stamp_tax_sell: float = 0.001,          # 卖出印花税
        default_volatility: float = 0.02,
        sqrt_law_k: float = 0.5,
    ):
        self.commission = default_commission
        self.stamp_tax = stamp_tax_sell
        self.default_vol = default_volatility
        self.k = sqrt_law_k

    def execute(
        self,
        action: str,                    # 'buy' / 'sell'
        ref_price: float,                # 基准价 (如 open, vwap)
        shares: int,
        daily_volume: int,
        volatility: float | None = None,
        trade_time: datetime | None = None,
        is_limit_up: bool = False,
        is_limit_down: bool = False,
        is_suspended: bool = False,
    ) -> ExecutionResult:
        """单笔执行模拟.

        Returns:
            ExecutionResult, cost 已包含冲击 + 佣金 + 印花税.
        """
        # 不可交易场景
        if is_suspended:
            return ExecutionResult(shares, 0, 0, 0, 0, 0, "停牌")
        if action == "buy" and is_limit_up:
            return ExecutionResult(shares, 0, 0, 0, 0, 0, "涨停买不到")
        if action == "sell" and is_limit_down:
            return ExecutionResult(shares, 0, 0, 0, 0, 0, "跌停卖不掉")

        vol = volatility or self.default_vol
        trade_amount = shares * ref_price

        # 1. 冲击成本 (bps)
        impact_bps = square_root_impact(
            trade_amount, max(daily_volume * ref_price, 1.0),
            vol, k=self.k,
        )
        # 2. 时段乘数
        window_mult = 1.0
        if trade_time is not None:
            window_mult = OptimalTradingWindow.cost_multiplier(trade_time)
            if window_mult >= 99:
                return ExecutionResult(shares, 0, 0, 0, 0, 0, "非交易时段")
        impact_bps *= window_mult

        # 3. 实际成交价 (方向)
        fill_price = ref_price * (
            1 + (impact_bps / 1e4 if action == "buy" else -impact_bps / 1e4)
        )

        # 4. 佣金 / 印花税
        commission_cost = trade_amount * self.commission
        stamp_cost = trade_amount * self.stamp_tax if action == "sell" else 0
        total_cost_yuan = (
            commission_cost + stamp_cost +
            abs(fill_price - ref_price) * shares
        )
        total_cost_bps = total_cost_yuan / trade_amount * 1e4

        slippage = (fill_price - ref_price) / ref_price * 1e4
        if action == "sell":
            slippage = -slippage

        return ExecutionResult(
            requested_shares=shares,
            filled_shares=shares,
            avg_fill_price=float(fill_price),
            total_cost_yuan=float(total_cost_yuan),
            cost_bps=float(total_cost_bps),
            slippage_bps=float(slippage),
        )

    def execute_sliced(
        self,
        action: str,
        slices: list[dict],
        prices_by_time: dict,       # {time_offset_min: ref_price}
        daily_volume: int,
        volatility: float | None = None,
    ) -> dict:
        """对已切分的订单批量执行, 返回综合成本."""
        results = []
        total_filled = 0
        total_cost = 0.0
        total_notional = 0.0

        for s in slices:
            offset = s.get("time_offset_minutes", 0)
            price = prices_by_time.get(offset) or \
                    prices_by_time.get(list(prices_by_time.keys())[0], 0)
            res = self.execute(
                action=action, ref_price=price,
                shares=s["shares"], daily_volume=daily_volume,
                volatility=volatility,
            )
            results.append(res)
            total_filled += res.filled_shares
            total_cost += res.total_cost_yuan
            total_notional += res.filled_shares * res.avg_fill_price

        avg_price = total_notional / max(total_filled, 1)
        return {
            "slices": results,
            "total_filled": total_filled,
            "avg_price": float(avg_price),
            "total_cost_yuan": float(total_cost),
            "effective_cost_bps": float(
                total_cost / max(total_notional, 1) * 1e4
            ),
        }
