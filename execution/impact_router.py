"""冲击感知路由 - 结合 Level2 盘口和冲击模型动态分单."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from market_microstructure.impact import (
    square_root_impact, almgren_chriss_impact,
)


@dataclass
class RoutingPlan:
    total_shares: int
    slices: list[dict]
    expected_total_cost_bps: float
    expected_total_cost_yuan: float
    duration_minutes: float
    notes: str = ""


class ImpactAwareRouter:
    """根据盘口深度和 Almgren-Chriss 模型动态决定拆单策略."""

    def __init__(
        self,
        target_participation: float = 0.05,      # 占日成交 5%
        max_ticks_per_slice: float = 3.0,          # 单笔最多穿多少 tick
        urgency: float = 0.5,                     # 0=慢但便宜, 1=快但贵
    ):
        self.target_participation = target_participation
        self.max_ticks_per_slice = max_ticks_per_slice
        self.urgency = urgency

    def plan_order(
        self,
        total_shares: int,
        price: float,
        daily_volume: int,          # 股数
        volatility: float,           # 日波动率 (如 0.02)
        book: dict | None = None,    # {'bid': [(p, v), ...], 'ask': [...]}
    ) -> RoutingPlan:
        """生成订单切片计划.

        逻辑:
            1. 估算总冲击 (AC 模型)
            2. 如果冲击 > urgency_tolerance, 拆更多片
            3. 如果盘口深度够, 直接吃市价
            4. 否则分批, 每批不超过 max_ticks × tick_depth
        """
        trade_amount = total_shares * price
        daily_amount = daily_volume * price

        ac = almgren_chriss_impact(
            trade_amount, daily_amount, volatility,
            is_buy=True,
        )

        # 1. 确定切片数
        if ac["total_bps"] < 20:
            n_slices = 3
        elif ac["total_bps"] < 50:
            n_slices = 6
        elif ac["total_bps"] < 100:
            n_slices = 10
        else:
            n_slices = max(15, int(ac["total_bps"] / 10))

        # 急迫性调整
        if self.urgency > 0.7:
            n_slices = max(1, int(n_slices * 0.5))
        elif self.urgency < 0.3:
            n_slices = int(n_slices * 1.5)

        # 2. 估算每片参与率
        per_slice_participation = (
            self.target_participation / n_slices
            if self.target_participation else 1.0 / n_slices
        )

        slice_shares = (total_shares // n_slices // 100) * 100
        remainder = total_shares - slice_shares * n_slices

        slices = []
        total_cost = 0.0
        for i in range(n_slices):
            s = slice_shares + (100 if i < remainder // 100 else 0)
            if s <= 0:
                continue
            slice_cost = square_root_impact(
                s * price, daily_amount, volatility,
            )
            total_cost += slice_cost * s * price / 1e4
            slices.append({
                "slice_index": i,
                "shares": int(s),
                "expected_cost_bps": float(slice_cost),
                "expected_cost_yuan": float(slice_cost * s * price / 1e4),
                "time_offset_minutes": i * (240 / n_slices),   # 4h / n
            })

        # 3. 盘口深度优化
        if book:
            level1_depth = (book.get("ask", [(0, 0)])[0][1]
                            if book.get("ask") else 0)
            if slice_shares > level1_depth * self.max_ticks_per_slice:
                # 警告: 每片太大
                for s in slices:
                    s["warning"] = "超过盘口 N 档, 会穿价"

        total_pct = total_cost / trade_amount * 1e4 if trade_amount else 0
        return RoutingPlan(
            total_shares=total_shares,
            slices=slices,
            expected_total_cost_bps=float(total_pct),
            expected_total_cost_yuan=float(total_cost),
            duration_minutes=float(n_slices * (240 / max(n_slices, 1))),
            notes=f"AC={ac['total_bps']:.1f}bps, "
                  f"urgency={self.urgency}, slices={len(slices)}",
        )


def split_order_by_participation(
    total_shares: int,
    market_volume_forecast: list[int],  # 未来每个时间段预期成交量
    target_participation: float = 0.10,
) -> list[int]:
    """按市场预期成交量拆单, 每段占该时段 X%."""
    slices = []
    remaining = total_shares
    for mv in market_volume_forecast:
        my_share = min(remaining, int(mv * target_participation))
        my_share = (my_share // 100) * 100
        slices.append(my_share)
        remaining -= my_share
        if remaining <= 0:
            break
    # 剩余分布到最后
    if remaining > 0 and slices:
        slices[-1] += (remaining // 100) * 100
    return slices
