"""订单拆分算法: TWAP / VWAP / POV / ImpactAware.

所有 Slicer 都实现 `execution.base.Slicer` 协议:
    slicer.slice(OrderRequest) -> ExecutionPlan

策略定位:
    TWAP: 时间均匀, 不看量, 小单慢策略
    VWAP: 按历史日内成交量曲线分布, 机构标配
    POV : 跟随当前成交量, 动态
    ImpactAware: 基于 sqrt 律 + 盘口深度 + 目标参与率, 推荐
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta

import numpy as np
import pandas as pd

from market_microstructure.impact import square_root_impact

from .base import (
    ExecutionPlan, OrderRequest, OrderSlice, Side, Slicer,
    clip_to_trading_session, effective_trading_minutes,
)


def _cost_of_slice(
    shares: int, price: float, adv_yuan: float, volatility: float,
    n_slices: int, slicer_name: str,
) -> tuple[float, float]:
    """单片成本. eta/gamma 口径同 ImpactAwareRouter."""
    notional = shares * price
    one_shot_bps = square_root_impact(notional * n_slices, adv_yuan, volatility)
    eta, gamma = 0.65, 0.35
    sliced_bps_total = one_shot_bps * (eta / max(n_slices, 1) ** 0.5 + gamma)
    # 按均匀分摊给本片 (调用方保证切片均匀 / 按权重传 shares)
    per_slice_bps = sliced_bps_total / max(n_slices, 1)
    per_slice_yuan = per_slice_bps / 1e4 * (notional * n_slices) / max(n_slices, 1)
    return float(per_slice_bps), float(per_slice_yuan)


def _default_adv_yuan(req: OrderRequest) -> float:
    if req.adv_yuan and req.adv_yuan > 0:
        return req.adv_yuan
    if req.daily_volume and req.daily_volume > 0:
        return req.daily_volume * req.ref_price
    # 兜底: 用订单金额 × 10 作为"假设 ADV", 避免除零但保守估冲击
    return max(req.total_shares * req.ref_price * 10, 1.0)


class TWAPSlicer:
    """时间均匀拆单."""
    name = "TWAP"

    def __init__(self, n_slices: int = 10,
                  avoid_opening_min: int = 10, avoid_closing_min: int = 15):
        self.n_slices = n_slices
        self.avoid_opening_min = avoid_opening_min
        self.avoid_closing_min = avoid_closing_min

    def slice(self, req: OrderRequest) -> ExecutionPlan:
        if req.start_time is None or req.end_time is None:
            raise ValueError("TWAPSlicer 需要 start_time 和 end_time")

        eff_start, eff_end = clip_to_trading_session(
            req.start_time, req.end_time,
            self.avoid_opening_min, self.avoid_closing_min,
        )
        if eff_end <= eff_start:
            return ExecutionPlan(
                request=req, slices=[], total_cost_bps=0, total_cost_yuan=0,
                duration_minutes=0, strategy=self.name,
                notes="有效交易时段为空",
            )

        total_min = effective_trading_minutes(eff_start, eff_end)
        n = max(1, self.n_slices)
        per_slice_min = total_min / n
        per_slice_shares = (req.total_shares // n // 100) * 100
        remainder = req.total_shares - per_slice_shares * n
        remainder_lots = remainder // 100

        adv_yuan = _default_adv_yuan(req)

        slices: list[OrderSlice] = []
        cur = eff_start
        lunch_s = datetime.combine(eff_start.date(), time(11, 30))
        for i in range(n):
            shares = per_slice_shares + (100 if i < remainder_lots else 0)
            if shares <= 0:
                cur = cur + timedelta(minutes=per_slice_min)
                continue
            nxt = cur + timedelta(minutes=per_slice_min)
            if cur < lunch_s < nxt:
                nxt += timedelta(minutes=90)
            bps, yuan = _cost_of_slice(
                shares, req.ref_price, adv_yuan, req.volatility, n, self.name,
            )
            slices.append(OrderSlice(
                slice_index=i, shares=int(shares),
                start_time=cur, end_time=nxt,
                time_offset_minutes=(cur - eff_start).total_seconds() / 60,
                expected_participation=1.0 / n,
                expected_cost_bps=bps, expected_cost_yuan=yuan,
                notes=self.name,
            ))
            cur = nxt

        total_bps = sum(s.expected_cost_bps * s.shares for s in slices)
        total_bps = total_bps / max(req.total_shares, 1)
        total_yuan = sum(s.expected_cost_yuan for s in slices)
        return ExecutionPlan(
            request=req, slices=slices,
            total_cost_bps=float(total_bps), total_cost_yuan=float(total_yuan),
            duration_minutes=float(total_min),
            participation_rate=float(
                req.total_shares * req.ref_price / max(adv_yuan, 1.0)
            ),
            strategy=self.name,
        )


class VWAPSlicer:
    """按历史 U 型成交量曲线分布拆单."""
    name = "VWAP"

    def __init__(self, volume_curve: pd.Series | None = None, n_slices: int = 20):
        self.n_slices = n_slices
        self.volume_curve = (
            volume_curve if volume_curve is not None
            else self._default_a_share_curve()
        )

    @staticmethod
    def _default_a_share_curve() -> pd.Series:
        curve = np.ones(240)
        curve[0:15] = 3.5
        curve[15:30] = 1.5
        curve[30:100] = 0.8
        curve[100:120] = 1.1
        curve[120:140] = 0.9
        curve[140:210] = 0.75
        curve[210:240] = 2.0
        curve = curve / curve.sum()
        idx = pd.RangeIndex(0, 240, name="min_of_session")
        return pd.Series(curve, index=idx)

    def slice(self, req: OrderRequest) -> ExecutionPlan:
        start = req.start_time or datetime.combine(
            datetime.today().date(), time(9, 30)
        )
        n = max(1, self.n_slices)
        minutes_per_slice = 240 // n
        weights = []
        for i in range(n):
            s0 = i * minutes_per_slice
            s1 = min((i + 1) * minutes_per_slice, 240)
            weights.append(float(self.volume_curve.iloc[s0:s1].sum()))
        total_w = sum(weights) or 1.0

        adv_yuan = _default_adv_yuan(req)
        slices: list[OrderSlice] = []
        remaining = req.total_shares
        session_base = datetime.combine(start.date(), time(9, 30))
        for i, w in enumerate(weights):
            share = int((w / total_w) * req.total_shares)
            share = (share // 100) * 100
            if i == len(weights) - 1:
                share = (remaining // 100) * 100
            if share <= 0:
                continue
            remaining -= share

            s0 = i * minutes_per_slice
            session_start = session_base + timedelta(minutes=s0)
            # 跳过午休
            if session_start.time() >= time(11, 30):
                session_start += timedelta(minutes=90)
            slice_end = session_start + timedelta(minutes=minutes_per_slice)
            bps, yuan = _cost_of_slice(
                share, req.ref_price, adv_yuan, req.volatility, n, self.name,
            )
            slices.append(OrderSlice(
                slice_index=i, shares=int(share),
                start_time=session_start, end_time=slice_end,
                time_offset_minutes=float(s0),
                expected_participation=float(w / total_w),
                expected_cost_bps=bps, expected_cost_yuan=yuan,
                notes=self.name,
            ))

        total_bps = (sum(s.expected_cost_bps * s.shares for s in slices)
                     / max(req.total_shares, 1))
        total_yuan = sum(s.expected_cost_yuan for s in slices)
        return ExecutionPlan(
            request=req, slices=slices,
            total_cost_bps=float(total_bps), total_cost_yuan=float(total_yuan),
            duration_minutes=240.0,
            participation_rate=float(
                req.total_shares * req.ref_price / max(adv_yuan, 1.0)
            ),
            strategy=self.name,
        )


class POVSlicer:
    """Percent of Volume: 占当前市场成交量固定比例, 动态拆.

    POV 严格意义需要盘中实时量, 此处在 OrderRequest.meta 里可传
    `market_volume_forecast`: list[int] 未来每段预期成交量. 没有则
    退化为均匀 TWAP 预估.
    """
    name = "POV"

    def __init__(self, participation_rate: float = 0.10):
        self.rate = participation_rate

    def slice(self, req: OrderRequest) -> ExecutionPlan:
        mv_forecast = req.meta.get("market_volume_forecast")
        adv_yuan = _default_adv_yuan(req)
        if not mv_forecast:
            # 退化 TWAP
            return TWAPSlicer(n_slices=10).slice(req)

        remaining = req.total_shares
        slices: list[OrderSlice] = []
        start = req.start_time or datetime.combine(
            datetime.today().date(), time(9, 40)
        )
        n = len(mv_forecast)
        for i, mv in enumerate(mv_forecast):
            if remaining <= 0:
                break
            target = min(remaining, int(mv * self.rate))
            target = (target // 100) * 100
            if target <= 0:
                continue
            bps, yuan = _cost_of_slice(
                target, req.ref_price, adv_yuan, req.volatility, n, self.name,
            )
            slices.append(OrderSlice(
                slice_index=i, shares=int(target),
                time_offset_minutes=float(i * (240 / n)),
                expected_participation=self.rate,
                expected_cost_bps=bps, expected_cost_yuan=yuan,
                notes=self.name,
            ))
            remaining -= target
        total_bps = (sum(s.expected_cost_bps * s.shares for s in slices)
                     / max(req.total_shares, 1))
        total_yuan = sum(s.expected_cost_yuan for s in slices)
        return ExecutionPlan(
            request=req, slices=slices,
            total_cost_bps=float(total_bps), total_cost_yuan=float(total_yuan),
            duration_minutes=float(len(slices) * (240 / max(n, 1))),
            participation_rate=float(
                req.total_shares * req.ref_price / max(adv_yuan, 1.0)
            ),
            strategy=self.name,
        )


