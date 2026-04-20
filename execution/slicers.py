"""订单拆分算法: TWAP / VWAP / POV.

TWAP (Time-Weighted Average Price):
    均匀分布在时间窗口内. 简单, 不看成交量.
    适合: 流动性差的小盘, 或对冲击成本不敏感的慢策略.

VWAP (Volume-Weighted Average Price):
    按历史成交量分布拆单, 尽量跟上市场节奏, 不惊扰盘口.
    适合: 大单, 机构标配.

POV (Percent of Volume):
    始终保持成交占当前市场成交量的 X%. 动态调整.
    适合: 没有预测未来成交量的场景, 跟随实际流动性.

Implementation Shortfall (Almgren-Chriss):
    最小化 "期望成本 + λ × 风险方差", 理论最优.
    需要预测未来波动, 实现复杂.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class OrderSlice:
    start_time: datetime
    end_time: datetime
    target_shares: int
    expected_participation: float   # 占该时段成交量的比例
    notes: str = ""


class TWAPSlicer:
    """时间均匀拆单."""

    def __init__(self, avoid_opening_min: int = 10, avoid_closing_min: int = 15):
        self.avoid_opening_min = avoid_opening_min
        self.avoid_closing_min = avoid_closing_min

    def slice(
        self, total_shares: int,
        start: datetime, end: datetime,
        n_slices: int = 10,
    ) -> list[OrderSlice]:
        # 计算 effective 开始/结束
        eff_start = max(
            start,
            datetime.combine(start.date(), time(9, 30)) +
            timedelta(minutes=self.avoid_opening_min),
        )
        close_t = datetime.combine(end.date(), time(15, 0)) - \
                  timedelta(minutes=self.avoid_closing_min)
        eff_end = min(end, close_t)

        if eff_end <= eff_start:
            return []

        # 扣除午休
        total_minutes = (eff_end - eff_start).total_seconds() / 60
        lunch_start = datetime.combine(eff_start.date(), time(11, 30))
        lunch_end = datetime.combine(eff_start.date(), time(13, 0))
        if eff_start < lunch_start < eff_end:
            lunch = (min(lunch_end, eff_end) -
                     max(lunch_start, eff_start)).total_seconds() / 60
            total_minutes -= max(0, lunch)

        per_slice_min = total_minutes / n_slices
        per_slice_shares = total_shares // n_slices
        remainder = total_shares - per_slice_shares * n_slices

        slices = []
        cur = eff_start
        for i in range(n_slices):
            next_t = cur + timedelta(minutes=per_slice_min)
            # 跳过午休
            if cur < lunch_start < next_t:
                next_t = next_t + timedelta(minutes=90)
            shares = per_slice_shares + (1 if i < remainder else 0)
            # 整手
            shares = (shares // 100) * 100
            if shares > 0:
                slices.append(OrderSlice(
                    start_time=cur, end_time=next_t,
                    target_shares=shares,
                    expected_participation=1.0 / n_slices,
                    notes="TWAP",
                ))
            cur = next_t
        return slices


class VWAPSlicer:
    """按历史成交量分布拆单. 需要过去 N 日分钟成交量曲线."""

    def __init__(self, volume_curve: pd.Series | None = None):
        """volume_curve: pd.Series index=minute_of_day, values=avg volume."""
        self.volume_curve = volume_curve or self._default_a_share_curve()

    @staticmethod
    def _default_a_share_curve() -> pd.Series:
        """A股典型日内成交量曲线 (U 型).

        开盘 10 分钟占日成交 ~18%
        11:00-11:30 占 ~12%
        下午开盘占 ~8%
        尾盘 30 分钟占 ~15%
        """
        minutes = list(range(240))   # 4 小时 = 240 min
        curve = np.ones(240)
        # 上午 (分钟 0-120)
        curve[0:15] = 3.5              # 开盘 15 分钟高峰
        curve[15:30] = 1.5
        curve[30:100] = 0.8
        curve[100:120] = 1.1           # 上午收盘前
        # 下午 (分钟 120-240)
        curve[120:140] = 0.9            # 下午开盘温和
        curve[140:210] = 0.75
        curve[210:240] = 2.0            # 尾盘
        curve = curve / curve.sum()
        idx = pd.RangeIndex(0, 240, name="min_of_session")
        return pd.Series(curve, index=idx)

    def slice(
        self, total_shares: int,
        start: datetime, end: datetime,
        n_slices: int = 20,
    ) -> list[OrderSlice]:
        # 把 start-end 映射到 minute_of_session
        minutes_per_slice = 240 // n_slices
        slices = []
        cur = datetime.combine(start.date(), time(9, 30))
        total_weight = 0.0
        weights = []
        for i in range(n_slices):
            start_min = i * minutes_per_slice
            end_min = min((i + 1) * minutes_per_slice, 240)
            w = self.volume_curve.iloc[start_min:end_min].sum()
            weights.append(w)
            total_weight += w

        remaining = total_shares
        for i, w in enumerate(weights):
            share = int((w / total_weight) * total_shares)
            share = (share // 100) * 100
            if i == len(weights) - 1:
                share = (remaining // 100) * 100
            remaining -= share
            if share <= 0:
                continue

            # 映射真实时间 (跳过午休)
            start_min = i * minutes_per_slice
            session_start = cur + timedelta(minutes=start_min)
            if session_start.time() >= time(11, 30):
                session_start += timedelta(minutes=90)
            slice_end = session_start + timedelta(minutes=minutes_per_slice)

            slices.append(OrderSlice(
                start_time=session_start, end_time=slice_end,
                target_shares=share,
                expected_participation=float(w / total_weight),
                notes="VWAP",
            ))
        return slices


class POVSlicer:
    """Percent of Volume: 始终占当前市场成交量 X%.

    用法 (实盘):
        while still_have_shares_to_fill:
            market_vol = current_minute_market_volume()
            my_vol = market_vol * participation_rate
            submit(my_vol)
    """

    def __init__(self, participation_rate: float = 0.10):
        self.rate = participation_rate

    def estimate_duration_minutes(
        self, total_shares: int, expected_market_volume_per_min: int,
    ) -> float:
        """根据参与率预估完成时间."""
        my_rate = expected_market_volume_per_min * self.rate
        if my_rate <= 0:
            return float("inf")
        return total_shares / my_rate

    def next_slice_size(self, market_vol_last_minute: int) -> int:
        """基于上一分钟市场成交量, 给出本分钟下单量."""
        raw = int(market_vol_last_minute * self.rate)
        return (raw // 100) * 100
