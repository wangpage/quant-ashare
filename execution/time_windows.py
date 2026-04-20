"""最优交易时段 - A股不是 4 小时都能好交易.

时段分析 (经验+学术, A股高频数据):
    09:15-09:25 集合竞价     → 流动性不稳, 避免
    09:30-09:40 开盘 10 分钟  → 隔夜信息冲击, 高波动, 冲击成本 × 2
    09:40-11:30 上午正常时段  → 最优交易窗口
    11:30-13:00 午休          → 无交易
    13:00-14:30 下午时段      → 正常, 但 14:00 后逐渐走弱
    14:30-14:45 尾盘前期      → 机构调仓, 部分可用
    14:45-15:00 尾盘 15 分钟  → 流动性差 + 对倒, 冲击 × 2
    15:00 收盘

美国学术研究 (Madhavan 2002):
    开盘 / 收盘的价格发现效率最低, 价差最宽.

实战规则:
    - 策略信号在 09:30 产生 → 09:40 后分批入场
    - 信号在 14:30 产生 → 14:45 前必须完成
    - 涨跌停票避免 14:50+ 交易 (经常触发熔断)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from enum import Enum

import pandas as pd


class WindowQuality(str, Enum):
    OPTIMAL = "optimal"            # 最优: 09:40-11:30, 13:00-14:30
    ACCEPTABLE = "acceptable"      # 可接受: 09:30-09:40, 14:30-14:45
    AVOID = "avoid"                # 避免: 开盘 / 尾盘 15 分钟
    FORBIDDEN = "forbidden"        # 禁止: 集合竞价 / 非交易时段


@dataclass
class OptimalTradingWindow:
    """A股时段质量分类器."""

    @staticmethod
    def classify(t: datetime | time) -> WindowQuality:
        tt = t if isinstance(t, time) else t.time()
        # 非交易时段
        if not (time(9, 15) <= tt <= time(15, 0)):
            return WindowQuality.FORBIDDEN
        # 集合竞价
        if time(9, 15) <= tt < time(9, 30):
            return WindowQuality.FORBIDDEN
        # 午休
        if time(11, 30) < tt < time(13, 0):
            return WindowQuality.FORBIDDEN
        # 开盘 10 分钟
        if time(9, 30) <= tt < time(9, 40):
            return WindowQuality.AVOID
        # 尾盘 15 分钟
        if time(14, 45) <= tt <= time(15, 0):
            return WindowQuality.AVOID
        # 上午正常 + 下午前半段
        if time(9, 40) <= tt <= time(11, 30):
            return WindowQuality.OPTIMAL
        if time(13, 0) <= tt < time(14, 30):
            return WindowQuality.OPTIMAL
        # 下午后段
        if time(14, 30) <= tt < time(14, 45):
            return WindowQuality.ACCEPTABLE
        return WindowQuality.FORBIDDEN

    @staticmethod
    def cost_multiplier(t: datetime | time) -> float:
        """不同时段的冲击成本乘数 (相对 OPTIMAL = 1.0)."""
        q = OptimalTradingWindow.classify(t)
        return {
            WindowQuality.OPTIMAL: 1.0,
            WindowQuality.ACCEPTABLE: 1.3,
            WindowQuality.AVOID: 2.0,
            WindowQuality.FORBIDDEN: 99.0,
        }[q]

    @staticmethod
    def next_optimal_window(t: datetime) -> datetime:
        """返回当前时点之后最近的最优交易窗口起点."""
        tt = t.time()
        if tt < time(9, 40):
            return t.replace(hour=9, minute=40, second=0, microsecond=0)
        if tt < time(13, 0):
            if tt <= time(11, 30):
                return t       # 当前就是最优
            return t.replace(hour=13, minute=0, second=0, microsecond=0)
        if tt < time(14, 30):
            return t
        # 已过最优时段, 下一交易日
        next_day = t + pd.Timedelta(days=1)
        while next_day.weekday() >= 5:
            next_day += pd.Timedelta(days=1)
        return next_day.replace(hour=9, minute=40, second=0, microsecond=0)


def is_tradeable_now(t: datetime | None = None) -> bool:
    """当前时点是否可下单 (排除集合竞价和午休)."""
    t = t or datetime.now()
    if t.weekday() >= 5:
        return False
    q = OptimalTradingWindow.classify(t)
    return q != WindowQuality.FORBIDDEN


def avoid_auction_window(t: datetime | None = None,
                          buffer_minutes: int = 5) -> bool:
    """是否在集合竞价附近 buffer 分钟内 (建议回避)."""
    t = t or datetime.now()
    tt = t.time()
    # 开盘前 buffer
    if time(9, 15) <= tt <= time(9, 30 + buffer_minutes // 60 + 0):
        open_buf = (datetime.combine(t.date(), time(9, 30)) +
                    pd.Timedelta(minutes=buffer_minutes)).time()
        if tt <= open_buf:
            return True
    # 收盘前 buffer
    close_buf = (datetime.combine(t.date(), time(15, 0)) -
                 pd.Timedelta(minutes=buffer_minutes)).time()
    if close_buf <= tt <= time(15, 0):
        return True
    return False
