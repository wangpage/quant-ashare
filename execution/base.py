"""执行层基础协议与共享数据类 - 统一接口.

旧版痛点:
    slicers.py 有 OrderSlice (dataclass), impact_router.py 用 dict, 回测
    simulator.py 又用自己的 ExecutionResult. 三者互不兼容, 上游调用方要
    做大量 glue code.

统一约定:
    1. `OrderRequest` - 下单请求 (输入)
    2. `OrderSlice`   - 单个切片 (输出, 含时段/股数/预期成本)
    3. `ExecutionPlan` - 切片列表 + 汇总成本 (输出)
    4. `Slicer`       - 协议, 所有分单算法都实现 .slice(request)
    5. `ExecutionEngine` - 统一入口, 按策略选 Slicer → 叠加路由器 → 执行

任何分单算法 (TWAP/VWAP/POV/ImpactAware) 只需实现 Slicer.slice(req) → ExecutionPlan.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class OrderSlice:
    """统一切片数据类, 所有 Slicer 产出这个类型."""
    slice_index: int
    shares: int
    start_time: datetime | None = None
    end_time: datetime | None = None
    time_offset_minutes: float = 0.0
    expected_participation: float = 0.0
    expected_cost_bps: float = 0.0
    expected_cost_yuan: float = 0.0
    notes: str = ""
    warnings: list[str] = field(default_factory=list)


@dataclass
class OrderRequest:
    """统一下单请求."""
    code: str
    side: Side
    total_shares: int
    ref_price: float
    adv_yuan: float | None = None        # 20 日平均成交额, 推荐
    daily_volume: int | None = None       # 若无 ADV 用当日 volume
    volatility: float = 0.02
    start_time: datetime | None = None
    end_time: datetime | None = None
    book: dict | None = None
    urgency: float = 0.5
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionPlan:
    """切片计划 + 聚合成本."""
    request: OrderRequest
    slices: list[OrderSlice]
    total_cost_bps: float
    total_cost_yuan: float
    duration_minutes: float
    participation_rate: float = 0.0
    strategy: str = ""
    notes: str = ""

    @property
    def total_shares(self) -> int:
        return sum(s.shares for s in self.slices)


@runtime_checkable
class Slicer(Protocol):
    """所有分单算法的协议."""
    name: str

    def slice(self, request: OrderRequest) -> ExecutionPlan:
        ...


# ---------- 时段共用工具 ----------
_MARKET_OPEN = time(9, 30)
_LUNCH_START = time(11, 30)
_LUNCH_END = time(13, 0)
_MARKET_CLOSE = time(15, 0)


def clip_to_trading_session(
    start: datetime, end: datetime,
    avoid_opening_min: int = 10, avoid_closing_min: int = 15,
) -> tuple[datetime, datetime]:
    """把任意 [start, end] 区间裁剪到 A股有效交易时段.

    - 跳过集合竞价 (≤09:30)
    - 避让开盘 avoid_opening_min 分钟
    - 避让尾盘 avoid_closing_min 分钟
    - 午休 (11:30-13:00) 不扣除, 由调用方按需跳过.
    """
    day = start.date()
    market_open_day = datetime.combine(day, _MARKET_OPEN)
    market_close_day = datetime.combine(day, _MARKET_CLOSE)
    eff_start = max(start, market_open_day + timedelta(minutes=avoid_opening_min))
    eff_end = min(end, market_close_day - timedelta(minutes=avoid_closing_min))
    return eff_start, eff_end


def effective_trading_minutes(
    start: datetime, end: datetime,
) -> float:
    """计算 [start, end] 内剔除午休的实际交易分钟数."""
    day = start.date()
    lunch_s = datetime.combine(day, _LUNCH_START)
    lunch_e = datetime.combine(day, _LUNCH_END)
    total = (end - start).total_seconds() / 60
    if end <= lunch_s or start >= lunch_e:
        return max(0.0, total)
    overlap = (min(end, lunch_e) - max(start, lunch_s)).total_seconds() / 60
    return max(0.0, total - overlap)
