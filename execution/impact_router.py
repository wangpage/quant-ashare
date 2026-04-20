"""冲击感知路由 - 结合 Level2 盘口与 Almgren-Chriss 动态分单.

## 冲击成本的正确理解

sqrt 律对参与率 (participation rate) 敏感. 分片执行通过"摊薄参与率"
降低总冲击:

    一次性下单: cost_bps = k × σ × √(Q / ADV) × 1e4
    分成 N 片匀速执行: 每片 cost_bps / √N, 累计 N × cost_bps/√N = √N × cost_bps/N
                     = cost_bps / √N  (比一次性低 √N 倍)

即: **分片越多, 总冲击越低**, 但时间风险 (机会成本) 增大.
这就是 Almgren-Chriss 的经典 cost-risk trade-off.

## 本实现的关键约定

1. `adv_yuan` 推荐传 **过去 20 日平均成交额 (ADV)**, 比前一日稳定
2. 冲击以 **总订单** 为口径计算一次, 再按股数均摊给每片
3. 每片显示的 `expected_cost_bps` 是摊到该片的 bps (总金额 × bps / 1e4),
   不是把 slice 当独立订单重新跑 sqrt. 旧实现的累加方式在数学上等价于
   用 √N 倍, 但容易被误读, 现改为"先总量估冲击, 再均摊".
4. 参与率 > 30% 会触发"冲击增强系数" (大订单的非线性溢价).
"""
from __future__ import annotations

from dataclasses import dataclass

from market_microstructure.impact import (
    almgren_chriss_impact, square_root_impact,
)
from utils.logger import logger

from .base import (
    ExecutionPlan, OrderRequest, OrderSlice, Side,
)


# 盘中有效交易分钟数 (扣除集合竞价+午休, ~240 min)
_TRADING_MINUTES = 240.0

# 参与率警戒线: > 此值的订单加非线性溢价 (超大单增量冲击)
_PARTICIPATION_WARN = 0.30
_PARTICIPATION_DANGER = 0.50


@dataclass
class RoutingPlan:
    """向后兼容的 dict 字段版. 新代码请用 execution.base.ExecutionPlan."""
    total_shares: int
    slices: list[dict]
    expected_total_cost_bps: float   # 基于总金额的冲击 + 手续费预估 (bps)
    expected_total_cost_yuan: float  # 冲击成本 (元), 不含印花税/佣金
    duration_minutes: float
    participation_rate: float         # 相对 ADV 的总参与率
    notes: str = ""


class ImpactAwareRouter:
    """根据盘口深度和 Almgren-Chriss 模型动态决定拆单策略.

    同时实现 ``execution.base.Slicer`` 协议: ``router.slice(req) -> ExecutionPlan``.
    """
    name = "ImpactAware"

    def __init__(
        self,
        target_participation: float = 0.05,    # 每片占 ADV 的目标比例
        max_ticks_per_slice: float = 3.0,       # 单片最多穿多少 tick
        urgency: float = 0.5,                   # 0=慢但便宜, 1=快但贵
    ):
        self.target_participation = target_participation
        self.max_ticks_per_slice = max_ticks_per_slice
        self.urgency = urgency

    # ---------- 统一协议入口 ----------
    def slice(self, request: OrderRequest) -> ExecutionPlan:
        """Slicer 协议实现. 内部委托给 plan_order."""
        self.urgency = request.urgency
        plan = self.plan_order(
            total_shares=request.total_shares,
            price=request.ref_price,
            adv_yuan=request.adv_yuan,
            volatility=request.volatility,
            book=request.book,
            daily_volume=request.daily_volume,
        )
        slices = [
            OrderSlice(
                slice_index=s["slice_index"], shares=int(s["shares"]),
                time_offset_minutes=float(s["time_offset_minutes"]),
                expected_participation=float(s.get("participation_est", 0)),
                expected_cost_bps=float(s["expected_cost_bps"]),
                expected_cost_yuan=float(s["expected_cost_yuan"]),
                warnings=[s["warning"]] if "warning" in s else [],
                notes=self.name,
            )
            for s in plan.slices
        ]
        return ExecutionPlan(
            request=request, slices=slices,
            total_cost_bps=plan.expected_total_cost_bps,
            total_cost_yuan=plan.expected_total_cost_yuan,
            duration_minutes=plan.duration_minutes,
            participation_rate=plan.participation_rate,
            strategy=self.name, notes=plan.notes,
        )

    def plan_order(
        self,
        total_shares: int,
        price: float,
        adv_yuan: float | None = None,       # 过去 20 日平均成交额 (推荐)
        volatility: float = 0.02,            # 日波动率
        book: dict | None = None,
        daily_volume: int | None = None,      # 向后兼容: 股数口径
    ) -> RoutingPlan:
        """生成订单切片计划.

        Args:
            total_shares: 总订单股数
            price: 当前参考价
            adv_yuan: 20 日 ADV (元), **推荐口径**. 比前一日成交量稳定.
            volatility: 日波动率
            book: 盘口 {'bid': [(p, v), ...], 'ask': [...]}
            daily_volume: (股数) 若未传 adv_yuan, 用 daily_volume × price 近似

        Returns:
            RoutingPlan: 切片计划, 每片含 (股数, 时间偏移, 预期摊销成本).
        """
        # 1. 计算 ADV (元)
        if adv_yuan is None:
            if daily_volume is None:
                raise ValueError("必须传 adv_yuan (推荐) 或 daily_volume")
            adv_yuan = max(daily_volume * price, 1.0)
            logger.debug(
                "未传 adv_yuan, 退化为 daily_volume × price, "
                "建议上游改用 20 日 ADV"
            )
        adv_yuan = max(adv_yuan, 1.0)

        trade_amount = total_shares * price
        total_participation = trade_amount / adv_yuan

        # 2. Almgren-Chriss 估算一次性下单的总冲击 bps
        ac = almgren_chriss_impact(
            trade_amount, adv_yuan, volatility, is_buy=True,
        )
        total_shot_bps = ac["total_bps"]

        # 3. 根据一次性冲击 + 急迫性决定切片数
        if total_shot_bps < 20:
            n_slices = 3
        elif total_shot_bps < 50:
            n_slices = 6
        elif total_shot_bps < 100:
            n_slices = 10
        else:
            n_slices = max(15, int(total_shot_bps / 10))

        if self.urgency > 0.7:
            n_slices = max(1, int(n_slices * 0.5))
        elif self.urgency < 0.3:
            n_slices = int(n_slices * 1.5)

        # 目标参与率约束: 若单片金额 / ADV > target, 强制拆更多
        if self.target_participation > 0:
            min_slices_for_target = int(
                total_participation / self.target_participation + 0.999
            )
            n_slices = max(n_slices, min_slices_for_target)
        n_slices = max(1, n_slices)

        # 4. 分配整手, 剩余塞到末尾
        slice_shares_base = (total_shares // n_slices // 100) * 100
        remainder = total_shares - slice_shares_base * n_slices
        remainder_lots = remainder // 100

        slice_shares_list: list[int] = []
        for i in range(n_slices):
            s = slice_shares_base + (100 if i < remainder_lots else 0)
            if s > 0:
                slice_shares_list.append(s)

        # 5. 关键公式: 分片后"总冲击 bps" = sqrt(1/N) × 一次性 bps.
        #    以总金额为基数计算, 再按股数摊到每片.
        sliced_total_bps = self._sliced_total_cost_bps(
            trade_amount, adv_yuan, volatility,
            n_slices=len(slice_shares_list),
            total_participation=total_participation,
        )
        sliced_total_yuan = sliced_total_bps / 1e4 * trade_amount

        slices = []
        per_slice_minutes = _TRADING_MINUTES / max(n_slices, 1)
        for i, s in enumerate(slice_shares_list):
            weight = s / total_shares if total_shares > 0 else 0.0
            slice_yuan = sliced_total_yuan * weight
            slice_bps = slice_yuan / max(s * price, 1.0) * 1e4
            slices.append({
                "slice_index": i,
                "shares": int(s),
                "expected_cost_bps": float(slice_bps),
                "expected_cost_yuan": float(slice_yuan),
                "time_offset_minutes": float(i * per_slice_minutes),
                "participation_est": float(total_participation / max(n_slices, 1)),
            })

        # 6. 盘口深度优化
        if book:
            level1_ask_qty = (book.get("ask") or [(0, 0)])[0][1]
            for s in slices:
                if s["shares"] > level1_ask_qty * self.max_ticks_per_slice:
                    s["warning"] = "超过盘口 N 档, 可能穿价"

        notes = (
            f"AC={total_shot_bps:.1f}bps, sliced={sliced_total_bps:.1f}bps, "
            f"pr={total_participation:.2%}, slices={len(slices)}, "
            f"urgency={self.urgency}"
        )
        if total_participation > _PARTICIPATION_DANGER:
            notes += " ⚠️ 参与率极高"
        elif total_participation > _PARTICIPATION_WARN:
            notes += " ⚠️ 参与率警戒"

        return RoutingPlan(
            total_shares=total_shares,
            slices=slices,
            expected_total_cost_bps=float(sliced_total_bps),
            expected_total_cost_yuan=float(sliced_total_yuan),
            duration_minutes=float(len(slice_shares_list) * per_slice_minutes),
            participation_rate=float(total_participation),
            notes=notes,
        )

    @staticmethod
    def _sliced_total_cost_bps(
        trade_amount: float, adv_yuan: float, volatility: float,
        n_slices: int, total_participation: float,
    ) -> float:
        """分片执行的总冲击 bps.

        理论: 匀速分 N 片 → 每片成本 = 一次性成本 / √N (临时冲击),
              累加后总成本 bps ≈ 一次性 bps / √N.

        极端溢价: 若总参与率 > 30%, 加 Almgren-Chriss 中永久冲击部分
                 (不随分片减少), 在此通过 ratio × 一次性 bps 简化:

                 total ≈ one_shot_bps × (η/√N + γ)
                 其中 η 是临时冲击占比 (默认 0.65),
                      γ 是永久冲击占比 (默认 0.35, 参考 AC 2000)
        """
        one_shot_bps = square_root_impact(trade_amount, adv_yuan, volatility)
        eta = 0.65   # temporary
        gamma = 0.35  # permanent
        sliced_bps = one_shot_bps * (eta / max(n_slices, 1) ** 0.5 + gamma)

        # 超大单溢价 (非线性)
        if total_participation > _PARTICIPATION_WARN:
            overrun = (total_participation - _PARTICIPATION_WARN) / _PARTICIPATION_WARN
            sliced_bps *= (1.0 + 0.5 * overrun)
        return float(sliced_bps)


def split_order_by_participation(
    total_shares: int,
    market_volume_forecast: list[int],
    target_participation: float = 0.10,
) -> list[int]:
    """按市场预期成交量拆单, 每段占该时段 X% (与 ImpactAwareRouter 解耦的轻量版)."""
    slices = []
    remaining = total_shares
    for mv in market_volume_forecast:
        my_share = min(remaining, int(mv * target_participation))
        my_share = (my_share // 100) * 100
        slices.append(my_share)
        remaining -= my_share
        if remaining <= 0:
            break
    if remaining > 0 and slices:
        slices[-1] += (remaining // 100) * 100
    return slices


def estimate_adv_yuan(
    volumes: list[float] | None = None,
    prices: list[float] | None = None,
    amounts: list[float] | None = None,
    window: int = 20,
) -> float:
    """计算 20 日 ADV (元) 的辅助函数.

    优先使用 amounts (元); 否则用 volumes × prices 对齐相乘.

    Args:
        volumes: 每日成交量 (股)
        prices: 每日收盘价 (与 volumes 等长)
        amounts: 每日成交额 (元), 若给出则直接取均值
        window: 窗口天数, 默认 20
    """
    if amounts:
        tail = amounts[-window:]
        return float(sum(tail) / max(len(tail), 1))
    if volumes and prices:
        tail_v = volumes[-window:]
        tail_p = prices[-window:]
        daily = [v * p for v, p in zip(tail_v, tail_p)]
        return float(sum(daily) / max(len(daily), 1))
    raise ValueError("请传 amounts, 或同时传 volumes 和 prices")
