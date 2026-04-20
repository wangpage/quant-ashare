"""PreTradeGate - 下单前强制过闸中间件.

## 设计目标

回测与实盘必须走**完全相同**的风控代码路径, 避免 "回测通过、实盘爆仓"
这种 A股最常见的灾难. 本中间件是所有下单的**唯一入口**:

    回测           实盘
      ↓             ↓
    PreTradeGate ───┘
      ↓
    Broker / Simulator

## 功能

1. 原子 check: 一次性跑完 AShareRiskManager 的 buy/sell 校验
2. 记录拒单原因链 (audit trail), 可导出为 CSV
3. 支持"硬规则"(必须通过) 和 "软规则"(记录 warning 但放行)
4. 统计: 通过率, 被拒的 top-N 原因, 每日调用曲线

任何新下单代码都必须调 ``PreTradeGate.check(order, ctx)``, 不能绕过.
研究 pipeline / daily trading / live trading 共用一份实现.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Literal

from utils.logger import logger

from .a_share_rules import (
    AShareRiskManager, Portfolio, Position, RiskCheckResult,
)


OrderSide = Literal["buy", "sell"]


@dataclass
class OrderIntent:
    """上游传来的下单意图, 不是最终订单.

    gate.check() 会根据风控规则返回 adjusted_shares 和 allow.
    """
    code: str
    side: OrderSide
    shares: int
    price: float
    prev_close: float
    industry: str = ""
    suspended: bool = False
    conviction: float = 0.5
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class GateDecision:
    """gate 返回的决策."""
    allow: bool
    adjusted_shares: int
    reason: str
    severity: Literal["PASS", "WARN", "HARD_REJECT"] = "PASS"
    checks_passed: list[str] = field(default_factory=list)


@dataclass
class GateStats:
    """累计统计, 用于回测后复盘或实盘监控看板."""
    n_orders: int = 0
    n_passed: int = 0
    n_warned: int = 0
    n_rejected: int = 0
    reject_reasons: dict[str, int] = field(default_factory=dict)
    warnings: dict[str, int] = field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        return self.n_passed / max(self.n_orders, 1)

    def top_reject_reasons(self, k: int = 5) -> list[tuple[str, int]]:
        return sorted(
            self.reject_reasons.items(), key=lambda x: -x[1]
        )[:k]

    def to_dict(self) -> dict:
        return {
            "n_orders": self.n_orders,
            "n_passed": self.n_passed,
            "n_warned": self.n_warned,
            "n_rejected": self.n_rejected,
            "pass_rate": self.pass_rate,
            "top_reject": self.top_reject_reasons(),
        }


class PreTradeGate:
    """所有下单的风控闸门. 回测与实盘共用一份实现.

    使用 (回测):
        gate = PreTradeGate(risk_manager=AShareRiskManager())
        for order_intent in intents:
            decision = gate.check(order_intent, portfolio, today)
            if not decision.allow:
                continue   # 记录拒单, 不下单
            submit(order_intent.code, decision.adjusted_shares)
        print(gate.stats.to_dict())

    使用 (实盘):
        # 启动时加载实盘 portfolio 状态
        gate = PreTradeGate(risk_manager=AShareRiskManager())
        # 每笔订单:
        decision = gate.check(intent, portfolio, datetime.now().date())
    """

    def __init__(
        self,
        risk_manager: AShareRiskManager | None = None,
        soft_checks: list[str] | None = None,
    ):
        self.rm = risk_manager or AShareRiskManager()
        # 软规则: 违反只发 warning 不拒单 (如 "行业集中度轻微超标")
        self.soft_checks = set(soft_checks or [])
        self.stats = GateStats()
        self.audit_trail: list[dict] = []

    def check(
        self,
        intent: OrderIntent,
        portfolio: Portfolio,
        today: date | datetime | None = None,
    ) -> GateDecision:
        """核心入口: 单笔订单的硬过闸."""
        today_date = today.date() if isinstance(today, datetime) else today
        today_date = today_date or date.today()
        self.stats.n_orders += 1

        if intent.side == "buy":
            res = self.rm.check_buy(
                code=intent.code, industry=intent.industry,
                price=intent.price, prev_close=intent.prev_close,
                shares=intent.shares, portfolio=portfolio,
                suspended=intent.suspended,
                signal_conviction=intent.conviction,
            )
        elif intent.side == "sell":
            res = self.rm.check_sell(
                code=intent.code, portfolio=portfolio, today=today_date,
                price=intent.price, prev_close=intent.prev_close,
                suspended=intent.suspended,
            )
        else:
            return self._record(
                intent,
                GateDecision(
                    allow=False, adjusted_shares=0,
                    reason=f"未知 side={intent.side}",
                    severity="HARD_REJECT",
                ),
            )

        adjusted = (
            res.adjusted_shares
            if res.adjusted_shares is not None
            else intent.shares
        )

        if not res.ok:
            # 软规则兜底: 若本次拒单原因在 soft_checks 里, 降级为 warning
            if any(sk in res.reason for sk in self.soft_checks):
                decision = GateDecision(
                    allow=True, adjusted_shares=adjusted,
                    reason=f"SOFT: {res.reason}", severity="WARN",
                    checks_passed=["soft_downgrade"],
                )
            else:
                decision = GateDecision(
                    allow=False, adjusted_shares=0,
                    reason=res.reason, severity="HARD_REJECT",
                )
        else:
            decision = GateDecision(
                allow=True, adjusted_shares=int(adjusted),
                reason=res.reason, severity="PASS",
                checks_passed=["all"],
            )
        return self._record(intent, decision)

    def _record(
        self, intent: OrderIntent, decision: GateDecision,
    ) -> GateDecision:
        if decision.severity == "PASS":
            self.stats.n_passed += 1
        elif decision.severity == "WARN":
            self.stats.n_warned += 1
            self.stats.warnings[decision.reason] = (
                self.stats.warnings.get(decision.reason, 0) + 1
            )
        else:
            self.stats.n_rejected += 1
            self.stats.reject_reasons[decision.reason] = (
                self.stats.reject_reasons.get(decision.reason, 0) + 1
            )

        self.audit_trail.append({
            "code": intent.code, "side": intent.side,
            "shares": intent.shares,
            "adjusted": decision.adjusted_shares,
            "allow": decision.allow,
            "severity": decision.severity,
            "reason": decision.reason,
            "ts": datetime.now().isoformat(timespec="seconds"),
        })
        # 控制审计日志内存; 回测大量订单时需保留必要信息
        if len(self.audit_trail) > 20000:
            self.audit_trail = self.audit_trail[-10000:]

        if not decision.allow:
            logger.debug(
                f"gate 拒 {intent.code} {intent.side} {intent.shares}: "
                f"{decision.reason}"
            )
        return decision

    def reset_stats(self) -> None:
        self.stats = GateStats()
        self.audit_trail.clear()


def build_default_gate(config: dict | None = None) -> PreTradeGate:
    """快捷构造: 读 CONFIG 风控段, 返回标准闸门."""
    return PreTradeGate(risk_manager=AShareRiskManager(config=config))
