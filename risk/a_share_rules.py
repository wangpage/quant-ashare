"""A股特化风控: 涨跌停、T+1、停牌、仓位、止损、熔断."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Literal

import numpy as np

from utils.config import CONFIG
from utils.logger import logger


@dataclass
class Position:
    code: str
    shares: int            # 持股数
    avg_cost: float        # 成本价
    current_price: float
    open_date: date        # 建仓日期 (T+1 判断)
    industry: str = ""

    @property
    def market_value(self) -> float:
        return self.shares * self.current_price

    @property
    def pnl_pct(self) -> float:
        return self.current_price / self.avg_cost - 1

    def can_sell(self, today: date) -> bool:
        """A股 T+1 规则: 买入当日不可卖."""
        return today > self.open_date


@dataclass
class Portfolio:
    cash: float
    initial_capital: float
    positions: dict[str, Position] = field(default_factory=dict)
    high_water_mark: float = 0.0
    daily_start_value: float = 0.0

    @property
    def total_value(self) -> float:
        return self.cash + sum(p.market_value for p in self.positions.values())

    @property
    def drawdown(self) -> float:
        if self.high_water_mark <= 0:
            return 0.0
        return (self.total_value - self.high_water_mark) / self.high_water_mark

    @property
    def daily_pnl_pct(self) -> float:
        if self.daily_start_value <= 0:
            return 0.0
        return (self.total_value - self.daily_start_value) / self.daily_start_value

    def position_pct(self, code: str) -> float:
        if code not in self.positions:
            return 0.0
        return self.positions[code].market_value / max(self.total_value, 1e-9)

    def industry_pct(self, industry: str) -> float:
        tv = self.total_value
        if tv <= 0:
            return 0.0
        mv = sum(p.market_value for p in self.positions.values() if p.industry == industry)
        return mv / tv


@dataclass
class RiskCheckResult:
    ok: bool
    reason: str = ""
    adjusted_shares: int | None = None


class AShareRiskManager:
    """执行所有 A股短线硬约束."""

    def __init__(self, config: dict | None = None):
        c = (config or CONFIG)["risk"]
        self.max_pos_stock    = c["max_position_per_stock"]
        self.max_pos_industry = c["max_position_per_industry"]
        self.stop_loss        = c["stop_loss"]
        self.stop_profit      = c["stop_profit"]
        self.daily_loss_limit = c["daily_loss_limit"]
        self.max_dd_limit     = c["max_drawdown_limit"]
        self.kelly_frac       = c["kelly_fraction"]
        self.min_cash_ratio   = c["min_cash_ratio"]
        self.halted_circuit = False   # 熔断状态

    # ---------- 开仓前检查 ----------
    def check_buy(
        self,
        code: str,
        industry: str,
        price: float,
        prev_close: float,
        shares: int,
        portfolio: Portfolio,
        suspended: bool = False,
        signal_conviction: float = 0.5,
    ) -> RiskCheckResult:

        # 1. 组合熔断
        if self.halted_circuit:
            return RiskCheckResult(False, "组合处于熔断冷静期")

        # 2. 停牌
        if suspended:
            return RiskCheckResult(False, f"{code} 停牌")

        # 3. 涨停板: A股涨幅>=9.5% 视为涨停, 无法买入
        if price >= prev_close * 1.095:
            return RiskCheckResult(False, f"{code} 已涨停, 禁止追涨")

        # 4. 当日亏损熔断
        if portfolio.daily_pnl_pct < self.daily_loss_limit:
            self.halted_circuit = True
            return RiskCheckResult(False, f"单日亏损超限 {portfolio.daily_pnl_pct:.2%}")

        # 5. 最大回撤熔断
        if portfolio.drawdown < self.max_dd_limit:
            self.halted_circuit = True
            return RiskCheckResult(False, f"最大回撤超限 {portfolio.drawdown:.2%}")

        # 6. 最小现金比例
        cash_ratio = portfolio.cash / max(portfolio.total_value, 1e-9)
        amount = price * shares
        if (portfolio.cash - amount) / max(portfolio.total_value, 1e-9) < self.min_cash_ratio:
            max_amount = portfolio.cash - portfolio.total_value * self.min_cash_ratio
            if max_amount <= 0:
                return RiskCheckResult(False, "现金不足以保留最低现金比例")
            shares = int(max_amount // (price * 100)) * 100  # 整手
            if shares <= 0:
                return RiskCheckResult(False, "调整后手数为0")

        # 7. 单票仓位上限
        curr_pos = portfolio.position_pct(code) if code in portfolio.positions else 0
        add_pct = (price * shares) / max(portfolio.total_value, 1e-9)
        if curr_pos + add_pct > self.max_pos_stock:
            allow = self.max_pos_stock - curr_pos
            if allow <= 0:
                return RiskCheckResult(False, "单票仓位已达上限")
            shares = int(portfolio.total_value * allow / price / 100) * 100
            if shares <= 0:
                return RiskCheckResult(False, "单票仓位调整后手数为0")

        # 8. 行业集中度
        ind_pct = portfolio.industry_pct(industry)
        if ind_pct + add_pct > self.max_pos_industry:
            return RiskCheckResult(False, f"行业{industry}集中度超限 {ind_pct:.2%}")

        # 9. 交易单元: A股 1手=100股
        shares = (shares // 100) * 100
        if shares < 100:
            return RiskCheckResult(False, "不足1手")

        return RiskCheckResult(True, "OK", adjusted_shares=shares)

    # ---------- 平仓前检查 ----------
    def check_sell(
        self, code: str, portfolio: Portfolio, today: date,
        price: float, prev_close: float, suspended: bool = False,
    ) -> RiskCheckResult:
        if suspended:
            return RiskCheckResult(False, f"{code} 停牌, 无法卖出")
        if code not in portfolio.positions:
            return RiskCheckResult(False, f"{code} 未持仓")
        pos = portfolio.positions[code]
        if not pos.can_sell(today):
            return RiskCheckResult(False, f"{code} T+1限制, 今日买入不可卖")
        # 跌停不能卖出
        if price <= prev_close * 0.905:
            return RiskCheckResult(False, f"{code} 跌停板, 无法卖出")
        return RiskCheckResult(True, "OK")

    # ---------- 动态止损/止盈 ----------
    def should_exit(self, pos: Position) -> tuple[bool, str]:
        if pos.pnl_pct <= self.stop_loss:
            return True, f"止损 {pos.pnl_pct:.2%}"
        if pos.pnl_pct >= self.stop_profit:
            return True, f"止盈 {pos.pnl_pct:.2%}"
        return False, ""

    # ---------- 凯利公式仓位 ----------
    def kelly_size(
        self, win_rate: float, win_loss_ratio: float,
        portfolio_value: float, conviction: float = 1.0,
    ) -> float:
        """返回建议仓位金额."""
        if win_rate <= 0 or win_loss_ratio <= 0:
            return 0.0
        kelly = win_rate - (1 - win_rate) / win_loss_ratio
        kelly = max(0.0, min(kelly, 1.0))
        kelly_scaled = kelly * self.kelly_frac * conviction
        return portfolio_value * min(kelly_scaled, self.max_pos_stock)

    # ---------- 每日重置 ----------
    def new_day(self, portfolio: Portfolio, current_value: float):
        """每个交易日开盘调用."""
        portfolio.daily_start_value = current_value
        portfolio.high_water_mark = max(portfolio.high_water_mark, current_value)
        # 回撤如果回到 95% 以上, 解除熔断
        if self.halted_circuit and portfolio.drawdown > self.max_dd_limit * 0.5:
            logger.info("组合回撤缓解, 解除熔断")
            self.halted_circuit = False
