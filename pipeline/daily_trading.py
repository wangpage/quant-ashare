"""实盘每日决策 pipeline.

每天盘前运行一次, 产出:
    - 今日股票池
    - Regime 判断
    - Agent 团队决策
    - 带冲击感知的下单计划
    - 交易完成后的反思存入 memory

不依赖任何 API Key 时用 mock backend 可演示全链路.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import pandas as pd

from utils.logger import logger


@dataclass
class DailyDecision:
    date: str
    regime: str
    position_mult: float
    candidates: list[dict] = field(default_factory=list)
    orders: list[dict] = field(default_factory=list)
    reflections_saved: int = 0
    notes: list[str] = field(default_factory=list)


class DailyTradingPipeline:
    """每日决策一键式流水线."""

    def __init__(
        self,
        top_k: int = 10,
        use_llm: bool = False,
        llm_backend: str = "mock",
        target_participation: float = 0.05,
    ):
        self.top_k = top_k
        self.use_llm = use_llm
        self.llm_backend = llm_backend
        self.target_participation = target_participation

    def run(
        self,
        index_df: pd.DataFrame,
        stocks_daily: pd.DataFrame,
        candidates_scored: dict[str, float],
        total_capital: float = 1_000_000,
    ) -> DailyDecision:
        """运行每日 pipeline.

        Args:
            index_df: 指数日线, 至少 200 日
            stocks_daily: 今日全市场个股快照 (含 pct_chg)
            candidates_scored: {code: model_score}, 量化模型输出
            total_capital: 组合总资金
        """
        today = date.today().isoformat()
        decision = DailyDecision(date=today, regime="unknown", position_mult=0.5)

        # 1) Regime 检测
        decision = self._stage_regime(index_df, stocks_daily, decision)

        # 2) 候选股票筛选 (top-K by score)
        decision = self._stage_screen(candidates_scored, decision)

        # 3) Agent 团队决策 (可选)
        if self.use_llm:
            decision = self._stage_agent(decision)

        # 4) 风控过滤
        decision = self._stage_risk(decision, total_capital)

        # 5) 冲击感知路由
        decision = self._stage_routing(decision, stocks_daily, total_capital)

        return decision

    # ---------- stages ----------
    def _stage_regime(self, index_df, stocks_daily, decision):
        try:
            from market_regime import RegimeDetector
            regime_signal = RegimeDetector().detect(index_df, stocks_daily)
            decision.regime = regime_signal.regime.value
            decision.position_mult = regime_signal.position_mult
            decision.notes.append(regime_signal.to_agent_context())
        except Exception as e:
            decision.notes.append(f"regime 失败: {e}")
        return decision

    def _stage_screen(self, scored, decision):
        sorted_c = sorted(scored.items(), key=lambda x: -x[1])
        top = sorted_c[: self.top_k]
        decision.candidates = [
            {"code": c, "score": float(s), "rank": i + 1}
            for i, (c, s) in enumerate(top)
        ]
        return decision

    def _stage_agent(self, decision):
        """调用 TradingAgentTeam 做多智能体决策."""
        try:
            from llm_layer import TradingAgentTeam
            team = TradingAgentTeam(backend=self.llm_backend, debate_rounds=1)
            for c in decision.candidates[: min(3, len(decision.candidates))]:
                try:
                    agent_dec = asyncio.run(team.decide_async(
                        code=c["code"], name=c["code"],
                        fundamentals="(自动生成, 实盘应拉 akshare 财报)",
                        kline="(K线摘要)", factor_score=c["score"],
                        indicators="(技术指标)", sentiment_data="(情绪)",
                        portfolio_state={"current_position": 0,
                                          "available_cash": 1_000_000,
                                          "total_value": 1_000_000,
                                          "position_count": 0,
                                          "daily_pnl": 0, "current_drawdown": 0,
                                          "cash_ratio": 1,
                                          "industry_distribution": "{}"},
                        macro_signals={"regime": decision.regime,
                                        "market_trend": "neutral",
                                        "money_effect": "neutral",
                                        "events": "无"},
                    ))
                    c["agent_action"] = agent_dec.action
                    c["agent_conviction"] = agent_dec.conviction
                except Exception as e:
                    c["agent_error"] = str(e)[:100]
        except Exception as e:
            decision.notes.append(f"agent 阶段失败: {e}")
        return decision

    def _stage_risk(self, decision, total_capital):
        """风控: regime_mult 总仓 + PreTradeGate 逐票过闸.

        实盘和回测共用 PreTradeGate 中间件, 保证规则一致.
        """
        tradeable_cash = total_capital * decision.position_mult
        per_stock_max = total_capital * 0.15
        n = len(decision.candidates)
        if n == 0:
            return decision
        alloc = min(tradeable_cash / n, per_stock_max)

        # 构造一个"当前 portfolio 状态"以便 gate 判仓位 / 现金率
        try:
            from risk import Portfolio, build_default_gate, OrderIntent
            gate = build_default_gate()
            portfolio = Portfolio(
                cash=total_capital, initial_capital=total_capital,
                high_water_mark=total_capital,
                daily_start_value=total_capital,
            )
            gate_enabled = True
        except Exception as e:
            decision.notes.append(f"gate 不可用: {e}")
            gate_enabled = False

        for c in decision.candidates:
            # agent 否决优先
            if c.get("agent_action") == "hold":
                c["alloc_cny"] = 0
                c["risk_filtered"] = True
                c["risk_reason"] = "agent_hold"
                continue

            if gate_enabled:
                price = float(c.get("ref_price") or c.get("price") or 0) or 50.0
                prev_close = float(c.get("prev_close") or price)
                shares = int((alloc / max(price, 1e-6)) // 100 * 100)
                if shares <= 0:
                    c["alloc_cny"] = 0
                    c["risk_filtered"] = True
                    c["risk_reason"] = "不足1手"
                    continue
                intent = OrderIntent(
                    code=c["code"], side="buy", shares=shares,
                    price=price, prev_close=prev_close,
                    industry=str(c.get("industry", "")),
                    suspended=bool(c.get("suspended", False)),
                    conviction=float(c.get("agent_conviction",
                                            c.get("score", 0.5))),
                )
                dec = gate.check(intent, portfolio)
                if dec.allow:
                    c["alloc_cny"] = float(dec.adjusted_shares * price)
                    c["risk_filtered"] = False
                    c["risk_reason"] = dec.reason
                else:
                    c["alloc_cny"] = 0
                    c["risk_filtered"] = True
                    c["risk_reason"] = dec.reason
            else:
                c["alloc_cny"] = float(alloc)
                c["risk_filtered"] = False

        if gate_enabled:
            decision.notes.append(f"gate_stats={gate.stats.to_dict()}")
        return decision

    def _stage_routing(self, decision, stocks_daily, total_capital):
        """用 ImpactAwareRouter 生成下单计划."""
        try:
            from execution import ImpactAwareRouter
            router = ImpactAwareRouter(
                target_participation=self.target_participation,
            )
            orders = []
            for c in decision.candidates:
                if c["alloc_cny"] <= 0:
                    continue
                # 从 stocks_daily 查 ref_price, volume (简化: 默认值)
                row = stocks_daily[
                    stocks_daily["code"] == c["code"]
                ].head(1) if "code" in stocks_daily.columns else None
                if row is None or row.empty:
                    ref_price = 50.0
                    daily_volume = 1_000_000
                else:
                    ref_price = float(row["close"].iloc[0]) if "close" in row.columns else 50.0
                    daily_volume = int(row["volume"].iloc[0]) if "volume" in row.columns else 1_000_000

                shares = int(c["alloc_cny"] / ref_price / 100) * 100
                if shares <= 0:
                    continue
                plan = router.plan_order(
                    total_shares=shares,
                    price=ref_price,
                    daily_volume=max(daily_volume, 10000),
                    volatility=0.025,
                )
                orders.append({
                    "code": c["code"],
                    "shares": shares,
                    "ref_price": ref_price,
                    "cost_bps": plan.expected_total_cost_bps,
                    "n_slices": len(plan.slices),
                    "notes": plan.notes,
                })
            decision.orders = orders
        except Exception as e:
            decision.notes.append(f"routing 失败: {e}")
        return decision

    def record_trade_outcomes(
        self, trades_closed: list[dict],
    ) -> int:
        """交易闭环后, 调用 Memory Curator 存反思.

        Args:
            trades_closed: [{code, entry_date, exit_date, entry_price,
                             exit_price, shares, pnl_pct, regime, ...}]
        """
        try:
            from memory import MemoryCurator
            from memory.curator import TradeOutcome
            curator = MemoryCurator(llm_backend=None)
            n_saved = 0
            for t in trades_closed:
                outcome = TradeOutcome(
                    code=t.get("code", ""),
                    entry_date=t.get("entry_date", ""),
                    exit_date=t.get("exit_date", ""),
                    entry_price=float(t.get("entry_price", 0)),
                    exit_price=float(t.get("exit_price", 0)),
                    shares=int(t.get("shares", 0)),
                    holding_days=int(t.get("holding_days", 0)),
                    pnl_pct=float(t.get("pnl_pct", 0)),
                    entry_reasoning=str(t.get("entry_reasoning", "")),
                    exit_trigger=str(t.get("exit_trigger", "未知")),
                    market_regime=str(t.get("regime", "unknown")),
                )
                if curator.reflect_on_trade(outcome):
                    n_saved += 1
            return n_saved
        except Exception as e:
            logger.warning(f"记录反思失败: {e}")
            return 0
