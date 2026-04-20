"""多智能体交易团队 (Hermes XML 风格输出).

架构:
  三分析师 (并行 spawn) -> 研究员多轮辩论 -> 交易员 -> 风控经理

每个 agent 的输出都是 Hermes XML tag, 用 xml_parser 稳定抽取.
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from utils.logger import logger

from . import prompts
from . import xml_parser as xp


@dataclass
class AgentDecision:
    action: str
    code: str
    size_pct: float
    price: float | None = None
    stop_loss_price: float | None = None
    take_profit_price: float | None = None
    holding_days: int = 5
    reasoning: str = ""
    conviction: float = 0.0
    score: float = 0.0
    analyst_views: dict = field(default_factory=dict)
    debate_log: list = field(default_factory=list)
    risk_decision: str = "approve"
    raw_traces: dict = field(default_factory=dict)


# ==================== LLM 后端 ====================
class _LLMBackend:
    """多后端统一接口, 支持 Anthropic / OpenAI / 国产 (DeepSeek/Kimi/Qwen/GLM) / OpenRouter / mock.

    所有 OpenAI 兼容 API 只要 base_url + api_key 即可工作.
    环境变量:
        ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL
        OPENAI_API_KEY / OPENAI_BASE_URL
        DEEPSEEK_API_KEY, MOONSHOT_API_KEY, DASHSCOPE_API_KEY, ZHIPUAI_API_KEY
        OPENROUTER_API_KEY
    """

    _OPENAI_COMPATIBLE = {
        "deepseek":   ("DEEPSEEK_API_KEY",   "https://api.deepseek.com/v1"),
        "moonshot":   ("MOONSHOT_API_KEY",   "https://api.moonshot.cn/v1"),
        "kimi":       ("MOONSHOT_API_KEY",   "https://api.moonshot.cn/v1"),
        "qwen":       ("DASHSCOPE_API_KEY",  "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        "dashscope":  ("DASHSCOPE_API_KEY",  "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        "zhipu":      ("ZHIPUAI_API_KEY",    "https://open.bigmodel.cn/api/paas/v4"),
        "glm":        ("ZHIPUAI_API_KEY",    "https://open.bigmodel.cn/api/paas/v4"),
        "openrouter": ("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1"),
    }

    def __init__(self, backend: str = "anthropic",
                 model: str = "claude-sonnet-4-6"):
        self.backend = backend
        self.model = model
        self._client = None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    def chat(self, prompt: str, max_tokens: int = 1500) -> str:
        import os
        if self.backend == "anthropic":
            import anthropic
            if self._client is None:
                self._client = anthropic.Anthropic(
                    base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
                )
            r = self._client.messages.create(
                model=self.model, max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return r.content[0].text

        if self.backend == "openai":
            import openai
            if self._client is None:
                self._client = openai.OpenAI(
                    base_url=os.getenv("OPENAI_BASE_URL") or None,
                )
            r = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
            return r.choices[0].message.content

        # OpenAI-兼容的国产 / 中转
        if self.backend in self._OPENAI_COMPATIBLE:
            import openai
            env_key, default_base = self._OPENAI_COMPATIBLE[self.backend]
            api_key = os.getenv(env_key)
            if not api_key:
                raise ValueError(f"缺环境变量 {env_key}")
            if self._client is None:
                self._client = openai.OpenAI(
                    api_key=api_key,
                    base_url=os.getenv(f"{env_key}_BASE_URL", default_base),
                )
            r = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
            return r.choices[0].message.content

        if self.backend == "mock":
            return self._mock_response(prompt)

        raise ValueError(f"未知 backend: {self.backend}")

    def _mock_response(self, prompt: str) -> str:
        """根据 prompt 角色返回不同 mock 输出, 便于测试."""
        is_risk = "风控经理" in prompt or "risk_manager" in prompt.lower() \
                  or "approve / modify / reject" in prompt
        is_trader = "A股交易员" in prompt and not is_risk
        action = "approve" if is_risk else ("buy" if is_trader else "hold")
        view = "approve" if is_risk else "bullish"
        return f"""<THINKING>mock agent 思考中</THINKING>
<SCRATCHPAD>mock 演算</SCRATCHPAD>
<PLAN>size_pct=0.1 entry=100 stop=95 target=110 holding=5</PLAN>
<REASONING>mock 推理1; mock 推理2</REASONING>
<INNER_MONOLOGUE>mock 自省</INNER_MONOLOGUE>
<REFLECTION>mock 反思</REFLECTION>
<RISK>mock 风险1; mock 风险2</RISK>
<SCORE>0.3</SCORE>
<CONVICTION>0.5</CONVICTION>
<ACTION>{action}</ACTION>
<SOLUTION>{view}, action={action}, size_pct=0.10</SOLUTION>
<EXPLANATION>mock 一句话</EXPLANATION>"""


# ==================== 团队 ====================
class TradingAgentTeam:
    def __init__(
        self,
        backend: str = "anthropic",
        analyst_model: str = "claude-haiku-4-5",
        researcher_model: str = "claude-sonnet-4-6",
        trader_model: str = "claude-sonnet-4-6",
        risk_model: str = "claude-haiku-4-5",
        debate_rounds: int = 2,
    ):
        self.fundamental = _LLMBackend(backend, analyst_model)
        self.technical   = _LLMBackend(backend, analyst_model)
        self.sentiment   = _LLMBackend(backend, analyst_model)
        self.bull        = _LLMBackend(backend, researcher_model)
        self.bear        = _LLMBackend(backend, researcher_model)
        self.judge       = _LLMBackend(backend, researcher_model)
        self.trader      = _LLMBackend(backend, trader_model)
        self.risk_mgr    = _LLMBackend(backend, risk_model)
        self.debate_rounds = debate_rounds

    # ---------- 三分析师 ----------
    def analyze_fundamental(self, code: str, name: str, fundamentals: str,
                            memory_recall: str = "无") -> dict:
        p = prompts.FUNDAMENTAL_ANALYST_PROMPT.format(
            code=code, name=name, fundamentals=fundamentals,
            memory_recall=memory_recall,
        )
        raw = self.fundamental.chat(p)
        return {**xp.extract_solution(raw), "_raw": raw,
                "view": xp.extract_view(raw)}

    def analyze_technical(self, code: str, name: str, kline: str,
                          factor_score: float, indicators: str,
                          memory_recall: str = "无") -> dict:
        p = prompts.TECHNICAL_ANALYST_PROMPT.format(
            code=code, name=name, kline=kline,
            factor_score=factor_score, indicators=indicators,
            memory_recall=memory_recall,
        )
        raw = self.technical.chat(p)
        return {**xp.extract_solution(raw), "_raw": raw,
                "view": xp.extract_view(raw)}

    def analyze_sentiment(self, code: str, name: str, sentiment_data: str,
                          memory_recall: str = "无") -> dict:
        p = prompts.SENTIMENT_ANALYST_PROMPT.format(
            code=code, name=name, sentiment_data=sentiment_data,
            memory_recall=memory_recall,
        )
        raw = self.sentiment.chat(p)
        return {**xp.extract_solution(raw), "_raw": raw,
                "view": xp.extract_view(raw)}

    # ---------- 研究员多轮辩论 ----------
    def debate(self, code: str, fund: dict, tech: dict, sent: dict) -> dict:
        debate_log = []

        bull_p = prompts.RESEARCHER_INITIAL_BULL_PROMPT.format(
            code=code,
            fundamental_view=fund.get("solution", ""),
            technical_view=tech.get("solution", ""),
            sentiment_view=sent.get("solution", ""),
        )
        bear_p = prompts.RESEARCHER_INITIAL_BEAR_PROMPT.format(
            code=code,
            fundamental_view=fund.get("solution", ""),
            technical_view=tech.get("solution", ""),
            sentiment_view=sent.get("solution", ""),
        )
        bull_raw = self.bull.chat(bull_p)
        bear_raw = self.bear.chat(bear_p)
        debate_log.append({"round": 0, "bull": bull_raw, "bear": bear_raw})

        for i in range(self.debate_rounds):
            bull_reb_p = prompts.RESEARCHER_BULL_REBUTTAL_PROMPT.format(
                bull_prev=xp.extract_tag(bull_raw, "SOLUTION"),
                bear_attack=xp.extract_tag(bear_raw, "REASONING"),
            )
            bear_reb_p = prompts.RESEARCHER_BEAR_REBUTTAL_PROMPT.format(
                bear_prev=xp.extract_tag(bear_raw, "SOLUTION"),
                bull_attack=xp.extract_tag(bull_raw, "REASONING"),
            )
            bull_raw = self.bull.chat(bull_reb_p)
            bear_raw = self.bear.chat(bear_reb_p)
            debate_log.append({"round": i + 1, "bull": bull_raw, "bear": bear_raw})

        judge_p = prompts.RESEARCHER_JUDGE_PROMPT.format(
            n_rounds=self.debate_rounds,
            final_bull=xp.extract_tag(bull_raw, "SOLUTION") + "\n" +
                       xp.extract_tag(bull_raw, "REASONING"),
            final_bear=xp.extract_tag(bear_raw, "SOLUTION") + "\n" +
                       xp.extract_tag(bear_raw, "REASONING"),
        )
        judge_raw = self.judge.chat(judge_p)
        return {
            **xp.extract_solution(judge_raw),
            "view": xp.extract_view(judge_raw),
            "_raw": judge_raw,
            "debate_log": debate_log,
        }

    # ---------- 交易员 ----------
    def make_trade(self, code: str, researcher: dict,
                   current_position: float, available_cash: float,
                   market_regime: str = "neutral",
                   max_position_pct: float = 0.15,
                   max_industry_pct: float = 0.30,
                   stop_loss: float = 5.0,
                   trade_memory: str = "无") -> dict:
        p = prompts.TRADER_PROMPT.format(
            researcher_output=researcher.get("solution", "") + " | "
                              + researcher.get("explanation", ""),
            current_position=current_position,
            available_cash=available_cash,
            market_regime=market_regime,
            max_position_pct=max_position_pct,
            max_industry_pct=max_industry_pct,
            stop_loss=stop_loss,
            trade_memory=trade_memory,
        )
        raw = self.trader.chat(p)
        out = xp.extract_solution(raw)
        out["_raw"] = raw
        # 从 SOLUTION/PLAN 抽 size/price
        sol = out.get("solution", "") + " " + out.get("plan", "")
        out["size_pct"] = _extract_pct(sol, ["size_pct", "目标仓位", "size"])
        out["entry_price"] = _extract_num(sol, ["entry", "入场", "买入价"])
        out["stop_loss_price"] = _extract_num(sol, ["stop", "止损"])
        out["take_profit_price"] = _extract_num(sol, ["target", "止盈", "目标位"])
        out["holding_days"] = int(_extract_num(sol, ["holding", "持仓"]) or 5)
        return out

    # ---------- 风控 ----------
    def risk_review(self, trade_order: dict, portfolio_state: dict,
                    macro_signals: dict, risk_memory: str = "无") -> dict:
        p = prompts.RISK_MANAGER_PROMPT.format(
            trade_order=json.dumps(trade_order, ensure_ascii=False, default=str)[:400],
            total_value=portfolio_state.get("total_value", 0),
            position_count=portfolio_state.get("position_count", 0),
            daily_pnl=portfolio_state.get("daily_pnl", 0),
            current_drawdown=portfolio_state.get("current_drawdown", 0),
            cash_ratio=portfolio_state.get("cash_ratio", 0),
            industry_distribution=portfolio_state.get("industry_distribution", ""),
            market_regime=macro_signals.get("regime", "neutral"),
            market_trend=macro_signals.get("market_trend", "neutral"),
            money_effect=macro_signals.get("money_effect", "neutral"),
            events=macro_signals.get("events", "none"),
            risk_memory=risk_memory,
        )
        raw = self.risk_mgr.chat(p)
        out = xp.extract_solution(raw)
        out["_raw"] = raw
        return out

    # ---------- 并行决策入口 ----------
    async def decide_async(
        self,
        code: str, name: str,
        fundamentals: str, kline: str, factor_score: float,
        indicators: str, sentiment_data: str,
        portfolio_state: dict, macro_signals: dict,
        memory_recall: dict | None = None,
    ) -> AgentDecision:
        m = memory_recall or {}
        logger.info(f"[{code}] 启动多智能体决策 (并行分析师)")

        # 1. 三分析师并行 (asyncio.to_thread 包 sync 调用)
        fund_task = asyncio.to_thread(
            self.analyze_fundamental, code, name, fundamentals,
            m.get("fundamental", "无"),
        )
        tech_task = asyncio.to_thread(
            self.analyze_technical, code, name, kline, factor_score,
            indicators, m.get("technical", "无"),
        )
        sent_task = asyncio.to_thread(
            self.analyze_sentiment, code, name, sentiment_data,
            m.get("sentiment", "无"),
        )
        fund, tech, sent = await asyncio.gather(fund_task, tech_task, sent_task)

        # 2. 多轮辩论
        researcher = await asyncio.to_thread(self.debate, code, fund, tech, sent)
        conviction = researcher.get("conviction") or 0.5
        score = researcher.get("score") or 0.0

        # 3. 交易员
        trade = await asyncio.to_thread(
            self.make_trade, code, researcher,
            portfolio_state.get("current_position", 0),
            portfolio_state.get("available_cash", 0),
            macro_signals.get("regime", "neutral"),
            0.15, 0.30, 5.0,
            m.get("trade", "无"),
        )

        # 4. 风控审核
        risk = await asyncio.to_thread(
            self.risk_review, trade, portfolio_state, macro_signals,
            m.get("risk", "无"),
        )

        final_action = trade.get("action", "hold")
        if risk.get("action") == "reject":
            final_action = "hold"
        size = float(risk.get("score") if risk.get("action") == "modify"
                     else trade.get("size_pct", 0) or 0)

        return AgentDecision(
            action=final_action,
            code=code,
            size_pct=size,
            price=trade.get("entry_price"),
            stop_loss_price=trade.get("stop_loss_price"),
            take_profit_price=trade.get("take_profit_price"),
            holding_days=trade.get("holding_days", 5),
            reasoning=f"{researcher.get('explanation', '')} | {trade.get('explanation', '')}",
            conviction=conviction,
            score=score,
            analyst_views={"fundamental": fund, "technical": tech,
                           "sentiment": sent, "research": researcher},
            debate_log=researcher.get("debate_log", []),
            risk_decision=risk.get("action", "approve"),
            raw_traces={
                "fundamental": fund.get("_raw"),
                "technical": tech.get("_raw"),
                "sentiment": sent.get("_raw"),
                "researcher": researcher.get("_raw"),
                "trader": trade.get("_raw"),
                "risk": risk.get("_raw"),
            },
        )

    def decide(self, *args, **kwargs) -> AgentDecision:
        """同步入口."""
        return asyncio.run(self.decide_async(*args, **kwargs))


# ==================== 小工具 ====================
def _extract_num(text: str, keys: list[str]) -> float | None:
    for k in keys:
        m = re.search(rf"{k}[^\d-]*(-?\d+(?:\.\d+)?)", text, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
    return None


def _extract_pct(text: str, keys: list[str]) -> float | None:
    v = _extract_num(text, keys)
    if v is None:
        return None
    # 如果是 0-100 的整数, 归一化到 0-1
    if v > 1.0:
        v = v / 100.0
    return max(0.0, min(1.0, v))
