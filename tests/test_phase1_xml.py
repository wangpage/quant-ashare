"""Phase 1 测试: XML 结构化 Prompt + Parser.

不需要真实 LLM API, 用 mock 输出验证:
  1. xml_parser 能稳定抽取闭合/未闭合 tag
  2. prompts.py 所有模板都能 .format 成功
  3. agents.TradingAgentTeam(backend='mock') 端到端跑通
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tabulate import tabulate

from llm_layer import xml_parser as xp
from llm_layer import prompts
from llm_layer.agents import TradingAgentTeam


def _a(cond, msg): return bool(cond), msg


# ==================== T1: xml_parser ====================
def test_closed_tags():
    text = """<THINKING>思考内容</THINKING>
<REASONING>理由1;理由2</REASONING>
<SCORE>0.35</SCORE>
<CONVICTION>0.7</CONVICTION>
<ACTION>buy</ACTION>"""
    return [
        _a(xp.extract_tag(text, "THINKING") == "思考内容", "闭合 THINKING"),
        _a(xp.extract_tag(text, "REASONING") == "理由1;理由2", "闭合 REASONING"),
        _a(xp.extract_score(text, "SCORE") == 0.35, "SCORE 数值"),
        _a(xp.extract_score(text, "CONVICTION") == 0.7, "CONVICTION"),
        _a(xp.extract_action(text) == "buy", "ACTION buy"),
    ]


def test_unclosed_tags():
    text = """<THINKING>半截思考
<REASONING>接着是推理</REASONING>"""
    return [
        _a("半截思考" in xp.extract_tag(text, "THINKING"), "未闭合 THINKING 回退"),
        _a(xp.extract_tag(text, "REASONING") == "接着是推理", "闭合 REASONING 正常"),
    ]


def test_extract_solution():
    text = """<THINKING>x</THINKING>
<REASONING>y</REASONING>
<SCORE>0.8</SCORE>
<CONVICTION>0.9</CONVICTION>
<ACTION>buy</ACTION>
<SOLUTION>bullish, buy, size_pct=0.1</SOLUTION>"""
    sol = xp.extract_solution(text)
    return [
        _a(sol["score"] == 0.8, "score"),
        _a(sol["conviction"] == 0.9, "conviction"),
        _a(sol["action"] == "buy", "action"),
        _a("bullish" in sol["solution"], "solution"),
    ]


def test_view_detection():
    return [
        _a(xp.extract_view("<SOLUTION>bullish 强</SOLUTION>") == "bullish", "bullish 英文"),
        _a(xp.extract_view("<SOLUTION>看空</SOLUTION>") == "bearish", "看空 中文"),
        _a(xp.extract_view("<SOLUTION>不明</SOLUTION>") == "neutral", "neutral"),
    ]


def test_na_handling():
    text = "<SCORE>N/A</SCORE>"
    return [_a(xp.extract_score(text) is None, "N/A 返回 None")]


# ==================== T2: prompts 模板 ====================
def test_prompt_format():
    """所有 prompt 模板都能 .format 成功, 无占位符漏写."""
    test_kwargs = {
        "code": "300750", "name": "宁德时代",
        "fundamentals": "市值 1.4 万亿", "memory_recall": "无",
        "kline": "近30日均线上行", "factor_score": 0.42,
        "indicators": "RSI 65", "sentiment_data": "新闻中性",
        "fundamental_view": "bullish", "technical_view": "neutral",
        "sentiment_view": "bearish", "event_view": "neutral: 无事件信号",
        "radar_summary": "(测试)",
        "bull_prev": "多头立论...", "bear_attack": "空头攻击...",
        "bear_prev": "空头立论...", "bull_attack": "多头攻击...",
        "n_rounds": 2, "final_bull": "最终多头", "final_bear": "最终空头",
        "researcher_output": "裁决: neutral",
        "current_position": 0, "available_cash": 1_000_000,
        "market_regime": "bull_trending",
        "max_position_pct": 0.15, "max_industry_pct": 0.30,
        "stop_loss": 5.0, "trade_memory": "无",
        "trade_order": "{'action':'buy'}", "total_value": 1_000_000,
        "position_count": 0, "daily_pnl": 0,
        "current_drawdown": 0, "cash_ratio": 1,
        "industry_distribution": "{}",
        "market_trend": "up", "money_effect": "温", "events": "无",
        "risk_memory": "无",
        "trade": "mock trade", "entry_reasoning": "...",
        "exit_trigger": "止盈", "pnl": 5.2, "holding_days": 3,
        "regime": "bull_trending", "reflections": "历史反思...",
    }
    templates = [
        "FUNDAMENTAL_ANALYST_PROMPT", "TECHNICAL_ANALYST_PROMPT",
        "SENTIMENT_ANALYST_PROMPT", "EVENT_ANALYST_PROMPT",
        "RESEARCHER_INITIAL_BULL_PROMPT",
        "RESEARCHER_INITIAL_BEAR_PROMPT", "RESEARCHER_BULL_REBUTTAL_PROMPT",
        "RESEARCHER_BEAR_REBUTTAL_PROMPT", "RESEARCHER_JUDGE_PROMPT",
        "TRADER_PROMPT", "RISK_MANAGER_PROMPT",
        "POST_TRADE_REFLECTION_PROMPT", "MEMORY_NUDGE_PROMPT",
    ]
    checks = []
    for name in templates:
        tpl = getattr(prompts, name)
        try:
            out = tpl.format(**test_kwargs)
            checks.append(_a(len(out) > 100, f"{name} 渲染"))
        except KeyError as e:
            checks.append(_a(False, f"{name} 缺字段: {e}"))
    return checks


# ==================== T3: Agent Team (mock) ====================
def test_agent_team_mock():
    import asyncio
    team = TradingAgentTeam(backend="mock", debate_rounds=1)
    decision = asyncio.run(team.decide_async(
        code="300750", name="宁德时代",
        fundamentals="mock 基本面",
        kline="mock k线", factor_score=0.4,
        indicators="mock 指标", sentiment_data="mock 情绪",
        portfolio_state={"current_position": 0, "available_cash": 1_000_000,
                         "total_value": 1_000_000, "position_count": 0,
                         "daily_pnl": 0, "current_drawdown": 0,
                         "cash_ratio": 1, "industry_distribution": "{}"},
        macro_signals={"regime": "bull_trending", "market_trend": "up",
                       "money_effect": "温", "events": "无"},
    ))
    return [
        _a(decision.action in ("buy", "sell", "hold"), f"action={decision.action}"),
        _a(0 <= decision.conviction <= 1, f"conviction={decision.conviction}"),
        _a(decision.code == "300750", "code"),
        _a("fundamental" in decision.analyst_views, "有基本面分析"),
        _a("research" in decision.analyst_views, "有研究员裁决"),
        _a(len(decision.debate_log) > 0, "有辩论日志"),
        _a(decision.risk_decision in ("approve", "modify", "reject"), "风控输出"),
    ]


def main():
    all_rows = []
    total = total_pass = 0
    for name, fn in [
        ("xml_closed", test_closed_tags),
        ("xml_unclosed", test_unclosed_tags),
        ("xml_solution", test_extract_solution),
        ("xml_view", test_view_detection),
        ("xml_na", test_na_handling),
        ("prompt_format", test_prompt_format),
        ("agent_team_mock", test_agent_team_mock),
    ]:
        for ok, desc in fn():
            all_rows.append([name, desc, "✓" if ok else "✗"])
            total_pass += ok
            total += 1

    print("\n==== Phase 1: XML + Prompt + Agent 测试 ====")
    print(tabulate(all_rows, headers=["模块", "用例", "结果"]))
    print(f"\n通过率: {total_pass}/{total}")
    sys.exit(0 if total_pass == total else 1)


if __name__ == "__main__":
    main()
