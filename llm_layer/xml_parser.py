"""从 Hermes XML 输出中稳定抽取 tag 内容.

不依赖严格 XML 解析器, 因为 LLM 可能有嵌套/未闭合/特殊字符.
采用宽松正则 + 回退策略:
  1. 首选 <TAG>...</TAG> 非贪婪匹配
  2. 若未闭合, 匹配 <TAG> 到下一个 <OTHER_TAG> 或文末
"""
from __future__ import annotations

import re
from typing import Any

# 所有我们关心的 tag, 按 prompt 设计
KNOWN_TAGS = [
    "SCRATCHPAD", "THINKING", "PLAN", "REASONING",
    "INNER_MONOLOGUE", "REFLECTION", "EXECUTION",
    "SOLUTION", "EXPLANATION", "RISK",
    "SCORE", "CONVICTION", "ACTION",
]


def extract_tag(text: str, tag: str, default: str = "") -> str:
    """抽取 <TAG>...</TAG> 内容, 剥除首尾空白.

    Args:
        text: LLM 原始输出
        tag: 不含 < > 的标签名, 例如 "REASONING"
        default: 找不到时返回

    Returns:
        tag 内部文本. 如果未闭合, 匹配到下一个已知 tag 为止.
    """
    if not text:
        return default

    closed = re.search(
        rf"<{tag}>(.*?)</{tag}>",
        text, re.DOTALL | re.IGNORECASE,
    )
    if closed:
        return closed.group(1).strip()

    # 未闭合: 匹配到下一个已知 tag 或文末
    other = "|".join(KNOWN_TAGS)
    unclosed = re.search(
        rf"<{tag}>(.*?)(?:<(?:{other})>|$)",
        text, re.DOTALL | re.IGNORECASE,
    )
    if unclosed:
        return unclosed.group(1).strip()

    return default


def extract_all(text: str) -> dict[str, str]:
    """一次抽取所有已知 tag 的内容."""
    return {t.lower(): extract_tag(text, t) for t in KNOWN_TAGS}


def extract_score(text: str, tag: str = "SCORE") -> float | None:
    """从 tag 内抽取第一个浮点数 [-1, 1] 或 [0, 1]."""
    raw = extract_tag(text, tag)
    if not raw or raw.strip().upper() == "N/A":
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", raw)
    if not m:
        return None
    try:
        v = float(m.group(0))
        return max(-1.0, min(1.0, v))
    except ValueError:
        return None


def extract_action(text: str) -> str:
    """从 <ACTION> 抽 buy/sell/hold/approve/modify/reject."""
    raw = extract_tag(text, "ACTION").lower()
    for keyword in ["buy", "sell", "hold", "approve", "modify", "reject"]:
        if keyword in raw:
            return keyword
    # 回退: 从 SOLUTION 中找
    sol = extract_tag(text, "SOLUTION").lower()
    for keyword in ["buy", "sell", "hold"]:
        if keyword in sol:
            return keyword
    return "hold"


def extract_solution(text: str) -> dict[str, Any]:
    """把一次 LLM 输出整理成结构化结果, 用于下游消费."""
    return {
        "score":        extract_score(text, "SCORE"),
        "conviction":   extract_score(text, "CONVICTION"),
        "action":       extract_action(text),
        "thinking":     extract_tag(text, "THINKING"),
        "reasoning":    extract_tag(text, "REASONING"),
        "plan":         extract_tag(text, "PLAN"),
        "reflection":   extract_tag(text, "REFLECTION"),
        "inner":        extract_tag(text, "INNER_MONOLOGUE"),
        "risk":         extract_tag(text, "RISK"),
        "solution":     extract_tag(text, "SOLUTION"),
        "explanation":  extract_tag(text, "EXPLANATION"),
        "scratchpad":   extract_tag(text, "SCRATCHPAD"),
    }


def extract_view(text: str) -> str:
    """<SOLUTION> 里找 bullish/bearish/neutral 关键字."""
    sol = extract_tag(text, "SOLUTION").lower()
    if "bullish" in sol or "看多" in sol or "做多" in sol:
        return "bullish"
    if "bearish" in sol or "看空" in sol or "做空" in sol:
        return "bearish"
    return "neutral"
