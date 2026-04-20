"""记忆策展人 (Curator) - Hermes 风格:

职责:
  1. 每笔交易闭环后调用 reflect_on_trade(), LLM 提炼反思存入记忆
  2. 每周调用 weekly_nudge(), LLM 合并/去重/提炼过去 1 周的反思
  3. 每月调用 monthly_audit(), 评估哪些 skill 仍然有效

参考 Hermes Agent "agent-curated memory with periodic nudges".
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from utils.logger import logger

from .storage import MemoryStore, default_store


@dataclass
class TradeOutcome:
    """闭环交易记录, 用于复盘."""
    code: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    shares: int
    holding_days: int
    pnl_pct: float                 # 收益率 (小数)
    entry_reasoning: str           # 入场时的 reasoning
    exit_trigger: str              # 止盈/止损/主动卖
    market_regime: str             # 当时的 regime

    @property
    def pnl_absolute(self) -> float:
        return (self.exit_price - self.entry_price) * self.shares


class MemoryCurator:
    def __init__(self, llm_backend=None, store: MemoryStore | None = None):
        """
        Args:
            llm_backend: 可调用对象, 接受 prompt 返回字符串 (Hermes XML).
                         为 None 时使用 mock (便于离线测试).
            store: MemoryStore 实例, 默认全局单例.
        """
        self.llm = llm_backend
        self.store = store or default_store()

    # ==================== 1. 单笔交易反思 ====================
    def reflect_on_trade(self, trade: TradeOutcome) -> int | None:
        """对一笔闭环交易做 LLM 复盘, 提炼反思存入记忆.

        Returns:
            新增记忆的 id, LLM 不可用时返回 None.
        """
        from llm_layer import prompts, xml_parser as xp

        prompt = prompts.POST_TRADE_REFLECTION_PROMPT.format(
            trade=self._fmt_trade(trade),
            entry_reasoning=trade.entry_reasoning[:500],
            exit_trigger=trade.exit_trigger,
            pnl=trade.pnl_pct * 100,
            holding_days=trade.holding_days,
            regime=trade.market_regime,
        )
        raw = self._call_llm(prompt)
        if not raw:
            logger.info(f"[{trade.code}] 未配置 LLM, 跳过反思 (规则化模式记录基础信息)")
            return self._save_basic_reflection(trade)

        reflection = xp.extract_tag(raw, "REFLECTION")
        if not reflection:
            reflection = xp.extract_tag(raw, "SOLUTION") or raw[:500]

        quality_score_raw = xp.extract_tag(raw, "SCORE")
        try:
            quality = float(quality_score_raw) if quality_score_raw else 0.0
        except ValueError:
            quality = 0.0

        id_ = self.store.add(
            kind="reflection",
            content=reflection,
            code=trade.code,
            outcome_pnl=trade.pnl_pct,
            metadata={
                "regime": trade.market_regime,
                "holding_days": trade.holding_days,
                "quality_score": quality,
                "exit_trigger": trade.exit_trigger,
                "entry_date": trade.entry_date,
                "exit_date": trade.exit_date,
            },
        )
        logger.info(f"[{trade.code}] 反思已存 id={id_}, pnl={trade.pnl_pct:+.2%}")
        return id_

    def _save_basic_reflection(self, trade: TradeOutcome) -> int:
        """LLM 不可用时的 fallback: 规则化生成反思."""
        if trade.pnl_pct > 0.05:
            content = f"[{trade.market_regime}][{trade.holding_days}日持有] 盈利 {trade.pnl_pct:+.2%}, 触发 {trade.exit_trigger}. 入场逻辑可能有效."
        elif trade.pnl_pct < -0.03:
            content = f"[{trade.market_regime}][{trade.holding_days}日持有] 亏损 {trade.pnl_pct:+.2%}, 触发 {trade.exit_trigger}. 入场逻辑需复盘."
        else:
            content = f"[{trade.market_regime}] 微幅 {trade.pnl_pct:+.2%}, 中性记录."
        return self.store.add(
            kind="reflection", content=content, code=trade.code,
            outcome_pnl=trade.pnl_pct,
            metadata={"regime": trade.market_regime,
                      "holding_days": trade.holding_days,
                      "fallback": True},
        )

    # ==================== 2. 每周整合 ====================
    def weekly_nudge(self, days: int = 7) -> list[int]:
        """整合过去 N 天的反思, 提炼精华规则, 旧反思降权.

        Returns: 新增 rule 记录的 id 列表.
        """
        from llm_layer import prompts, xml_parser as xp

        recent = self.store.recent(kind="reflection", days=days, limit=100)
        if len(recent) < 3:
            logger.info(f"反思数量 ({len(recent)}) 太少, 跳过本周整合")
            return []

        reflections_txt = "\n".join(
            f"- [{r.ts}][pnl={r.outcome_pnl:+.2%}][{r.code}] {r.content[:150]}"
            if r.outcome_pnl is not None else
            f"- [{r.ts}][{r.code}] {r.content[:150]}"
            for r in recent
        )
        prompt = prompts.MEMORY_NUDGE_PROMPT.format(reflections=reflections_txt)

        raw = self._call_llm(prompt)
        if not raw:
            logger.info("LLM 不可用, 跳过本周整合")
            return []

        solution = xp.extract_tag(raw, "SOLUTION") or xp.extract_tag(raw, "REFLECTION")
        if not solution:
            return []

        # 把结论拆成多条 rule (按行分, 过滤空)
        rules = [l.strip().lstrip("•-*0123456789. ") for l in solution.split("\n") if l.strip()]
        rules = [r for r in rules if len(r) > 10]

        ids = []
        for rule in rules:
            id_ = self.store.add(
                kind="rule",
                content=rule,
                metadata={"source": "weekly_nudge",
                          "aggregated_from": len(recent)},
            )
            ids.append(id_)
        logger.info(f"本周整合完成: {len(recent)} 条反思 -> {len(rules)} 条规则")
        return ids

    # ==================== 3. 每月审计 ====================
    def monthly_audit(self) -> dict:
        """审计: 规则有效性 / 反思堆积 / 是否需要淘汰."""
        stats = self.store.stats()
        old_reflections = self.store.recent(kind="reflection", days=180, limit=1000)

        # 把超过 90 天的亏损反思但 pnl 很小 (<-5%) 的保留, 微亏的可以考虑清理
        to_delete = []
        now = int(time.time())
        for r in old_reflections:
            if (now - r.ts) > 90 * 86400:   # 超过 90 天
                if r.outcome_pnl is not None and -0.05 < r.outcome_pnl < 0.05:
                    to_delete.append(r.id)

        for id_ in to_delete:
            self.store.delete(id_)

        result = {
            **stats,
            "audited_at": now,
            "deleted_noise": len(to_delete),
        }
        logger.info(f"月度审计: {result}")
        return result

    # ==================== 工具 ====================
    def _call_llm(self, prompt: str) -> str | None:
        if self.llm is None:
            return None
        try:
            if callable(self.llm):
                return self.llm(prompt)
            if hasattr(self.llm, "chat"):
                return self.llm.chat(prompt)
        except Exception as e:
            logger.warning(f"LLM 调用失败: {e}")
        return None

    def _fmt_trade(self, t: TradeOutcome) -> str:
        return (f"{t.code} {t.entry_date}->{t.exit_date} "
                f"buy@{t.entry_price:.2f} sell@{t.exit_price:.2f} "
                f"shares={t.shares} regime={t.market_regime}")
