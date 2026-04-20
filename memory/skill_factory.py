"""Skill 工厂 - 从成功交易自动提炼可复用模式.

灵感: Hermes Agent "autonomous skill creation after complex tasks".

思路:
  1. 每月扫描过去 N 天的成功交易 (pnl > 阈值)
  2. 按特征聚类 (regime + holding_days + 信号类型)
  3. 对每个簇, 让 LLM 提炼 "condition -> action" 规则
  4. 成功率 >= 55% 且样本 >= 5 才激活为 skill
  5. 未来决策时, 如果当前情景匹配某 skill, 作为 few-shot 注入 prompt
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from utils.logger import logger

from .storage import MemoryStore, default_store


@dataclass
class Skill:
    id: int | None
    pattern_name: str
    conditions: str
    actions: str
    success_rate: float
    avg_pnl: float
    sample_count: int


class SkillFactory:
    def __init__(self, llm_backend=None, store: MemoryStore | None = None,
                 min_samples: int = 5, min_success_rate: float = 0.55,
                 profit_threshold: float = 0.03):
        self.llm = llm_backend
        self.store = store or default_store()
        self.min_samples = min_samples
        self.min_success_rate = min_success_rate
        self.profit_threshold = profit_threshold

    # ==================== 主入口: 月度 skill 生成 ====================
    def generate_skills(self, days: int = 90) -> list[Skill]:
        """扫描过去 N 天反思, 按簇生成 skill."""
        reflections = self.store.recent(kind="reflection", days=days, limit=500)
        if len(reflections) < self.min_samples:
            logger.info(f"反思数量 ({len(reflections)}) 不足 {self.min_samples}, 跳过 skill 生成")
            return []

        # 简单聚类: 按 regime + holding_days 分桶 + 涨跌符号
        clusters: dict[tuple, list] = defaultdict(list)
        for r in reflections:
            regime = r.metadata.get("regime", "neutral")
            hold = r.metadata.get("holding_days", 5)
            hold_bucket = "short" if hold <= 3 else ("mid" if hold <= 7 else "long")
            direction = "up" if (r.outcome_pnl or 0) > 0 else "down"
            clusters[(regime, hold_bucket, direction)].append(r)

        skills = []
        for (regime, hold, direction), cluster in clusters.items():
            if len(cluster) < self.min_samples:
                continue

            pnls = [r.outcome_pnl for r in cluster if r.outcome_pnl is not None]
            if not pnls:
                continue

            win = sum(1 for p in pnls if p > self.profit_threshold)
            success_rate = win / len(pnls)
            avg_pnl = sum(pnls) / len(pnls)

            if success_rate < self.min_success_rate:
                continue

            pattern_name = f"{regime}_{hold}_{direction}"
            cluster_content = "\n".join(r.content[:300] for r in cluster[:20])

            # 用 LLM 提炼规则 (若不可用, 用模板)
            conditions, actions = self._extract_rule(
                regime, hold, direction, cluster_content, pnls,
            )

            skill_id = self.store.upsert_skill(
                pattern_name=pattern_name,
                conditions=conditions,
                actions=actions,
                success_rate=success_rate,
                avg_pnl=avg_pnl,
                sample_count=len(cluster),
            )
            skills.append(Skill(
                id=skill_id, pattern_name=pattern_name,
                conditions=conditions, actions=actions,
                success_rate=success_rate, avg_pnl=avg_pnl,
                sample_count=len(cluster),
            ))
            logger.info(f"Skill '{pattern_name}': "
                        f"样本={len(cluster)} 胜率={success_rate:.1%} "
                        f"均值={avg_pnl:+.2%}")

        return skills

    # ==================== 给 agent 的 few-shot 注入 ====================
    def recall_skills_for_context(
        self, regime: str, horizon_days: int = 5,
    ) -> str:
        """给交易员 agent 的 few-shot 注入: 返回匹配当前情景的 skills."""
        all_skills = self.store.active_skills(
            min_samples=self.min_samples,
            min_success=self.min_success_rate,
        )
        if not all_skills:
            return "无历史验证的 skill."

        hold_bucket = "short" if horizon_days <= 3 else ("mid" if horizon_days <= 7 else "long")

        matched = [s for s in all_skills
                   if regime in s["pattern_name"] or hold_bucket in s["pattern_name"]]
        if not matched:
            matched = all_skills[:3]

        lines = []
        for s in matched[:5]:
            lines.append(
                f"[{s['pattern_name']}] 胜率={s['success_rate']:.0%} "
                f"均值={s['avg_pnl']:+.2%} (n={s['sample_count']})\n"
                f"  条件: {s['conditions'][:150]}\n"
                f"  建议: {s['actions'][:150]}"
            )
        return "\n".join(lines)

    # ==================== 淘汰 ====================
    def deactivate_stale_skills(self, max_stale_days: int = 180) -> int:
        """把很久没触发或胜率下降的 skill 停用."""
        active = self.store.active_skills(min_samples=0, min_success=0)
        import time
        now = int(time.time())
        n_deactivated = 0
        for s in active:
            age_days = (now - s["last_updated"]) / 86400
            if age_days > max_stale_days or s["success_rate"] < 0.45:
                self.store.deactivate_skill(s["id"])
                n_deactivated += 1
        logger.info(f"停用 skill {n_deactivated} 个")
        return n_deactivated

    # ==================== 内部工具 ====================
    def _extract_rule(self, regime: str, hold_bucket: str, direction: str,
                      cluster_content: str, pnls: list[float]) -> tuple[str, str]:
        """用 LLM 提炼 condition -> action; 无 LLM 时用模板."""
        if self.llm is None:
            conditions = (f"regime={regime} AND 持仓区间={hold_bucket} "
                          f"AND 历史方向={direction}")
            actions = (f"若出现此情景, 参考历史 {len(pnls)} 笔样本, "
                       f"均值 {sum(pnls)/len(pnls):+.2%}")
            return conditions, actions

        prompt = f"""从以下 {len(pnls)} 笔历史交易反思中, 提炼 1 条可复用的交易规则.
            所有样本都处于 regime={regime}, 持仓 {hold_bucket} 期, 最终方向 {direction}.

            样本反思:
            {cluster_content}

            输出:
            <REASONING>共同模式是什么</REASONING>
            <SOLUTION>
            条件: [满足什么条件时触发]
            行动: [应该怎么做]
            </SOLUTION>
            """
        try:
            raw = self.llm(prompt) if callable(self.llm) else self.llm.chat(prompt)
            from llm_layer import xml_parser as xp
            sol = xp.extract_tag(raw, "SOLUTION")
            if "条件" in sol and "行动" in sol:
                parts = sol.split("行动")
                cond = parts[0].replace("条件", "").replace(":", "").strip()
                act = parts[1].replace(":", "").strip()
                return cond, act
            return sol[:200], "参考历史样本"
        except Exception as e:
            logger.warning(f"LLM 提炼规则失败: {e}")
            return (f"regime={regime} AND hold={hold_bucket}",
                    f"参考 {len(pnls)} 笔样本")
