"""把 memory.db 里和某只股相关的 radar 事件 + analysis, 组织成文本给 event_analyst.

决策流程:
  1. 拉最近 since_hours 内的 radar_event
  2. 按三级优先筛出和该 code/name 相关的:
     (a) analysis.deep.targets[].code == 输入 code  (最强相关)
     (b) analysis.triage.entities[].code == 输入 code
     (c) analysis.triage.entities[].name 包含 输入 name (模糊)
  3. 超过 max_events 按 |conf| * exp(-hours_since/24) 排序截断
  4. 拼成 ~300 字文本

对外只暴露 build_radar_summary_for_code(code, name, since_hours=72) -> str.
"""
from __future__ import annotations

import math
import time
from typing import Any

from memory.storage import MemoryStore, MemoryRecord
from utils.logger import logger


def _event_match_level(event: MemoryRecord, code: str, name: str) -> int:
    """0 = 无关; 1 = 模糊名匹配; 2 = triage entity code 匹配; 3 = deep target 命中."""
    a = event.metadata.get("analysis") or {}
    d = a.get("deep") or {}
    tr = a.get("triage") or {}

    for t in d.get("targets", []) or []:
        if (t.get("code") or "").strip() == code:
            return 3

    for e in tr.get("entities", []) or []:
        if (e.get("code") or "").strip() == code:
            return 2

    if name:
        for e in tr.get("entities", []) or []:
            if name and name in (e.get("name") or ""):
                return 1
        # 也检查标题里是否直接出现名字
        if name in (event.content or ""):
            return 1

    return 0


def _event_weight(event: MemoryRecord, match_level: int, now_ts: int) -> float:
    """事件相关性加权: 匹配级别 × conf × 时间衰减."""
    a = event.metadata.get("analysis") or {}
    d = a.get("deep") or {}
    # 从 deep.targets 里取和 code 相关的 conf; 若无则用 triage tradability 粗估
    max_conf = 0.0
    for t in d.get("targets", []) or []:
        c = float(t.get("conf") or 0)
        if c > max_conf:
            max_conf = c
    if max_conf == 0 and not d:
        tradability_score = {
            "high": 0.5, "medium": 0.3, "low": 0.1, "none": 0.0
        }.get((a.get("triage") or {}).get("tradability"), 0.1)
        max_conf = tradability_score

    hours_since = max(0, (now_ts - event.ts) / 3600)
    decay = math.exp(-hours_since / 24)
    return match_level * max(0.2, max_conf) * decay


def _format_event(event: MemoryRecord, code: str) -> str:
    """把一条 MemoryRecord + analysis 格式化成几行文本."""
    a = event.metadata.get("analysis") or {}
    tr = a.get("triage") or {}
    d = a.get("deep") or {}
    src = event.metadata.get("source", "?")
    t_str = time.strftime("%m-%d %H:%M", time.localtime(event.ts))

    lines = [f"[{t_str} {src}] {event.content[:80]}"]

    if tr:
        lines.append(
            f"  triage: {tr.get('event_type','?')} · 可交易性 {tr.get('tradability','?')} "
            f"· 新颖度 {tr.get('novelty',0):.1f}"
        )
        if tr.get("oneline"):
            lines.append(f"  一句话: {tr['oneline'][:80]}")

    if d:
        # 只显示和该 code 相关的 target (或者如果没有, 显示第一个)
        targets = d.get("targets", []) or []
        related_targets = [t for t in targets if (t.get("code") or "") == code]
        show_targets = related_targets or targets[:1]  # 主要目标

        for t in show_targets:
            direction = t.get("direction", "?")
            conf = t.get("conf", 0)
            hl = t.get("half_life_hours", 0)
            lines.append(
                f"  deep: 方向 {direction} · conf {conf} · 半衰期 {hl}h"
            )
            if t.get("thesis"):
                lines.append(f"     thesis: {t['thesis'][:140]}")

        if d.get("already_priced_in"):
            lines.append(f"     already_priced_in: {d['already_priced_in']}")
        if d.get("risk_flags"):
            lines.append(f"     risk: {d['risk_flags'][:140]}")
        if d.get("supply_chain"):
            lines.append(f"     supply_chain: {d['supply_chain'][:160]}")
    else:
        lines.append("  (仅 triage, 未 deep)")

    return "\n".join(lines)


def build_radar_summary_for_code(
    code: str, name: str = "",
    since_hours: int = 72, max_events: int = 5,
    store: MemoryStore | None = None,
) -> str:
    """返回该股近期相关 radar 事件的结构化文本.

    Args:
        code: 6 位股票代码
        name: 股票中文名 (辅助模糊匹配, 可留空)
        since_hours: 时间窗, 默认 72h
        max_events: 最多展示几条
        store: MemoryStore, None 则新建

    Returns:
        多行文本, 若无相关事件返回 "近 {since_hours}h 无该股相关 radar 事件 / 证据".
    """
    if not code:
        return "无股票代码, 无法查询 radar 事件"

    store = store or MemoryStore()
    since_ts = int(time.time()) - since_hours * 3600
    events = store.query_radar_events(since_ts=since_ts, limit=500)

    scored: list[tuple[float, int, MemoryRecord]] = []
    now = int(time.time())
    for e in events:
        lvl = _event_match_level(e, code, name)
        if lvl == 0:
            continue
        w = _event_weight(e, lvl, now)
        scored.append((w, lvl, e))

    if not scored:
        return f"近 {since_hours}h 无该股 ({code} {name}) 相关 radar 事件 / 证据"

    # 按 weight 降序, 截前 max_events
    scored.sort(key=lambda x: -x[0])
    top = scored[:max_events]

    deep_hits = sum(1 for _, lvl, _ in top if lvl == 3)
    triage_hits = sum(1 for _, lvl, _ in top if lvl == 2)
    fuzzy_hits = sum(1 for _, lvl, _ in top if lvl == 1)

    header = f"近 {since_hours}h 涉及 {code} {name} 的 radar 事件 (共 {len(top)} 条):"
    body = "\n\n".join(_format_event(e, code) for _, _, e in top)

    summary_line = (
        f"\n\n汇总: deep 直接命中该股 {deep_hits} 条, "
        f"triage 实体命中 {triage_hits} 条, 模糊名匹配 {fuzzy_hits} 条."
    )
    return f"{header}\n\n{body}{summary_line}"
