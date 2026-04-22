"""从 radar analysis 中挑高置信度可交易候选, 供 paper_trade_runner 注入.

职责:
  - 查 memory.db 里今日(或指定窗口)被 radar_worker 分析过的事件
  - 抽 analysis.deep.targets 里 direction=long & conf >= min_conf 的标的
  - 同时抽 direction=avoid 的标的, 供上游"强制不买"
  - 按 conf DESC 排序, 返回前 N 个

用例:
  >>> from llm_layer.radar_candidates import (
  ...     get_radar_long_candidates, get_radar_avoid_codes)
  >>> longs = get_radar_long_candidates(since_hours=24, min_conf=0.6, top=5)
  >>> avoids = get_radar_avoid_codes(since_hours=24, min_conf=0.6)
"""
from __future__ import annotations

import time
from typing import Iterable

from memory.storage import MemoryStore
from utils.logger import logger


def _today_open_ts() -> int:
    import datetime as _dt
    now = _dt.datetime.now()
    return int(now.replace(hour=9, minute=0, second=0, microsecond=0).timestamp())


def _iter_targets(since_ts: int, direction: str,
                   min_conf: float, min_half_life_hours: int = 0,
                   store: MemoryStore | None = None) -> list[dict]:
    """从 radar_event 里按 direction 过滤 deep.targets, 返回展开列表.

    Returns list of dicts: [{code, name, direction, conf, half_life_hours,
                             event_id, event_ts, event_title, thesis, entry}]
    """
    store = store or MemoryStore()
    events = store.query_radar_events(since_ts=since_ts, limit=500)
    out = []
    for e in events:
        a = e.metadata.get("analysis") or {}
        d = a.get("deep") or {}
        for t in d.get("targets", []) or []:
            if t.get("direction") != direction:
                continue
            conf = float(t.get("conf") or 0)
            if conf < min_conf:
                continue
            hl = int(t.get("half_life_hours") or 0)
            if hl < min_half_life_hours:
                continue
            code = (t.get("code") or "").strip()
            if not code or not code.isdigit():  # 过滤 -- 或非 6 位
                continue
            out.append({
                "code": code,
                "name": t.get("name") or "",
                "direction": direction,
                "conf": conf,
                "half_life_hours": hl,
                "thesis": t.get("thesis") or "",
                "entry": t.get("entry") or "",
                "event_id": e.id,
                "event_ts": e.ts,
                "event_title": e.content[:80],
            })
    return out


def get_radar_long_candidates(
    since_hours: int = 0,
    min_conf: float = 0.6,
    top: int = 5,
    min_half_life_hours: int = 24,
    store: MemoryStore | None = None,
) -> list[dict]:
    """挑 radar 里高置信度 long 候选.

    Args:
        since_hours: 窗口小时, 0 表示从今天 9:00 开始
        min_conf: 置信度阈值
        top: 最多返回多少只
        min_half_life_hours: 半衰期下限 (小于此不接入 paper trade,
          因为 T+1 才能买, 短半衰期会被衰减殆尽)

    Returns:
        [{code, name, conf, half_life_hours, thesis, ...}], 按 conf DESC
    """
    since_ts = (int(time.time()) - since_hours * 3600
                if since_hours else _today_open_ts())
    longs = _iter_targets(since_ts, "long", min_conf, min_half_life_hours, store)
    # 同 code 去重, 保留最高 conf
    by_code: dict[str, dict] = {}
    for c in longs:
        if c["code"] not in by_code or c["conf"] > by_code[c["code"]]["conf"]:
            by_code[c["code"]] = c
    ranked = sorted(by_code.values(), key=lambda x: -x["conf"])[:top]
    return ranked


def get_radar_avoid_codes(
    since_hours: int = 0,
    min_conf: float = 0.6,
    store: MemoryStore | None = None,
) -> set[str]:
    """挑 radar 里明确 avoid 的代码集合, 供 paper trade 强制剔除.

    典型场景: 龙头已 price-in, analyst 告诉你别追高. 即使 ML 还排在 top,
    这条 avoid 应该压过 ML.
    """
    since_ts = (int(time.time()) if False  # 保留未来扩展
                else _today_open_ts())
    avoids = _iter_targets(since_ts, "avoid", min_conf, 0, store)
    return {a["code"] for a in avoids}


def reorder_top_with_radar(
    top_index: Iterable[str],
    radar_longs: list[dict],
    radar_avoids: set[str],
    max_k: int,
) -> tuple[list[str], dict]:
    """重排 top-K 列表: radar longs 插队到最前, avoids 强制剔除.

    Args:
        top_index: ML 模型原始排序 (按 rank 的 code list)
        radar_longs: get_radar_long_candidates 返回值
        radar_avoids: get_radar_avoid_codes 返回值
        max_k: 最终输出上限

    Returns:
        (final_codes_list, stats_dict)
        stats 包含: radar_added, ml_kept, avoid_removed, avoid_hit_in_top
    """
    radar_codes = [c["code"] for c in radar_longs]

    # 1. ML top 里剔除 avoid
    ml_kept = [c for c in top_index if c not in radar_avoids]
    avoid_hit = [c for c in top_index if c in radar_avoids]

    # 2. radar longs 在前 (已按 conf 排序), 然后是 ML 剩余
    seen: set[str] = set()
    final: list[str] = []
    for code in radar_codes + list(ml_kept):
        if code in seen:
            continue
        seen.add(code)
        final.append(code)
        if len(final) >= max_k:
            break

    stats = {
        "radar_added": len([c for c in radar_codes if c in final]),
        "ml_kept": len([c for c in final if c not in radar_codes]),
        "avoid_removed": len(avoid_hit),
        "avoid_hit_in_top": avoid_hit,
    }
    return final, stats


def log_injection_summary(longs: list[dict], avoids: set[str],
                            stats: dict) -> None:
    """paper_trade_runner 调用时, 打印可审计的一段 log."""
    logger.info(
        f"[radar 注入] long候选 {len(longs)} 只 / avoid {len(avoids)} 只 "
        f"→ 实际注入 {stats['radar_added']} 只, ML 保留 {stats['ml_kept']} 只, "
        f"剔除 avoid 命中 {stats['avoid_removed']} 只"
    )
    for c in longs[:stats["radar_added"]]:
        logger.info(
            f"  [long] {c['code']} {c['name']:10s} conf={c['conf']:.2f} "
            f"半衰期{c['half_life_hours']}h  来源#{c['event_id']}"
        )
    for code in stats["avoid_hit_in_top"]:
        logger.info(f"  [avoid] {code} (ML 推荐但 radar 建议避开)")
