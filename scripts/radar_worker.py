"""Radar 事件后台分析器 daemon.

从 memory.db 拉 kind=radar_event 且 metadata.analysis IS NULL 的事件,
依次 triage -> (可选) deep_analyze -> 回写 metadata.analysis.

用法:
  python scripts/radar_worker.py                 # 常驻, 每 30s 一轮
  python scripts/radar_worker.py --once          # 跑一轮就退, 调试用
  python scripts/radar_worker.py --limit 5       # 每轮最多处理 5 条
  python scripts/radar_worker.py --deep-budget 3 # 每轮最多触发 3 次 deep_analyze
  python scripts/radar_worker.py --interval 60   # 循环间隔 60s
  python scripts/radar_worker.py --since-hours 2 # 只处理最近 2h 的事件, 默认 24h

后台启动:
  nohup python scripts/radar_worker.py > logs/radar_worker.log 2>&1 &
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 加载 .env (cron_daily.py 模式)
_env = ROOT / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

from utils.logger import logger  # noqa: E402
from memory.storage import MemoryStore  # noqa: E402
from llm_layer.radar_analyst import (  # noqa: E402
    triage, deep_analyze, should_deep_analyze,
)
from llm_layer import market_context as mc  # noqa: E402


_STOP = False


def _handle_signal(signum, _frame):
    global _STOP
    logger.info(f"收到 signal {signum}, 下一轮后退出")
    _STOP = True


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


def analyze_one(store: MemoryStore, event_rec, deep_budget_left: int) -> tuple[dict, bool]:
    """对一条 MemoryRecord 跑 triage + (可选) deep, 写回 metadata.

    Returns:
        (analysis_dict, consumed_deep_slot: bool)
    """
    event = {
        "title": event_rec.content,
        "content": event_rec.metadata.get("raw", {}).get("content") or event_rec.content,
        "source": event_rec.metadata.get("source", "?"),
        "score": event_rec.metadata.get("score", 0),
        "tags": event_rec.metadata.get("tags") or [],
        "code": event_rec.code,
    }

    t_res = triage(event)
    logger.info(
        f"[triage #{event_rec.id}] type={t_res['event_type']} "
        f"tradability={t_res['tradability']} needs_deep={t_res['needs_deep']} "
        f"latency={t_res['_latency_s']}s  oneline={t_res['oneline'][:50]}"
    )

    do_deep = should_deep_analyze(t_res) and deep_budget_left > 0
    consumed = False
    d_res = None
    if do_deep:
        d_res = deep_analyze(event, t_res)
        consumed = True
        logger.info(
            f"[deep   #{event_rec.id}] targets={len(d_res['targets'])} "
            f"priced_in={d_res['already_priced_in']} latency={d_res['_latency_s']}s"
        )

    analysis = {
        "ts_analyzed": int(time.time()),
        "triage": {
            "event_type": t_res["event_type"],
            "entities": t_res["entities"],
            "tradability": t_res["tradability"],
            "novelty": t_res["novelty"],
            "needs_deep": t_res["needs_deep"],
            "oneline": t_res["oneline"],
            "backend": t_res["_backend"],
            "model": t_res["_model"],
            "latency_s": t_res["_latency_s"],
        },
    }
    if d_res:
        analysis["deep"] = {
            "thesis": d_res["thesis"],
            "targets": d_res["targets"],
            "already_priced_in": d_res["already_priced_in"],
            "evidence": d_res["evidence"],
            "supply_chain": d_res["supply_chain"],
            "risk_flags": d_res["risk_flags"],
            "backend": d_res["_backend"],
            "model": d_res["_model"],
            "latency_s": d_res["_latency_s"],
        }

    new_meta = dict(event_rec.metadata)
    new_meta["analysis"] = analysis
    store.update_memory_metadata(event_rec.id, new_meta)
    return analysis, consumed


def run_round(store: MemoryStore, limit: int, deep_budget: int,
              since_hours: int) -> dict:
    """跑一轮: 拉待分析事件, 逐条分析."""
    since_ts = int(time.time()) - since_hours * 3600
    events = store.query_radar_events(
        since_ts=since_ts, needs_analysis=True, limit=limit,
    )
    if not events:
        return {"processed": 0, "deep": 0}

    logger.info(f"本轮待分析 {len(events)} 条, deep 预算 {deep_budget}")
    processed = 0
    deep_used = 0
    for rec in events:
        if _STOP:
            break
        try:
            _, consumed = analyze_one(store, rec, deep_budget - deep_used)
            processed += 1
            if consumed:
                deep_used += 1
        except Exception as e:
            logger.exception(f"分析事件 id={rec.id} 失败: {e}")

    return {"processed": processed, "deep": deep_used}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="跑一轮就退")
    ap.add_argument("--interval", type=int, default=30, help="循环间隔秒")
    ap.add_argument("--limit", type=int, default=20, help="每轮处理上限")
    ap.add_argument("--deep-budget", type=int, default=5, help="每轮 deep 上限")
    ap.add_argument("--since-hours", type=int, default=24, help="只处理最近 N 小时事件")
    args = ap.parse_args()

    logger.info(
        f"radar_worker 启动 | once={args.once} interval={args.interval}s "
        f"limit={args.limit} deep_budget={args.deep_budget}"
    )
    store = MemoryStore()

    # 预热 market_context 缓存
    mc._ensure_loaded()

    rounds = 0
    t_start = time.time()
    total_processed = 0
    total_deep = 0

    while not _STOP:
        t0 = time.time()
        try:
            stats = run_round(
                store, args.limit, args.deep_budget, args.since_hours,
            )
        except Exception as e:
            logger.exception(f"run_round 异常: {e}")
            stats = {"processed": 0, "deep": 0}

        rounds += 1
        total_processed += stats["processed"]
        total_deep += stats["deep"]
        dt = time.time() - t0
        logger.info(
            f"round {rounds} done: processed={stats['processed']} "
            f"deep={stats['deep']} in {dt:.1f}s"
        )

        if args.once or _STOP:
            break

        # 睡眠到下一轮, 支持中途退出
        sleep_left = max(0, args.interval - dt)
        while sleep_left > 0 and not _STOP:
            step = min(1.0, sleep_left)
            time.sleep(step)
            sleep_left -= step

    uptime = time.time() - t_start
    logger.info(
        f"radar_worker 退出 | rounds={rounds} total_processed={total_processed} "
        f"total_deep={total_deep} uptime={uptime:.1f}s"
    )


if __name__ == "__main__":
    main()
