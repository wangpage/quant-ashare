"""查看 Radar 上行的事件流, 用于 Week 1 联通测试.

用法:
  python scripts/inspect_radar_events.py                  # 今日全部
  python scripts/inspect_radar_events.py --hours 2        # 最近 2h
  python scripts/inspect_radar_events.py --source cls     # 仅财联社
  python scripts/inspect_radar_events.py --min-score 60   # 仅高分事件
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory.storage import MemoryStore


def fmt_ts(ts: int) -> str:
    return time.strftime("%H:%M:%S", time.localtime(ts))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hours", type=int, default=24)
    p.add_argument("--source", type=str, default=None)
    p.add_argument("--min-score", type=int, default=0)
    p.add_argument("--limit", type=int, default=200)
    args = p.parse_args()

    store = MemoryStore()
    since = int(time.time()) - args.hours * 3600

    print(f"=== Radar Events (last {args.hours}h) ===\n")
    stats = store.radar_stats(since)
    print(f"总数: {stats['total']}  高分(>=60): {stats['high_score']}")
    print("按源:")
    for src, n in sorted(stats["by_source"].items(), key=lambda x: -x[1]):
        print(f"  {src:20s} {n}")
    print()

    events = store.query_radar_events(
        since_ts=since, source=args.source,
        min_score=args.min_score, limit=args.limit,
    )
    if not events:
        print("无匹配事件.")
        return

    print(f"明细 (显示 {len(events)} 条):\n")
    for e in events:
        src = e.metadata.get("source", "?")
        score = e.metadata.get("score", 0)
        tags = ",".join(e.metadata.get("tags") or [])
        code = e.code or "--"
        print(f"[{fmt_ts(e.ts)}] score={score:3d} {src:18s} {code:8s} "
              f"{e.content[:60]}  #{tags}")


if __name__ == "__main__":
    main()
