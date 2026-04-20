"""T3-mock: 用 Simulator 代替 NATS 的延迟/吞吐压测.

无需真实 NATS 环境, 纯进程内跑. 用于:
  - 验证压测框架本身正确
  - 基准测试解析+风控+聚合链路的CPU开销
  - 拿到真实凭证前, 先验证代码无 bug
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from tabulate import tabulate

from level2 import Level2Simulator


async def run(qps: int, duration: int):
    sim = Level2Simulator(codes=["300750", "600519"], qps=qps, inject_error_rate=0.001)
    samples = {"trade": [], "order": [], "book": []}

    def rec_trade(t):
        samples["trade"].append(t.latency_ms)

    def rec_order(o):
        samples["order"].append(o.latency_ms)

    def rec_book(b):
        samples["book"].append(b.latency_ms)

    sim.on_trade(rec_trade)
    sim.on_order(rec_order)
    sim.on_book(rec_book)

    await sim.run(duration_seconds=duration)
    return sim, samples


def _pct(arr):
    if not arr:
        return {"p50": 0, "p95": 0, "p99": 0, "max": 0, "n": 0}
    a = np.array(arr)
    return {"p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)), "max": float(a.max()), "n": len(a)}


def main():
    duration = 10    # mock 模式跑 10s 足够
    for qps in [1000, 5000, 10000]:
        print(f"\n{'='*60}")
        print(f"  Mock 压测: QPS={qps}, duration={duration}s")
        print('='*60)
        sim, samples = asyncio.run(run(qps, duration))

        print(f"\n实际吞吐: {sim.stats.qps:.0f} msg/s  (目标 {qps})")
        print(f"解析错误 (注入): {sim.stats.errors_injected}")
        print()

        rows = []
        for n in ["trade", "order", "book"]:
            d = _pct(samples[n])
            rows.append([n, d["n"], f"{d['p50']:.3f}", f"{d['p95']:.3f}",
                         f"{d['p99']:.3f}", f"{d['max']:.3f}"])
        print(tabulate(rows, headers=["类型", "样本", "p50 ms", "p95 ms", "p99 ms", "max ms"]))


if __name__ == "__main__":
    main()
