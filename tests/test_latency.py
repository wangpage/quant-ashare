"""T3: 延迟与吞吐量压测.

指标:
  - p50 / p95 / p99 端到端延迟 (exchange_ts -> recv_ts)
  - QPS (峰值与均值)
  - 丢包率 (RingBuffer 满溢)
  - 解析错误率
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from tabulate import tabulate

from level2 import Level2NatsClient
from utils.logger import logger


async def run():
    cli = Level2NatsClient()
    await cli.connect()
    codes = cli.cfg["test_instruments"]
    await cli.subscribe_instruments(codes)

    duration = cli.cfg["stress_test"]["duration_seconds"]
    warmup = cli.cfg["stress_test"]["warmup_seconds"]

    # 延迟样本池 (采样 1/10, 防 OOM)
    samples = {"trade": [], "order": [], "book": []}

    def rec_trade(t):
        if cli.stats.trade_count % 10 == 0:
            samples["trade"].append(t.latency_ms)

    def rec_order(o):
        if cli.stats.order_count % 10 == 0:
            samples["order"].append(o.latency_ms)

    def rec_book(b):
        samples["book"].append(b.latency_ms)

    cli.on_trade(rec_trade)
    cli.on_order(rec_order)
    cli.on_book(rec_book)

    logger.info(f"预热 {warmup}s ...")
    await asyncio.sleep(warmup)
    cli.stats.msg_count = 0
    cli.stats.trade_count = 0
    cli.stats.order_count = 0
    cli.stats.book_count = 0
    cli.stats.total_latency_ms = 0.0
    cli.stats.max_latency_ms = 0.0
    cli.stats.start_ts = time.time()
    samples = {"trade": [], "order": [], "book": []}

    logger.info(f"开始压测 {duration}s ...")
    await asyncio.sleep(duration)

    stats = cli.stats
    parse_stats = cli.parser.stats
    buf_stats = {
        "trade_buf": cli.trade_buf.stats,
        "order_buf": cli.order_buf.stats,
        "book_buf":  cli.book_buf.stats,
    }

    await cli.disconnect()
    return stats, parse_stats, buf_stats, samples


def _percentiles(arr: list[float]) -> dict:
    if not arr:
        return {"p50": 0, "p95": 0, "p99": 0, "max": 0, "n": 0}
    a = np.array(arr)
    return {
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": float(np.max(a)),
        "n": len(a),
    }


def main():
    stats, parse, bufs, samples = asyncio.run(run())

    tp_lat = _percentiles(samples["trade"])
    od_lat = _percentiles(samples["order"])
    ob_lat = _percentiles(samples["book"])

    print("\n==== T3 延迟压测 (ms) ====")
    print(tabulate(
        [
            ["trade", tp_lat["n"], f"{tp_lat['p50']:.2f}", f"{tp_lat['p95']:.2f}",
             f"{tp_lat['p99']:.2f}", f"{tp_lat['max']:.2f}"],
            ["order", od_lat["n"], f"{od_lat['p50']:.2f}", f"{od_lat['p95']:.2f}",
             f"{od_lat['p99']:.2f}", f"{od_lat['max']:.2f}"],
            ["book",  ob_lat["n"], f"{ob_lat['p50']:.2f}", f"{ob_lat['p95']:.2f}",
             f"{ob_lat['p99']:.2f}", f"{ob_lat['max']:.2f}"],
        ],
        headers=["类型", "样本数", "p50", "p95", "p99", "max"],
    ))

    print("\n==== T3 吞吐量 ====")
    print(tabulate(
        [
            ["总消息数",   stats.msg_count],
            ["trade",       stats.trade_count],
            ["order",       stats.order_count],
            ["book",        stats.book_count],
            ["运行时长 (s)", f"{stats.elapsed:.1f}"],
            ["平均 QPS",    f"{stats.qps:.0f}"],
            ["平均延迟",    f"{stats.avg_latency_ms:.2f} ms"],
            ["最大延迟",    f"{stats.max_latency_ms:.2f} ms"],
        ],
        headers=["指标", "值"],
    ))

    print("\n==== T3 RingBuffer ====")
    print(tabulate(
        [
            ["trade_buf", bufs["trade_buf"]["pushed"], bufs["trade_buf"]["dropped"],
             f"{bufs['trade_buf']['drop_rate']:.4%}"],
            ["order_buf", bufs["order_buf"]["pushed"], bufs["order_buf"]["dropped"],
             f"{bufs['order_buf']['drop_rate']:.4%}"],
            ["book_buf",  bufs["book_buf"]["pushed"],  bufs["book_buf"]["dropped"],
             f"{bufs['book_buf']['drop_rate']:.4%}"],
        ],
        headers=["buffer", "pushed", "dropped", "drop_rate"],
    ))

    print("\n==== T3 解析错误 ====")
    print(f"  解析错误率: {parse['error_rate']:.6%}")

    # 判定
    import yaml
    cfg = yaml.safe_load(open(Path(__file__).resolve().parent.parent / "config" / "level2.yaml"))
    latency_sla = cfg["stress_test"]["latency_sla_ms"]
    qps_sla = cfg["stress_test"]["throughput_sla_per_sec"]

    print("\n==== T3 SLA 判定 ====")
    rows = []
    for n, d in [("trade", tp_lat), ("order", od_lat), ("book", ob_lat)]:
        ok = d["p99"] <= latency_sla
        rows.append([f"{n} p99 延迟", f"{d['p99']:.2f}ms", f"≤{latency_sla}ms", "PASS" if ok else "FAIL"])
    qps_ok = stats.qps >= qps_sla * 0.1  # 盘外测试, 10% 即算通过
    rows.append(["平均 QPS", f"{stats.qps:.0f}", f"≥{qps_sla*0.1:.0f} (盘外)",
                 "PASS" if qps_ok else "FAIL"])
    drop_ok = all(b["drop_rate"] < 0.001 for b in bufs.values())
    rows.append(["buffer 丢包", "累计", "<0.1%", "PASS" if drop_ok else "FAIL"])
    err_ok = parse["error_rate"] < 0.001
    rows.append(["解析错误", f"{parse['error_rate']:.4%}", "<0.1%", "PASS" if err_ok else "FAIL"])
    print(tabulate(rows, headers=["指标", "实测", "SLA", "结论"]))


if __name__ == "__main__":
    main()
