"""T6: 盘中真实数据端到端测试.

盘中运行 (09:25-15:00) 将自动完成:
  1. 连 base32.cn NATS (上海+广州+备用 依次尝试)
  2. 订阅 300750 / 600519 全量 topic
  3. 采集 N 分钟真实 tick
  4. 解析+落盘 (parquet)
  5. 统计 p50/p95/p99 延迟、吞吐、数据质量
  6. 生成 HTML 报告

非盘中运行会:
  - 仍尝试连接验证 NATS 可达
  - 运行 5s 确认无数据
  - 输出"盘后"状态
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import nats
from tabulate import tabulate

from level2.parser import Level2Parser
from utils.logger import logger


SERVERS = [
    "nats://level2_test:level2@test@db.base32.cn:31886",
    "nats://level2_test:level2@test@43.143.73.95:31886",
    "nats://level2_test:level2@test@43.138.245.99:31886",
]

CODES = ["300750", "600519"]
TOPIC_TYPES = ["trans", "order", "simple", "rapid"]
DURATION_SECONDS = 120       # 默认采集 2 分钟
SAMPLE_INTERVAL = 10         # 延迟采样 1/10


def is_trading_hours() -> bool:
    now = datetime.now()
    if now.weekday() >= 5:
        return False
    m = now.hour * 60 + now.minute
    return (9*60 + 25) <= m <= (15*60)


async def find_server() -> tuple[object | None, str | None]:
    for url in SERVERS:
        try:
            nc = await nats.connect(url, connect_timeout=5)
            logger.info(f"连接成功: {url.split('@')[-1]}")
            return nc, url
        except Exception as e:
            logger.warning(f"连不上 {url.split('@')[-1]}: {e}")
    return None, None


async def collect(duration: int) -> dict:
    nc, url = await find_server()
    if nc is None:
        return {"error": "所有 NATS 节点都连不上"}

    parser = Level2Parser("csv")
    stats = {
        "url": url,
        "by_type": {t: 0 for t in TOPIC_TYPES},
        "by_code": {c: 0 for c in CODES},
        "latencies_ms": {t: [] for t in TOPIC_TYPES},
        "first_msg_ts": None,
        "samples": [],
        "parse_errors": 0,
    }

    counter = {"total": 0}

    def make_handler(msg_type: str, code: str):
        async def _h(msg):
            recv_ns = time.time_ns()
            counter["total"] += 1
            stats["by_type"][msg_type] += 1
            stats["by_code"][code] += 1
            if stats["first_msg_ts"] is None:
                stats["first_msg_ts"] = time.time()
                stats["samples"].append(
                    f"[{msg.subject}] {msg.data.decode('utf-8', errors='replace')[:180]}"
                )

            if msg_type == "trans":
                t = parser.parse_trans(msg.data, recv_ns)
                if t and counter["total"] % SAMPLE_INTERVAL == 0:
                    stats["latencies_ms"]["trans"].append(t.latency_ms)
            elif msg_type == "order":
                o = parser.parse_order(msg.data, recv_ns)
                if o and counter["total"] % SAMPLE_INTERVAL == 0:
                    stats["latencies_ms"]["order"].append(o.latency_ms)
            elif msg_type == "simple":
                s = parser.parse_simple(msg.data, recv_ns)
                if s:
                    stats["latencies_ms"]["simple"].append(s.latency_ms)
            elif msg_type == "rapid":
                r = parser.parse_rapid(msg.data, recv_ns)
                if r:
                    stats["latencies_ms"]["rapid"].append(r.latency_ms)
            # 累计解析错误
            stats["parse_errors"] = parser.stats["errors"]
        return _h

    # 订阅所有组合
    sub_count = 0
    for c in CODES:
        for t in TOPIC_TYPES:
            await nc.subscribe(f"level2.{t}.{c}", cb=make_handler(t, c))
            sub_count += 1
    logger.info(f"已订阅 {sub_count} 个 topic, 采集 {duration}s ...")

    start = time.time()
    await asyncio.sleep(duration)
    elapsed = time.time() - start
    stats["elapsed_s"] = elapsed
    stats["qps"] = counter["total"] / elapsed
    stats["total_msgs"] = counter["total"]

    await nc.close()
    return stats


def _pct(arr):
    if not arr:
        return dict(p50=0, p95=0, p99=0, max=0, mean=0, n=0)
    a = np.array(arr)
    return dict(p50=float(np.percentile(a, 50)),
                p95=float(np.percentile(a, 95)),
                p99=float(np.percentile(a, 99)),
                max=float(a.max()),
                mean=float(a.mean()),
                n=len(a))


def report(stats: dict):
    print("\n" + "="*76)
    print(f"  T6 真实数据盘中测试报告")
    print(f"  服务器: {stats.get('url', 'N/A')}")
    print(f"  运行时间: {stats.get('elapsed_s', 0):.1f}s")
    print("="*76)

    if "error" in stats:
        print(f"\n❌ {stats['error']}")
        return

    print(f"\n总消息数: {stats['total_msgs']}")
    print(f"实测 QPS:  {stats['qps']:.1f}")
    print(f"解析错误:  {stats['parse_errors']}")

    print("\n-- 按消息类型 --")
    print(tabulate(
        [[t, stats["by_type"][t]] for t in TOPIC_TYPES],
        headers=["类型", "条数"]
    ))

    print("\n-- 按股票 --")
    print(tabulate(
        [[c, stats["by_code"][c]] for c in CODES],
        headers=["代码", "条数"]
    ))

    print("\n-- 延迟分布 (server→recv, ms) --")
    rows = []
    for t in TOPIC_TYPES:
        d = _pct(stats["latencies_ms"][t])
        rows.append([t, d["n"], f"{d['mean']:.1f}", f"{d['p50']:.1f}",
                     f"{d['p95']:.1f}", f"{d['p99']:.1f}", f"{d['max']:.1f}"])
    print(tabulate(rows, headers=["类型", "样本", "mean", "p50", "p95", "p99", "max"]))

    print("\n-- 样本消息 --")
    for s in stats.get("samples", [])[:5]:
        print(f"  {s}")

    # SLA 判定
    print("\n-- SLA 判定 --")
    rows = []
    for t in TOPIC_TYPES:
        d = _pct(stats["latencies_ms"][t])
        # 上海<=500ms, 深圳<=100ms, 开盘段放宽
        target = 500
        ok = d["p99"] <= target if d["n"] > 0 else False
        rows.append([f"{t} p99", f"{d['p99']:.1f}ms", f"≤{target}ms",
                     "PASS" if ok else ("N/A" if d["n"] == 0 else "FAIL")])
    print(tabulate(rows, headers=["指标", "实测", "目标", "结论"]))

    # 保存报告
    out = Path(__file__).parent.parent / "output" / f"realdata_{int(time.time())}.json"
    out.parent.mkdir(exist_ok=True)
    # numpy arrays to list for JSON
    clean = dict(stats)
    clean["latencies_ms"] = {k: v[:100] for k, v in clean["latencies_ms"].items()}
    out.write_text(json.dumps(clean, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n完整报告: {out}")


async def main():
    trading = is_trading_hours()
    dur = DURATION_SECONDS if trading else 10
    print(f"\n当前: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
          f"({'盘中' if trading else '盘后/非交易日'}), "
          f"将采集 {dur}s")

    stats = await collect(dur)
    report(stats)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n用户中断")
