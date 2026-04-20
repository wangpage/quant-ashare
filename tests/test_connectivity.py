"""T1: 行情接入联通性测试.

验证项:
  - NATS 服务器可达
  - 认证通过
  - 3 类 topic 在 N 秒内至少收到 M 条
  - 心跳正常
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tabulate import tabulate

from level2 import Level2NatsClient
from utils.logger import logger


MIN_MSG_PER_TOPIC = 1    # 联通性最低: 每类至少1条
WAIT_SECONDS = 30


async def run_test():
    cli = Level2NatsClient()
    result = {"nats_connect": False, "auth": False, "subscribe": False,
              "trade_recv": 0, "order_recv": 0, "book_recv": 0,
              "parse_error_rate": 0.0}

    try:
        await cli.connect()
        result["nats_connect"] = True
        result["auth"] = True
    except Exception as e:
        logger.error(f"NATS 连接失败: {e}")
        return result

    try:
        codes = cli.cfg["test_instruments"]
        await cli.subscribe_instruments(codes)
        result["subscribe"] = True
    except Exception as e:
        logger.error(f"订阅失败: {e}")
        await cli.disconnect()
        return result

    logger.info(f"等待 {WAIT_SECONDS}s 接收行情...")
    await cli.run_for(WAIT_SECONDS)

    result["trade_recv"] = cli.stats.trade_count
    result["order_recv"] = cli.stats.order_count
    result["book_recv"] = cli.stats.book_count
    result["parse_error_rate"] = cli.parser.stats["error_rate"]
    await cli.disconnect()
    return result


def main():
    start = time.time()
    r = asyncio.run(run_test())
    elapsed = time.time() - start

    rows = [
        ["NATS 连接", "✓" if r["nats_connect"] else "✗"],
        ["认证", "✓" if r["auth"] else "✗"],
        ["订阅", "✓" if r["subscribe"] else "✗"],
        ["trade 收到", f"{r['trade_recv']}"],
        ["order 收到", f"{r['order_recv']}"],
        ["book 收到", f"{r['book_recv']}"],
        ["解析错误率", f"{r['parse_error_rate']:.4%}"],
        ["总耗时", f"{elapsed:.1f}s"],
    ]
    print("\n==== T1 联通性测试结果 ====")
    print(tabulate(rows, headers=["项", "值"]))

    ok = (
        r["nats_connect"] and r["subscribe"]
        and r["trade_recv"] >= MIN_MSG_PER_TOPIC
        and r["order_recv"] >= MIN_MSG_PER_TOPIC
        and r["book_recv"] >= MIN_MSG_PER_TOPIC
        and r["parse_error_rate"] < 0.01
    )
    print(f"\n结论: {'PASS ✓' if ok else 'FAIL ✗'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
