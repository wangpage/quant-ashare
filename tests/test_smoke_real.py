"""T1-smoke: 最小可行连通性测试 - 直接仿照文档示例代码.

成功条件:
  1. TCP 能连到 nats://level2_test:level2@test@db.base32.cn:31886
  2. 10s 内至少收到 1 条消息
"""
from __future__ import annotations

import asyncio
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import nats
from nats.errors import ConnectionClosedError, TimeoutError, NoServersError


SERVERS = [
    "nats://level2_test:level2@test@db.base32.cn:31886",       # 文档推荐
    "nats://level2_test:level2@test@43.143.73.95:31886",       # 上海直连
    "nats://level2_test:level2@test@43.138.245.99:31886",      # 广州直连
]


async def try_server(url: str, wait_seconds: int = 10) -> dict:
    result = {"url": url, "connected": False, "msg_count": 0, "first_msg": None,
              "error": None, "topics_with_data": set(), "first_msg_latency_ms": None}
    try:
        nc = await nats.connect(url, connect_timeout=5)
    except Exception as e:
        result["error"] = f"连接失败: {e}"
        return result

    result["connected"] = True
    print(f"  ✓ 连接成功: {url}")

    async def handler(msg):
        recv_ns = time.time_ns()
        result["msg_count"] += 1
        if result["first_msg"] is None:
            data = msg.data.decode("utf-8", errors="replace")
            result["first_msg"] = f"[{msg.subject}]: {data[:200]}"
            parts = data.split(",")
            if len(parts) >= 4:
                try:
                    server_ms = int(parts[3])
                    if server_ms > 1_000_000_000_000:
                        result["first_msg_latency_ms"] = recv_ns / 1e6 - server_ms
                except Exception:
                    pass
        result["topics_with_data"].add(msg.subject)

    codes = ["300750", "600519"]
    topics = ["trans", "order", "simple", "rapid"]
    for c in codes:
        for t in topics:
            await nc.subscribe(f"level2.{t}.{c}", cb=handler)
    await nc.subscribe("level1.market.300750", cb=handler)
    await nc.subscribe("level1.market.600519", cb=handler)

    print(f"  已订阅 {len(codes) * len(topics) + 2} 个 topic, 等待 {wait_seconds}s ...")
    await asyncio.sleep(wait_seconds)
    await nc.close()

    result["topics_with_data"] = list(result["topics_with_data"])
    return result


async def main():
    print("="*72)
    print("  Level2 真实服务器烟雾测试")
    print("  账号: level2_test / level2@test  (仅支持 300750 / 600519)")
    print("  时间:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*72)

    now = datetime.now()
    is_trading_hours = (
        now.weekday() < 5 and
        (9*60 + 15 <= now.hour*60 + now.minute <= 15*60 + 0)
    )
    print(f"\n当前{'盘中' if is_trading_hours else '盘后/非交易日'} "
          f"({'应有实时推送' if is_trading_hours else '可能无实时数据'})")

    for url in SERVERS:
        print(f"\n→ 尝试: {url.split('@')[-1]}")
        r = await try_server(url, wait_seconds=10)
        if not r["connected"]:
            print(f"  ✗ {r['error']}")
            continue

        print(f"  收到消息数: {r['msg_count']}")
        print(f"  活跃 topic: {len(r['topics_with_data'])}")
        if r["topics_with_data"]:
            for t in r["topics_with_data"][:6]:
                print(f"    • {t}")
        if r["first_msg"]:
            print(f"  首条消息:")
            print(f"    {r['first_msg']}")
        if r["first_msg_latency_ms"] is not None:
            print(f"  首条消息延迟: {r['first_msg_latency_ms']:.1f} ms")

        if r["msg_count"] > 0:
            print(f"\n✅ 真实行情流通！使用此 URL 继续完整测试.")
            return 0

    print("\n⚠ 所有服务器都连上了但未收到消息, 可能原因:")
    print("  1. 盘后时段 (建议 09:15-15:00 再测)")
    print("  2. 测试账号标的受限 (仅 300750/600519)")
    print("  3. 账号权限未开通逐笔 (需申请)")
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
