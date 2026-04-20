"""base32.cn Level2 NATS 客户端.

依据《沪深Level2高级行情接口文档》:
  - URL: nats://user:pass@host:31886
  - 订阅: level2.trans/order/rapid/simple/depth.{code}
         level1.market/flow.{code}
  - 消息格式: CSV
  - 限制: 单IP <=50连接, 单账号 <=100订阅
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import yaml

from utils.config import PROJECT_ROOT
from utils.logger import logger

from .parser import Level2Parser
from .buffer import RingBuffer


@dataclass
class Level2Stats:
    trans: int = 0
    order: int = 0
    rapid: int = 0
    simple: int = 0
    depth: int = 0
    l1_market: int = 0
    l1_flow: int = 0
    parse_errors: int = 0
    total_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    start_ts: float = 0.0

    @property
    def total(self) -> int:
        return (self.trans + self.order + self.rapid + self.simple
                + self.depth + self.l1_market + self.l1_flow)

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_ts

    @property
    def qps(self) -> float:
        return self.total / max(self.elapsed, 1e-9)

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / max(self.total, 1)


class Level2NatsClient:
    def __init__(self, config_path: str | Path | None = None):
        path = Path(config_path) if config_path else PROJECT_ROOT / "config" / "level2.yaml"
        with open(path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        self.parser = Level2Parser("csv")
        self.trans_buf = RingBuffer(200_000)
        self.order_buf = RingBuffer(500_000)
        self.rapid_buf = RingBuffer(50_000)
        self.simple_buf = RingBuffer(100_000)
        self.depth_buf = RingBuffer(50_000)

        self.stats = Level2Stats(start_ts=time.time())
        self._nc = None
        self._cb: dict[str, Callable | None] = {
            "trans": None, "order": None, "rapid": None,
            "simple": None, "depth": None,
        }

    # ---------- 回调 ----------
    def on_trade(self, fn):  self._cb["trans"] = fn
    def on_order(self, fn):  self._cb["order"] = fn
    def on_rapid(self, fn):  self._cb["rapid"] = fn
    def on_simple(self, fn): self._cb["simple"] = fn
    def on_depth(self, fn):  self._cb["depth"] = fn
    def on_book(self, fn):   self._cb["simple"] = fn          # 旧 API

    # ---------- 连接 ----------
    async def connect(self):
        import nats

        conn = self.cfg["connection"]
        active = conn["active"]
        server = conn["servers"][active]
        urls = [server["host"], server.get("backup")]
        urls = [u for u in urls if u and "TODO" not in u]
        auth = conn["auth"]

        # 文档示例格式: nats://user:pass@host:port
        urls_with_auth = []
        for u in urls:
            # 如果 URL 已含 @, 不重复注入
            if "@" in u.split("//")[-1]:
                urls_with_auth.append(u)
            else:
                urls_with_auth.append(
                    u.replace("nats://", f"nats://{auth['user']}:{auth['password']}@")
                )

        self._nc = await nats.connect(
            servers=urls_with_auth,
            connect_timeout=conn["timeout_seconds"],
            ping_interval=conn["ping_interval"],
            max_reconnect_attempts=conn["max_reconnect"],
            reconnect_time_wait=conn["reconnect_wait"],
            error_cb=self._on_err,
            disconnected_cb=self._on_disc,
            reconnected_cb=self._on_reconn,
            closed_cb=self._on_close,
        )
        logger.info(f"NATS 已连接: {active} / user={auth['user']}")

    async def _on_err(self, e):      logger.error(f"NATS err: {e}")
    async def _on_disc(self):        logger.warning("NATS 断开")
    async def _on_reconn(self):      logger.info("NATS 重连")
    async def _on_close(self):       logger.warning("NATS 关闭")

    async def disconnect(self):
        if self._nc:
            await self._nc.drain()
            self._nc = None

    # ---------- 订阅 ----------
    async def subscribe_instruments(self, codes: list[str]):
        t = self.cfg["topics"]
        en = self.cfg["enable_types"]
        total_subs = 0
        for code in codes:
            if en.get("trans"):
                await self._nc.subscribe(t["trans"].format(code=code), cb=self._h_trans)
                total_subs += 1
            if en.get("order"):
                await self._nc.subscribe(t["order"].format(code=code), cb=self._h_order)
                total_subs += 1
            if en.get("rapid"):
                await self._nc.subscribe(t["rapid"].format(code=code), cb=self._h_rapid)
                total_subs += 10    # 文档: 1合成订单簿=10个逐笔量
            if en.get("simple"):
                await self._nc.subscribe(t["simple"].format(code=code), cb=self._h_simple)
                total_subs += 1
            if en.get("depth"):
                await self._nc.subscribe(t["depth"].format(code=code), cb=self._h_depth)
                total_subs += 1
        max_sub = self.cfg["limits"]["max_subscribe"]
        logger.info(f"订阅完成: {len(codes)}只 x 数据类型, 订阅计数={total_subs}/{max_sub}")
        if total_subs > max_sub:
            logger.warning(f"⚠ 订阅数 {total_subs} 超过限制 {max_sub}")

    # ---------- 消息处理 ----------
    def _update_latency(self, lat: float):
        self.stats.total_latency_ms += abs(lat)
        self.stats.max_latency_ms = max(self.stats.max_latency_ms, abs(lat))

    async def _h_trans(self, msg):
        recv = time.time_ns()
        t = self.parser.parse_trans(msg.data, recv)
        if not t:
            self.stats.parse_errors += 1
            return
        self.stats.trans += 1
        self._update_latency(t.latency_ms)
        self.trans_buf.push(t)
        if self._cb["trans"]:
            try: self._cb["trans"](t)
            except Exception as e: logger.warning(f"trans cb: {e}")

    async def _h_order(self, msg):
        recv = time.time_ns()
        o = self.parser.parse_order(msg.data, recv)
        if not o:
            self.stats.parse_errors += 1
            return
        self.stats.order += 1
        self._update_latency(o.latency_ms)
        self.order_buf.push(o)
        if self._cb["order"]:
            try: self._cb["order"](o)
            except Exception as e: logger.warning(f"order cb: {e}")

    async def _h_rapid(self, msg):
        recv = time.time_ns()
        r = self.parser.parse_rapid(msg.data, recv)
        if not r:
            self.stats.parse_errors += 1
            return
        self.stats.rapid += 1
        self._update_latency(r.latency_ms)
        self.rapid_buf.push(r)
        if self._cb["rapid"]:
            try: self._cb["rapid"](r)
            except Exception as e: logger.warning(f"rapid cb: {e}")

    async def _h_simple(self, msg):
        recv = time.time_ns()
        s = self.parser.parse_simple(msg.data, recv)
        if not s:
            self.stats.parse_errors += 1
            return
        self.stats.simple += 1
        self._update_latency(s.latency_ms)
        self.simple_buf.push(s)
        if self._cb["simple"]:
            try: self._cb["simple"](s)
            except Exception as e: logger.warning(f"simple cb: {e}")

    async def _h_depth(self, msg):
        recv = time.time_ns()
        d = self.parser.parse_depth(msg.data, recv)
        if not d:
            self.stats.parse_errors += 1
            return
        self.stats.depth += 1
        self._update_latency(d.latency_ms)
        self.depth_buf.push(d)
        if self._cb["depth"]:
            try: self._cb["depth"](d)
            except Exception as e: logger.warning(f"depth cb: {e}")

    async def run_for(self, seconds: int):
        await asyncio.sleep(seconds)


async def _demo():
    cli = Level2NatsClient()
    await cli.connect()
    await cli.subscribe_instruments(cli.cfg["test_instruments"])
    await cli.run_for(30)
    s = cli.stats
    logger.info(f"stats: trans={s.trans} order={s.order} simple={s.simple} "
                f"rapid={s.rapid} depth={s.depth} "
                f"qps={s.qps:.1f} avg_lat={s.avg_latency_ms:.2f}ms "
                f"max_lat={s.max_latency_ms:.2f}ms")
    await cli.disconnect()


if __name__ == "__main__":
    asyncio.run(_demo())
