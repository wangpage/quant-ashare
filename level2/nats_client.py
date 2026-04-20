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
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import yaml

from utils.config import PROJECT_ROOT, load_config
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
    out_of_order: int = 0           # 序号或时间戳逆序的消息
    clock_drift_ms_max: float = 0.0  # 服务器-本地时钟最大漂移
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


class _SequenceGuard:
    """按 (topic_prefix, code) 维度检查消息序号 + 时间戳单调性.

    A 股 Level2 因 UDP 重传 / 多路复用偶有乱序, 微结构因子 (VPIN / OIR /
    Kyle's λ) 对时序敏感, 乱序不检测会让信号含噪.

    策略: 维护每路最后的 (main_seq, sub_seq, server_time_ms), 新消息三者
    任意维度回退即视为乱序 → 计数 + 告警, 由调用方决定是否丢弃.
    """

    def __init__(self, warn_every: int = 500):
        self._last: dict[tuple, tuple[int, int, int]] = {}
        self._ooo_count: dict[tuple, int] = defaultdict(int)
        self._warn_every = warn_every

    def check(self, stream: str, code: str,
              main_seq: int, sub_seq: int, server_time_ms: int) -> bool:
        """返回 True = 正常有序; False = 乱序."""
        key = (stream, code)
        cur = (int(main_seq or 0), int(sub_seq or 0), int(server_time_ms or 0))
        last = self._last.get(key)
        self._last[key] = cur
        if last is None:
            return True
        # 序号或时间戳任一维度回退 = 乱序
        if cur < last:
            self._ooo_count[key] += 1
            n = self._ooo_count[key]
            if n == 1 or n % self._warn_every == 0:
                logger.warning(
                    f"乱序 {stream}/{code}: last={last} cur={cur} (累计 {n})"
                )
            return False
        return True


class Level2NatsClient:
    def __init__(self, config_path: str | Path | None = None,
                 drop_out_of_order: bool = False):
        path = Path(config_path) if config_path else PROJECT_ROOT / "config" / "level2.yaml"
        # 用 load_config 以展开 ${ENV:-default}, 避免凭证硬编码
        self.cfg = load_config(path)

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
        self._seq_guard = _SequenceGuard()
        self._drop_ooo = drop_out_of_order
        # 时钟漂移滚动样本 (服务器 ms - 本地 ms), 用于早期识别时间源异常
        self._drift_samples: deque = deque(maxlen=256)
        self._last_drift_warn_ts = 0.0

    # ---------- 回调 ----------
    def on_trade(self, fn):  self._cb["trans"] = fn
    def on_order(self, fn):  self._cb["order"] = fn
    def on_rapid(self, fn):  self._cb["rapid"] = fn
    def on_simple(self, fn): self._cb["simple"] = fn
    def on_depth(self, fn):  self._cb["depth"] = fn
    def on_book(self, fn):   self._cb["simple"] = fn          # 旧 API

    # ---------- 连接 ----------
    async def connect(self, *, max_retries: int = 6,
                      base_backoff: float = 1.0, max_backoff: float = 30.0):
        """带指数退避的连接. 断线由 nats 库自身的重连接管, 这里只管首次握手.

        退避序列: 1s, 2s, 4s, 8s, 16s, 30s (带 ±25% jitter 防惊群).
        """
        import nats

        conn = self.cfg["connection"]
        active = conn["active"]
        server = conn["servers"][active]
        urls = [server["host"], server.get("backup")]
        urls = [u for u in urls if u and "TODO" not in u]
        auth = conn["auth"]

        if not auth.get("user") or not auth.get("password"):
            raise RuntimeError(
                "Level2 凭证为空: 请设置环境变量 LEVEL2_USER / LEVEL2_PASSWORD "
                "或使用测试账号默认值"
            )

        # 文档示例格式: nats://user:pass@host:port
        urls_with_auth = []
        for u in urls:
            if "@" in u.split("//")[-1]:
                urls_with_auth.append(u)
            else:
                urls_with_auth.append(
                    u.replace("nats://", f"nats://{auth['user']}:{auth['password']}@")
                )

        last_err: Exception | None = None
        for attempt in range(max_retries):
            try:
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
                logger.info(
                    f"NATS 已连接: {active} / user={auth['user']} "
                    f"(尝试 {attempt + 1}/{max_retries})"
                )
                return
            except Exception as e:
                last_err = e
                if attempt == max_retries - 1:
                    break
                wait = min(base_backoff * (2 ** attempt), max_backoff)
                wait *= random.uniform(0.75, 1.25)  # jitter
                logger.warning(
                    f"NATS 连接失败 ({attempt + 1}/{max_retries}): {e}; "
                    f"{wait:.1f}s 后重试"
                )
                await asyncio.sleep(wait)
        raise RuntimeError(f"NATS 连接重试耗尽: {last_err}")

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

    def _track_drift(self, server_ms: int, recv_ns: int):
        """跟踪服务器-本地时钟漂移 (差值的中位数 vs 瞬时差).

        单条消息差值 = 网络延迟 + 时钟漂移, 不可分离; 但 rolling 中位数
        是网络延迟的一阶近似, 极端值 (> 5s) 多来自本地时钟错乱 (如容器
        未同步 NTP). A 股微结构因子对相对时序敏感而非绝对时序, 所以我们
        只监控、不尝试修正.
        """
        drift = recv_ns / 1e6 - server_ms
        self._drift_samples.append(drift)
        if abs(drift) > self.stats.clock_drift_ms_max:
            self.stats.clock_drift_ms_max = abs(drift)
        # 每 30s 检查一次是否需要告警
        now = time.time()
        if now - self._last_drift_warn_ts < 30:
            return
        if len(self._drift_samples) < 32:
            return
        self._last_drift_warn_ts = now
        sorted_samples = sorted(self._drift_samples)
        median = sorted_samples[len(sorted_samples) // 2]
        # 5 秒以上中位漂移几乎肯定是时钟错乱, 非网络延迟
        if abs(median) > 5000:
            logger.error(
                f"时钟漂移异常: median={median:.0f}ms (>5s), "
                f"检查本地 NTP 同步. 微结构因子将受影响"
            )

    def _record(self, stream: str, code: str, main_seq: int, sub_seq: int,
                server_ms: int, recv_ns: int) -> bool:
        """统一入口: 更新延迟 / 时钟 / 乱序统计. 返回是否应处理该消息."""
        self._track_drift(server_ms, recv_ns)
        ordered = self._seq_guard.check(stream, code, main_seq, sub_seq, server_ms)
        if not ordered:
            self.stats.out_of_order += 1
            if self._drop_ooo:
                return False
        return True

    async def _h_trans(self, msg):
        recv = time.time_ns()
        t = self.parser.parse_trans(msg.data, recv)
        if not t:
            self.stats.parse_errors += 1
            return
        if not self._record("trans", t.code, t.main_seq, t.sub_seq,
                            t.server_time_ms, recv):
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
        if not self._record("order", o.code,
                            getattr(o, "main_seq", 0), getattr(o, "sub_seq", 0),
                            getattr(o, "server_time_ms", 0), recv):
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
        if not self._record("rapid", r.code, 0, 0,
                            getattr(r, "server_time_ms", 0), recv):
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
        if not self._record("simple", s.code, 0, 0,
                            getattr(s, "server_time_ms", 0), recv):
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
        if not self._record("depth", d.code, 0, 0,
                            getattr(d, "server_time_ms", 0), recv):
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
