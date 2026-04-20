"""base32.cn Level2 行情模拟器 - CSV 格式, 匹配真实规范.

用于无真实环境时跑通端到端测试.
"""
from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from typing import Callable

from .parser import (Level2Parser, TradeTick, OrderTick,
                     RapidBook, SimpleBook, DepthBook, FundFlow)


@dataclass
class InstrumentState:
    code: str
    exchange: str               # XSHG (沪) / XSHE (深)
    price: float
    vol_pct: float = 0.002
    tick_size: float = 0.01
    order_seq: int = 1_000_000
    trade_seq: int = 20_000_000


@dataclass
class SimulatorStats:
    trans: int = 0
    order: int = 0
    rapid: int = 0
    simple: int = 0
    depth: int = 0
    errors_injected: int = 0
    start_ts: float = 0.0

    @property
    def total(self) -> int:
        return self.trans + self.order + self.rapid + self.simple + self.depth

    @property
    def qps(self) -> float:
        e = time.time() - self.start_ts
        return self.total / max(e, 1e-9)


class Level2Simulator:
    def __init__(
        self,
        codes: list[str],
        qps: int = 5000,
        init_prices: dict[str, float] | None = None,
        inject_error_rate: float = 0.001,
    ):
        self.qps = qps
        self.inject_error_rate = inject_error_rate
        default_prices = {"300750": 366.29, "600519": 1476.50}
        self.instruments: dict[str, InstrumentState] = {}
        for c in codes:
            self.instruments[c] = InstrumentState(
                code=c,
                exchange="XSHG" if c.startswith("6") else "XSHE",
                price=(init_prices or default_prices).get(c, 50.0),
            )
        self.parser = Level2Parser("csv")
        self.stats = SimulatorStats()
        self._cb: dict[str, Callable | None] = {
            "trans": None, "order": None, "rapid": None,
            "simple": None, "depth": None,
        }
        self._running = False

    def on_trade(self, fn):  self._cb["trans"] = fn
    def on_order(self, fn):  self._cb["order"] = fn
    def on_rapid(self, fn):  self._cb["rapid"] = fn
    def on_simple(self, fn): self._cb["simple"] = fn
    def on_depth(self, fn):  self._cb["depth"] = fn
    # 向后兼容旧API
    def on_book(self, fn):   self._cb["simple"] = fn

    # ---------- 价格演化 ----------
    def _step(self, inst: InstrumentState):
        shock = random.gauss(0, inst.vol_pct)
        inst.price = max(inst.tick_size, inst.price * (1 + shock))
        inst.price = round(inst.price / inst.tick_size) * inst.tick_size

    # ---------- 真实 CSV 消息生成 ----------
    def _exch_time(self) -> int:
        """HHMMSSsss 9位 时间."""
        t = time.time()
        ms = int((t - int(t)) * 1000)
        lt = time.localtime(t)
        return int(f"{lt.tm_hour:02d}{lt.tm_min:02d}{lt.tm_sec:02d}{ms:03d}")

    def _today(self) -> str:
        return time.strftime("%Y-%m-%d")

    def _make_trans(self, inst: InstrumentState) -> bytes:
        inst.trade_seq += 1
        volume = random.choice([100, 200, 500, 1000, 2000])
        tick_type = "T" if inst.exchange == "XSHG" else "1"
        fields = [
            inst.code, self._today(),
            str(self._exch_time()), str(int(time.time() * 1000)),
            tick_type, f"{inst.price:.3f}", str(volume),
            str(random.randint(1, 2100)), str(inst.trade_seq),
            str(random.randint(1000000, 9999999)),
            str(random.randint(1000000, 9999999)),
        ]
        return ",".join(fields).encode("utf-8")

    def _make_order(self, inst: InstrumentState) -> bytes:
        inst.order_seq += 1
        side = random.choice([1, 2])
        if inst.exchange == "XSHG":
            tick_type = random.choice(["A", "D"])
        else:
            tick_type = random.choice(["0", "1", "2", "3"])
        offset = random.randint(-10, 10) * inst.tick_size
        price = round(inst.price + offset, 3)
        fields = [
            inst.code, self._today(),
            str(self._exch_time()), str(int(time.time() * 1000)),
            tick_type, str(side), f"{price:.3f}",
            str(random.choice([100, 200, 500, 1000])),
            str(random.randint(1, 2100)), str(inst.order_seq),
            str(random.randint(10000000, 99999999)),
        ]
        return ",".join(fields).encode("utf-8")

    def _make_simple(self, inst: InstrumentState) -> bytes:
        ask1 = round(inst.price + inst.tick_size, 3)
        bid1 = round(inst.price - inst.tick_size, 3)
        fields = [
            f"{inst.code}.{inst.exchange}",
            str(self._exch_time()),
            str(int(time.time() * 1000)),
            f"{inst.price * 0.99:.3f}",   # pre_close
            str(random.randint(1_000_000, 10_000_000)),     # total_volume
            f"{random.uniform(1e8, 1e9):.2f}",              # total_amount
            f"{inst.price:.3f}",                            # last
            "0.0000",                                       # iopv
            str(random.randint(1000, 10000)),               # num_trades
            f"{ask1:.3f}", f"{bid1:.3f}",
            str(random.randint(100, 5000)),
            str(random.randint(100, 5000)),
        ]
        return ",".join(fields).encode("utf-8")

    def _make_rapid(self, inst: InstrumentState) -> bytes:
        """37 字段."""
        pre_close = inst.price * 0.99
        fields = [
            f"{inst.code}.{inst.exchange}",
            self._today(), str(self._exch_time()),
            str(int(time.time() * 1000)),
            f"{pre_close:.3f}", f"{inst.price * 1.005:.3f}",
            str(random.randint(1_000_000, 20_000_000)),
            f"{random.uniform(1e8, 1e9):.2f}",
            f"{inst.price:.3f}", "0.0000",
            f"{inst.price * 1.05:.3f}", f"{inst.price * 0.95:.3f}",
            f"{pre_close * 1.1:.3f}", f"{pre_close * 0.9:.3f}",
            str(random.randint(1000, 100000)),
        ]
        # 5档: ask_p, bid_p, ask_v, bid_v
        for i in range(5):
            ap = round(inst.price + (i + 1) * inst.tick_size, 3)
            bp = round(inst.price - (i + 1) * inst.tick_size, 3)
            fields += [f"{ap:.3f}", f"{bp:.3f}",
                       str(random.randint(100, 5000)),
                       str(random.randint(100, 5000))]
        # bid_count1, ask_count1
        fields += [str(random.randint(1, 10)), str(random.randint(1, 10))]
        return ",".join(fields).encode("utf-8")

    # ---------- 主循环 ----------
    async def run(self, duration_seconds: int = 60):
        self._running = True
        self.stats.start_ts = time.time()
        end_ts = self.stats.start_ts + duration_seconds
        interval = 1.0 / max(self.qps, 1)

        while self._running and time.time() < end_ts:
            inst = random.choice(list(self.instruments.values()))
            self._step(inst)

            inject_err = random.random() < self.inject_error_rate
            roll = random.random()

            if roll < 0.5:                 # 50% trans
                data = self._make_trans(inst) if not inject_err else b"bad,data"
                t = self.parser.parse_trans(data, time.time_ns())
                if t:
                    self.stats.trans += 1
                    if self._cb["trans"]:
                        try: self._cb["trans"](t)
                        except Exception: pass
                else:
                    self.stats.errors_injected += 1
            elif roll < 0.85:              # 35% order
                data = self._make_order(inst) if not inject_err else b"bad,data"
                o = self.parser.parse_order(data, time.time_ns())
                if o:
                    self.stats.order += 1
                    if self._cb["order"]:
                        try: self._cb["order"](o)
                        except Exception: pass
                else:
                    self.stats.errors_injected += 1
            elif roll < 0.95:              # 10% simple book
                data = self._make_simple(inst) if not inject_err else b"bad"
                s = self.parser.parse_simple(data, time.time_ns())
                if s:
                    self.stats.simple += 1
                    if self._cb["simple"]:
                        try: self._cb["simple"](s)
                        except Exception: pass
                else:
                    self.stats.errors_injected += 1
            else:                          # 5% rapid
                data = self._make_rapid(inst) if not inject_err else b"bad"
                r = self.parser.parse_rapid(data, time.time_ns())
                if r:
                    self.stats.rapid += 1
                    if self._cb["rapid"]:
                        try: self._cb["rapid"](r)
                        except Exception: pass
                else:
                    self.stats.errors_injected += 1

            await asyncio.sleep(interval)

        self._running = False

    def stop(self):
        self._running = False


if __name__ == "__main__":
    async def demo():
        s = Level2Simulator(codes=["300750", "600519"], qps=2000)
        await s.run(5)
        print(f"trans={s.stats.trans} order={s.stats.order} "
              f"simple={s.stats.simple} rapid={s.stats.rapid} "
              f"qps={s.stats.qps:.0f}")
    asyncio.run(demo())
