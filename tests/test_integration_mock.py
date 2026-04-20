"""T4-mock: Simulator -> 分钟K聚合 -> 风控 - 端到端集成测试.

验证 tick -> feature -> signal -> risk 整条链路在无真实NATS时能跑通.
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from tabulate import tabulate

from level2 import Level2Simulator, TradeTick, OrderBookSnapshot
from risk import AShareRiskManager
from risk.a_share_rules import Portfolio


class SecondBar:
    """为了在短测试里看到 bar, 用秒级而非分钟级."""
    def __init__(self, code, ts):
        self.code = code
        self.ts = ts
        self.open = None
        self.high = -1e18
        self.low = 1e18
        self.close = None
        self.volume = 0
        self.amount = 0.0
        self.trade_count = 0
        self.imb_sum = 0.0
        self.imb_n = 0

    def update_trade(self, t):
        if self.open is None:
            self.open = t.price
        self.close = t.price
        self.high = max(self.high, t.price)
        self.low = min(self.low, t.price)
        self.volume += t.volume
        self.amount += t.amount
        self.trade_count += 1

    def update_book(self, b):
        self.imb_sum += b.imbalance
        self.imb_n += 1

    @property
    def imbalance(self):
        return self.imb_sum / max(self.imb_n, 1)

    def as_dict(self):
        return dict(code=self.code, ts=self.ts, open=self.open, high=self.high,
                    low=self.low, close=self.close, volume=self.volume,
                    amount=self.amount, trades=self.trade_count,
                    imbalance=self.imbalance)


class Aggregator:
    def __init__(self, bucket_seconds: int = 1):
        self.bucket = bucket_seconds
        self.bars: dict = {}
        self.closed: list = []

    def _bin(self, ts_ns: int) -> int:
        return int(ts_ns // 1_000_000_000 // self.bucket)

    def on_trade(self, t: TradeTick):
        b = self._bin(t.timestamp_ns)
        k = (t.code, b)
        if k not in self.bars:
            for kk in list(self.bars):
                if kk[0] == t.code and kk[1] < b:
                    self.closed.append(self.bars.pop(kk).as_dict())
            self.bars[k] = SecondBar(t.code, b)
        self.bars[k].update_trade(t)

    def on_book(self, bk: OrderBookSnapshot):
        b = self._bin(bk.timestamp_ns)
        k = (bk.code, b)
        if k not in self.bars:
            self.bars[k] = SecondBar(bk.code, b)
        self.bars[k].update_book(bk)

    def flush(self):
        for bar in self.bars.values():
            self.closed.append(bar.as_dict())
        self.bars.clear()


async def run():
    sim = Level2Simulator(codes=["300750", "600519"], qps=3000)
    agg = Aggregator(bucket_seconds=1)
    sim.on_trade(agg.on_trade)
    sim.on_book(agg.on_book)

    await sim.run(duration_seconds=10)
    agg.flush()
    return sim, agg.closed


def risk_simulation(bars: list[dict]):
    rm = AShareRiskManager()
    pf = Portfolio(cash=1_000_000, initial_capital=1_000_000,
                   high_water_mark=1_000_000, daily_start_value=1_000_000)
    df = pd.DataFrame(bars).sort_values(["code", "ts"]) if bars else pd.DataFrame()
    actions = []
    for _, r in df.iterrows():
        if r["imbalance"] > 0.2 and r["close"]:
            shares = int((pf.total_value * 0.05) // (r["close"] * 100)) * 100
            chk = rm.check_buy(
                code=r["code"], industry="test",
                price=r["close"], prev_close=r["close"],
                shares=shares, portfolio=pf,
            )
            actions.append({
                "code": r["code"], "ts": r["ts"],
                "close": r["close"], "imbalance": round(r["imbalance"], 3),
                "ok": chk.ok, "reason": chk.reason,
            })
    return df, actions


def main():
    print("=" * 60)
    print("  T4 集成测试 (Mock 模式)")
    print("=" * 60)
    sim, bars = asyncio.run(run())
    df, actions = risk_simulation(bars)

    print(f"\n采集统计: trade={sim.stats.trade}, order={sim.stats.order}, "
          f"book={sim.stats.book}")
    print(f"QPS 实测: {sim.stats.qps:.0f}")

    print(f"\n秒级K线: {len(df)} 根")
    if not df.empty:
        print(df.tail(8).to_string(index=False))

    print(f"\n风控触发: {len(actions)} 条")
    if actions:
        ap = sum(a["ok"] for a in actions)
        print(f"  通过: {ap}, 拒绝: {len(actions)-ap}")
        print(tabulate(actions[:10], headers="keys"))


if __name__ == "__main__":
    main()
