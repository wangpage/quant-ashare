"""T4: 集成到 quant_ashare - 把 Level2 数据接入分钟级因子计算.

场景:
  - Level2 tick -> 实时合成 1min K线
  - 1min K线 + Level2 订单簿不平衡度 -> 分钟级因子
  - 模拟调用风控 + 生成信号
"""
from __future__ import annotations

import asyncio
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from tabulate import tabulate

from level2 import Level2NatsClient, TradeTick, OrderBookSnapshot
from risk import AShareRiskManager
from risk.a_share_rules import Portfolio
from utils.logger import logger


class MinuteBar:
    def __init__(self, code: str, minute: int):
        self.code = code
        self.minute = minute
        self.open = None
        self.high = -1e18
        self.low = 1e18
        self.close = None
        self.volume = 0
        self.amount = 0.0
        self.trade_count = 0
        self.imbalance_sum = 0.0
        self.imbalance_n = 0

    def update_trade(self, t: TradeTick):
        if self.open is None:
            self.open = t.price
        self.close = t.price
        self.high = max(self.high, t.price)
        self.low = min(self.low, t.price)
        self.volume += t.volume
        self.amount += t.amount
        self.trade_count += 1

    def update_book(self, b: OrderBookSnapshot):
        self.imbalance_sum += b.imbalance
        self.imbalance_n += 1

    @property
    def imbalance(self) -> float:
        return self.imbalance_sum / max(self.imbalance_n, 1)

    def as_dict(self):
        return {
            "code": self.code, "minute": self.minute,
            "open": self.open, "high": self.high, "low": self.low, "close": self.close,
            "volume": self.volume, "amount": self.amount,
            "trade_count": self.trade_count, "imbalance": self.imbalance,
        }


class BarAggregator:
    """把 tick 聚合成分钟K."""
    def __init__(self):
        self.bars: dict[tuple[str, int], MinuteBar] = {}
        self.closed: list[dict] = []

    def _minute(self, ts_ns: int) -> int:
        return int(ts_ns // 1_000_000_000 // 60)

    def on_trade(self, t: TradeTick):
        m = self._minute(t.timestamp_ns)
        key = (t.code, m)
        if key not in self.bars:
            # 上一分钟结束
            for k in list(self.bars.keys()):
                if k[0] == t.code and k[1] < m:
                    self.closed.append(self.bars.pop(k).as_dict())
            self.bars[key] = MinuteBar(t.code, m)
        self.bars[key].update_trade(t)

    def on_book(self, b: OrderBookSnapshot):
        m = self._minute(b.timestamp_ns)
        key = (b.code, m)
        if key not in self.bars:
            self.bars[key] = MinuteBar(b.code, m)
        self.bars[key].update_book(b)

    def flush_all(self):
        for bar in self.bars.values():
            self.closed.append(bar.as_dict())
        self.bars.clear()


async def run():
    cli = Level2NatsClient()
    agg = BarAggregator()
    await cli.connect()
    await cli.subscribe_instruments(cli.cfg["test_instruments"])

    cli.on_trade(agg.on_trade)
    cli.on_book(agg.on_book)

    duration = 60
    logger.info(f"运行 {duration}s 采集并聚合分钟K...")
    await cli.run_for(duration)
    agg.flush_all()
    await cli.disconnect()

    return agg.closed, cli.stats


def run_risk_simulation(bars: list[dict]):
    """用聚合好的 K线 模拟喂给风控模块."""
    if not bars:
        return pd.DataFrame(), []
    rm = AShareRiskManager()
    pf = Portfolio(cash=1_000_000, initial_capital=1_000_000,
                   high_water_mark=1_000_000, daily_start_value=1_000_000)
    df = pd.DataFrame(bars).sort_values(["code", "minute"])
    actions = []
    from datetime import date
    today = date.today()
    for _, r in df.iterrows():
        # 简化信号: 订单簿不平衡 > 0.3 触发模拟买入
        if r["imbalance"] > 0.3 and r["close"]:
            shares = int((pf.total_value * 0.05) // (r["close"] * 100)) * 100
            chk = rm.check_buy(
                code=r["code"], industry="test",
                price=r["close"], prev_close=r["close"],
                shares=shares, portfolio=pf,
            )
            actions.append({
                "code": r["code"], "minute": r["minute"],
                "signal": "buy", "price": r["close"],
                "imbalance": r["imbalance"],
                "approved": chk.ok, "reason": chk.reason,
            })
    return df, actions


def main():
    bars, stats = asyncio.run(run())
    df, actions = run_risk_simulation(bars)

    print("\n==== T4 分钟K聚合 ====")
    if df.empty:
        print("(无数据)")
    else:
        print(df.tail(10).to_string(index=False))
        print(f"\n总K线: {len(df)}")

    print("\n==== T4 风控模拟 ====")
    if not actions:
        print("(未触发信号)")
    else:
        print(tabulate(actions[:20], headers="keys"))
        approved = sum(a["approved"] for a in actions)
        print(f"\n触发 {len(actions)}, 通过风控 {approved}")

    print("\n==== T4 行情统计 ====")
    print(f"trade: {stats.trade_count}, order: {stats.order_count}, "
          f"book: {stats.book_count}, QPS: {stats.qps:.0f}")


if __name__ == "__main__":
    main()
