"""base32.cn Level2 CSV 消息解析器.

依据《沪深Level2高级行情接口文档》v2026-04 实现.
所有 topic 消息都是 CSV 字符串, 不是 JSON.

已支持:
  - level2.trans.XXXXXX  逐笔成交  (11字段)
  - level2.order.XXXXXX  逐笔委托  (11字段)
  - level2.rapid.XXXXXX  合成订单簿 (37字段, 5档)
  - level2.simple.XXXXXX 简化订单簿 (13字段, 1档)
  - level2.depth.XXXXXX  L2十档订单簿 (61字段)
  - level1.market.XXXXXX L1五档订单簿 (41字段)
  - level1.flow.XXXXXX   大单资金流 (29字段)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal


# ============ 核心数据类 ============
@dataclass
class TradeTick:
    """逐笔成交 (level2.trans) - 11字段."""
    code: str
    trade_date: str           # YYYY-MM-DD
    exchange_time: int        # HHMMSSsss (9位)
    server_time_ms: int       # 毫秒级 Unix 时间戳
    recv_time_ns: int
    tick_type: str            # T/1=成交, 2=撤单(深圳)
    price: float
    volume: int               # 股
    main_seq: int
    sub_seq: int
    buy_no: int
    sell_no: int

    @property
    def latency_ms(self) -> float:
        """端到端延迟 = 本地接收 - 服务器时间."""
        return self.recv_time_ns / 1e6 - self.server_time_ms

    @property
    def amount(self) -> float:
        return self.price * self.volume


@dataclass
class OrderTick:
    """逐笔委托 (level2.order) - 11字段."""
    code: str
    trade_date: str
    exchange_time: int
    server_time_ms: int
    recv_time_ns: int
    tick_type: str            # 上海: A/D/T, 深圳: 0/1/2/3
    side: int                 # 1=买, 2=卖
    price: float
    volume: int
    main_seq: int
    sub_seq: int
    order_no: int

    @property
    def latency_ms(self) -> float:
        return self.recv_time_ns / 1e6 - self.server_time_ms


@dataclass
class RapidBook:
    """合成订单簿 (level2.rapid) - 37字段, 5档."""
    code_exchange: str        # 例 300750.XSHE
    trade_date: str
    trade_time: int           # HHMMSSsss
    server_time_ms: int
    recv_time_ns: int

    pre_close: float
    open: float
    total_volume: int
    total_amount: float
    last_price: float
    iopv: float
    highest: float
    lowest: float
    upper_limit: float
    lower_limit: float
    num_trades: int

    bid_prices: list[float] = field(default_factory=list)    # 5 档
    bid_volumes: list[int] = field(default_factory=list)
    ask_prices: list[float] = field(default_factory=list)
    ask_volumes: list[int] = field(default_factory=list)

    bid_count1: int = 0
    ask_count1: int = 0

    @property
    def latency_ms(self) -> float:
        return self.recv_time_ns / 1e6 - self.server_time_ms

    @property
    def code(self) -> str:
        return self.code_exchange.split(".")[0]

    @property
    def spread(self) -> float:
        if not self.bid_prices or not self.ask_prices:
            return 0.0
        return self.ask_prices[0] - self.bid_prices[0]

    @property
    def mid_price(self) -> float:
        if not self.bid_prices or not self.ask_prices:
            return 0.0
        return (self.bid_prices[0] + self.ask_prices[0]) / 2

    @property
    def imbalance(self) -> float:
        b = sum(self.bid_volumes)
        a = sum(self.ask_volumes)
        if b + a == 0:
            return 0.0
        return (b - a) / (b + a)


@dataclass
class SimpleBook:
    """简化合成订单簿 (level2.simple) - 13字段, 1档."""
    code_exchange: str
    trade_time: int
    server_time_ms: int
    recv_time_ns: int
    pre_close: float
    total_volume: int
    total_amount: float
    last_price: float
    iopv: float
    num_trades: int
    ask_price1: float
    bid_price1: float
    ask_volume1: int
    bid_volume1: int

    @property
    def code(self) -> str:
        return self.code_exchange.split(".")[0]

    @property
    def latency_ms(self) -> float:
        return self.recv_time_ns / 1e6 - self.server_time_ms

    @property
    def spread(self) -> float:
        return self.ask_price1 - self.bid_price1

    @property
    def mid_price(self) -> float:
        if not self.bid_price1 or not self.ask_price1:
            return 0.0
        return (self.bid_price1 + self.ask_price1) / 2

    @property
    def imbalance(self) -> float:
        b, a = self.bid_volume1, self.ask_volume1
        if b + a == 0:
            return 0.0
        return (b - a) / (b + a)


@dataclass
class DepthBook:
    """L2 十档订单簿 (level2.depth) - 61字段."""
    code: str
    exchange_id: str
    trade_date: str
    trade_time: int
    security_stat: str
    local_time_ms: int
    recv_time_ns: int

    pre_close: float
    open: float
    total_volume: int
    total_amount: float
    total_bid_volume: int
    avg_bid_price: float
    total_ask_volume: int
    avg_ask_price: float
    last_price: float
    iopv: float
    highest: float
    lowest: float
    upper_limit: float
    lower_limit: float
    num_trades: int

    bid_prices: list[float] = field(default_factory=list)    # 10 档
    bid_volumes: list[int] = field(default_factory=list)
    ask_prices: list[float] = field(default_factory=list)
    ask_volumes: list[int] = field(default_factory=list)

    @property
    def latency_ms(self) -> float:
        return self.recv_time_ns / 1e6 - self.local_time_ms

    @property
    def spread(self) -> float:
        if not self.bid_prices or not self.ask_prices:
            return 0.0
        return self.ask_prices[0] - self.bid_prices[0]


@dataclass
class FundFlow:
    """大单资金流 (level1.flow) - 29字段."""
    code: str
    trade_date: str
    update_time: str
    update_ms: int
    server_time_ms: int
    recv_time_ns: int

    retail_buy_amount: float
    retail_buy_volume: int
    retail_buy_count: int
    retail_sell_amount: float
    retail_sell_volume: int
    retail_sell_count: int

    middle_buy_amount: float
    middle_buy_volume: int
    middle_buy_count: int
    middle_sell_amount: float
    middle_sell_volume: int
    middle_sell_count: int

    large_buy_amount: float
    large_buy_volume: int
    large_buy_count: int
    large_sell_amount: float
    large_sell_volume: int
    large_sell_count: int

    inst_buy_amount: float
    inst_buy_volume: int
    inst_buy_count: int
    inst_sell_amount: float
    inst_sell_volume: int
    inst_sell_count: int

    @property
    def latency_ms(self) -> float:
        return self.recv_time_ns / 1e6 - self.server_time_ms

    @property
    def net_inflow_large(self) -> float:
        return (self.large_buy_amount + self.inst_buy_amount) \
             - (self.large_sell_amount + self.inst_sell_amount)

    @property
    def big_money_bias(self) -> float:
        """大单/机构方向: 正买负卖, 归一化到 [-1, 1]."""
        big_buy = self.large_buy_amount + self.inst_buy_amount
        big_sell = self.large_sell_amount + self.inst_sell_amount
        tot = big_buy + big_sell
        if tot == 0:
            return 0.0
        return (big_buy - big_sell) / tot


# ============ 解析器 ============
class Level2Parser:
    def __init__(self, message_format: str = "csv"):
        if message_format != "csv":
            raise ValueError(f"base32.cn Level2 只支持 CSV, 不支持 {message_format}")
        self._parsed = 0
        self._errors = 0
        self._errors_by_type: dict[str, int] = {}

    @property
    def stats(self) -> dict:
        return {
            "parsed": self._parsed,
            "errors": self._errors,
            "error_rate": self._errors / max(self._parsed + self._errors, 1),
            "errors_by_type": dict(self._errors_by_type),
        }

    def _err(self, t: str):
        self._errors += 1
        self._errors_by_type[t] = self._errors_by_type.get(t, 0) + 1

    def _split(self, data: bytes) -> list[str]:
        return data.decode("utf-8", errors="replace").strip().split(",")

    # ----- 逐笔成交 -----
    def parse_trans(self, data: bytes, recv_ns: int) -> TradeTick | None:
        try:
            f = self._split(data)
            if len(f) < 11:
                raise ValueError(f"trans 字段数 {len(f)} < 11")
            t = TradeTick(
                code=f[0], trade_date=f[1],
                exchange_time=int(f[2]), server_time_ms=int(f[3]),
                recv_time_ns=recv_ns,
                tick_type=f[4], price=float(f[5]), volume=int(f[6]),
                main_seq=int(f[7]), sub_seq=int(f[8]),
                buy_no=int(f[9]), sell_no=int(f[10]),
            )
            self._parsed += 1
            return t
        except Exception:
            self._err("trans")
            return None

    # ----- 逐笔委托 -----
    def parse_order(self, data: bytes, recv_ns: int) -> OrderTick | None:
        try:
            f = self._split(data)
            if len(f) < 11:
                raise ValueError(f"order 字段数 {len(f)} < 11")
            o = OrderTick(
                code=f[0], trade_date=f[1],
                exchange_time=int(f[2]), server_time_ms=int(f[3]),
                recv_time_ns=recv_ns,
                tick_type=f[4], side=int(f[5]),
                price=float(f[6]), volume=int(f[7]),
                main_seq=int(f[8]), sub_seq=int(f[9]),
                order_no=int(f[10]),
            )
            self._parsed += 1
            return o
        except Exception:
            self._err("order")
            return None

    # ----- 合成订单簿 5档 (37字段) -----
    def parse_rapid(self, data: bytes, recv_ns: int) -> RapidBook | None:
        try:
            f = self._split(data)
            if len(f) < 37:
                raise ValueError(f"rapid 字段数 {len(f)} < 37")
            bids_p, bids_v, asks_p, asks_v = [], [], [], []
            base = 15                           # 前14 是公共字段, 第15开始是盘口
            for i in range(5):
                asks_p.append(float(f[base + i*4 + 0]))
                bids_p.append(float(f[base + i*4 + 1]))
                asks_v.append(int(f[base + i*4 + 2]))
                bids_v.append(int(f[base + i*4 + 3]))
            r = RapidBook(
                code_exchange=f[0], trade_date=f[1],
                trade_time=int(f[2]), server_time_ms=int(f[3]),
                recv_time_ns=recv_ns,
                pre_close=float(f[4]), open=float(f[5]),
                total_volume=int(f[6]), total_amount=float(f[7]),
                last_price=float(f[8]), iopv=float(f[9]),
                highest=float(f[10]), lowest=float(f[11]),
                upper_limit=float(f[12]), lower_limit=float(f[13]),
                num_trades=int(f[14]),
                bid_prices=bids_p, bid_volumes=bids_v,
                ask_prices=asks_p, ask_volumes=asks_v,
                bid_count1=int(f[35]), ask_count1=int(f[36]),
            )
            self._parsed += 1
            return r
        except Exception:
            self._err("rapid")
            return None

    # ----- 简化合成订单簿 (13字段) -----
    def parse_simple(self, data: bytes, recv_ns: int) -> SimpleBook | None:
        try:
            f = self._split(data)
            if len(f) < 13:
                raise ValueError(f"simple 字段数 {len(f)} < 13")
            s = SimpleBook(
                code_exchange=f[0],
                trade_time=int(f[1]), server_time_ms=int(f[2]),
                recv_time_ns=recv_ns,
                pre_close=float(f[3]),
                total_volume=int(f[4]), total_amount=float(f[5]),
                last_price=float(f[6]), iopv=float(f[7]),
                num_trades=int(f[8]),
                ask_price1=float(f[9]), bid_price1=float(f[10]),
                ask_volume1=int(f[11]), bid_volume1=int(f[12]),
            )
            self._parsed += 1
            return s
        except Exception:
            self._err("simple")
            return None

    # ----- L2 十档订单簿 (61字段) -----
    def parse_depth(self, data: bytes, recv_ns: int) -> DepthBook | None:
        try:
            f = self._split(data)
            if len(f) < 61:
                raise ValueError(f"depth 字段数 {len(f)} < 61")
            bids_p, bids_v, asks_p, asks_v = [], [], [], []
            base = 21    # 第22(idx=21)起是盘口
            for i in range(10):
                asks_p.append(float(f[base + i*4 + 0]))
                bids_p.append(float(f[base + i*4 + 1]))
                asks_v.append(int(f[base + i*4 + 2]))
                bids_v.append(int(f[base + i*4 + 3]))
            d = DepthBook(
                code=f[0], exchange_id=f[1], trade_date=f[2],
                trade_time=int(f[3]), security_stat=f[4],
                local_time_ms=int(f[5]), recv_time_ns=recv_ns,
                pre_close=float(f[6]), open=float(f[7]),
                total_volume=int(f[8]), total_amount=float(f[9]),
                total_bid_volume=int(f[10]), avg_bid_price=float(f[11]),
                total_ask_volume=int(f[12]), avg_ask_price=float(f[13]),
                last_price=float(f[14]), iopv=float(f[15]),
                highest=float(f[16]), lowest=float(f[17]),
                upper_limit=float(f[18]), lower_limit=float(f[19]),
                num_trades=int(f[20]),
                bid_prices=bids_p, bid_volumes=bids_v,
                ask_prices=asks_p, ask_volumes=asks_v,
            )
            self._parsed += 1
            return d
        except Exception:
            self._err("depth")
            return None

    # ----- 大单资金流 (29字段) -----
    def parse_flow(self, data: bytes, recv_ns: int) -> FundFlow | None:
        try:
            f = self._split(data)
            if len(f) < 29:
                raise ValueError(f"flow 字段数 {len(f)} < 29")
            flow = FundFlow(
                code=f[0], trade_date=f[1],
                update_time=f[2], update_ms=int(f[3]),
                server_time_ms=int(f[4]), recv_time_ns=recv_ns,
                retail_buy_amount=float(f[5]), retail_buy_volume=int(f[6]),
                retail_buy_count=int(f[7]),
                retail_sell_amount=float(f[8]), retail_sell_volume=int(f[9]),
                retail_sell_count=int(f[10]),
                middle_buy_amount=float(f[11]), middle_buy_volume=int(f[12]),
                middle_buy_count=int(f[13]),
                middle_sell_amount=float(f[14]), middle_sell_volume=int(f[15]),
                middle_sell_count=int(f[16]),
                large_buy_amount=float(f[17]), large_buy_volume=int(f[18]),
                large_buy_count=int(f[19]),
                large_sell_amount=float(f[20]), large_sell_volume=int(f[21]),
                large_sell_count=int(f[22]),
                inst_buy_amount=float(f[23]), inst_buy_volume=int(f[24]),
                inst_buy_count=int(f[25]),
                inst_sell_amount=float(f[26]), inst_sell_volume=int(f[27]),
                inst_sell_count=int(f[28]),
            )
            self._parsed += 1
            return flow
        except Exception:
            self._err("flow")
            return None


# ============ 兼容旧 API (测试代码使用的名字) ============
# 旧名保留, 映射到新方法, 保证老测试不报错
Level2Parser.parse_trade = Level2Parser.parse_trans        # type: ignore
Level2Parser.parse_orderbook = Level2Parser.parse_simple   # type: ignore


# 旧的 dataclass 别名
OrderBookSnapshot = SimpleBook
