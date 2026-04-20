"""买卖价差相关因子 - 衡量流动性成本的真实代理.

名词:
    Quoted Spread = ask1 - bid1              报价价差 (可见)
    Effective Spread = 2 × |price - mid|     有效价差 (实际成本)
    Realized Spread = 2 × (price - mid_future) 已实现价差 (信息含量扣除)

Roll (1984): 在无订单流信息时, 从成交价时序可估算隐含买卖价差.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def effective_spread(
    trade_price: float, bid: float, ask: float, is_buy: bool,
) -> float:
    """有效价差 (bps): 2 × |P - mid|.

    买卖方向需已知. 如果不知, 用 lee_ready.
    """
    mid = (bid + ask) / 2
    if mid == 0:
        return 0.0
    return 2 * abs(trade_price - mid) / mid * 1e4


def realized_spread(
    trade_price: float, mid_at_trade: float, mid_after: float,
    horizon_seconds: int = 5,
) -> float:
    """已实现价差 (bps): 扣除信息含量后的 "纯流动性成本".

        realized = 2 × (P_trade - mid_after) [买方向]
        realized = 2 × (mid_after - P_trade) [卖方向]

    mid_after: 成交后 N 秒的中间价 (一般 5 秒).

    解读:
        realized > 0 → 做市商赚了 (价格未反向)
        realized < 0 → 做市商亏了 (买入后价格继续上涨, 信息交易)
    """
    if mid_at_trade == 0:
        return 0.0
    return 2 * abs(trade_price - mid_after) / mid_at_trade * 1e4


def depth_weighted_midprice(
    bid_prices: list[float], ask_prices: list[float],
    bid_volumes: list[int], ask_volumes: list[int],
    depth: int = 5,
) -> float:
    """微结构中价 (micro-price): 按挂单量加权, 而非简单平均.

    micro = (bid × ask_vol + ask × bid_vol) / (bid_vol + ask_vol)

    直觉: 如果买单 10000 / 卖单 100, 说明买压大, 真实成交点靠近卖价.

    这是 Stoikov (2018) 推广的概念, 比 mid = (bid+ask)/2 精确得多.
    """
    b_p = np.asarray(bid_prices)[:depth]
    a_p = np.asarray(ask_prices)[:depth]
    b_v = np.asarray(bid_volumes)[:depth]
    a_v = np.asarray(ask_volumes)[:depth]

    if len(b_p) == 0 or len(a_p) == 0:
        return 0.0

    # 最简单 micro-price (1 档)
    bid1, ask1, bv, av = b_p[0], a_p[0], b_v[0], a_v[0]
    if bv + av == 0:
        return (bid1 + ask1) / 2
    return (bid1 * av + ask1 * bv) / (bv + av)


def roll_implicit_spread(trade_prices: pd.Series) -> float:
    """Roll (1984) 隐含价差估计.

    假设成交价噪声负自协方差 (买卖交替), 则:
        spread = 2 × sqrt(-Cov(ΔP_t, ΔP_{t-1}))

    用途: 无盘口数据时的 fallback 流动性代理.
    """
    deltas = trade_prices.diff().dropna()
    if len(deltas) < 3:
        return 0.0
    cov = np.cov(deltas[:-1], deltas[1:])[0, 1]
    if cov >= 0:
        return 0.0  # Roll 模型不适用
    return float(2 * np.sqrt(-cov))


def amihud_illiquidity(
    returns: pd.Series, volumes: pd.Series, window: int = 20,
) -> pd.Series:
    """Amihud (2002) 非流动性:
        ILLIQ = mean(|R_t| / V_t)

    直观: 单位成交额引起的价格变动幅度.
    大 = 流动性差. 是截面差异最大的流动性代理之一.
    """
    impact = returns.abs() / volumes.replace(0, np.nan)
    return impact.rolling(window).mean()
