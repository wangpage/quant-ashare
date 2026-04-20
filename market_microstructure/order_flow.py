"""订单流因子 - Level2 数据独家 alpha.

核心思想: 成交价不是信号, "谁在买" 才是信号.

VPIN (Volume-Synchronized Probability of Informed Trading):
    Easley, López de Prado, O'Hara (2012) 经典论文.
    2010 闪崩前 2 周 VPIN 预警准确率 > 90%.

OIR (Order Imbalance Ratio):
    (bid_volume - ask_volume) / (bid_volume + ask_volume)
    短期 (1-5 分钟) 价格预测力最强的微观因子之一.

Cancel Ratio:
    撤单 / 挂单 比率, 高 = 操纵嫌疑大.
    A股某些"妖股" 暴涨暴跌前撤单率会飙到 60% 以上.

Lee-Ready Algorithm:
    Lee & Ready (1991) 金融经典: 没有买卖方向标识时,
    用 tick rule + quote rule 推断 (准确率 ~80%).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ==================== Order Imbalance ====================
def order_imbalance_ratio(
    bid_volumes: list[int] | np.ndarray,
    ask_volumes: list[int] | np.ndarray,
    depth: int = 5,
) -> float:
    """OIR = (Sum(bid_vol) - Sum(ask_vol)) / (Sum(bid_vol) + Sum(ask_vol))

    Args:
        bid_volumes, ask_volumes: 盘口挂单量 (从 1 档到 N 档)
        depth: 聚合前 N 档

    Returns:
        OIR ∈ [-1, 1], > 0 表示买盘主导
    """
    b = np.asarray(bid_volumes)[:depth]
    a = np.asarray(ask_volumes)[:depth]
    total = b.sum() + a.sum()
    if total == 0:
        return 0.0
    return float((b.sum() - a.sum()) / total)


def weighted_oir(
    bid_prices: list[float], ask_prices: list[float],
    bid_volumes: list[int], ask_volumes: list[int],
    mid_price: float,
) -> float:
    """加权 OIR: 离盘口越近的挂单权重越高.

    公式: weight_i = exp(- |price_i - mid| / tick)
    """
    b_p = np.asarray(bid_prices)
    a_p = np.asarray(ask_prices)
    b_v = np.asarray(bid_volumes)
    a_v = np.asarray(ask_volumes)

    tick = max((a_p[0] - b_p[0]) * 2, 0.01)
    b_w = np.exp(-np.abs(b_p - mid_price) / tick)
    a_w = np.exp(-np.abs(a_p - mid_price) / tick)

    num = (b_v * b_w).sum() - (a_v * a_w).sum()
    den = (b_v * b_w).sum() + (a_v * a_w).sum()
    return float(num / den) if den > 0 else 0.0


# ==================== VPIN ====================
def vpin(
    signed_volumes: list[float] | np.ndarray,
    bucket_size: float,
    window_n: int = 50,
) -> float:
    """Volume-Synchronized Probability of Informed Trading.

    算法:
        1. 把成交量分到固定大小的 volume bucket 中
        2. 每个 bucket 计算 buy_vol 和 sell_vol
        3. VPIN = mean(|buy_vol - sell_vol| / bucket_size) over last N buckets

    Args:
        signed_volumes: 有向成交量序列 (+ 买 - 卖)
        bucket_size: 每桶大小 (股或金额)
        window_n: 计算均值的桶数

    Returns:
        VPIN ∈ [0, 1], > 0.4 预警信号

    参考: Easley, Prado, O'Hara. JFE 2012
    """
    sv = np.asarray(signed_volumes, dtype=float)
    abs_v = np.abs(sv)
    cum = np.cumsum(abs_v)

    # 分桶
    buckets: list[tuple[float, float]] = []
    idx = 0
    while idx < len(sv):
        start = idx
        end = idx
        filled = 0.0
        buy = sell = 0.0
        while end < len(sv) and filled < bucket_size:
            remaining = bucket_size - filled
            take = min(abs_v[end], remaining)
            if sv[end] > 0:
                buy += take
            else:
                sell += take
            filled += take
            if take < abs_v[end]:
                abs_v[end] -= take
                break
            end += 1
        buckets.append((buy, sell))
        idx = end

    if not buckets:
        return 0.0
    imb = [abs(b - s) / bucket_size for b, s in buckets[-window_n:]]
    return float(np.mean(imb))


# ==================== Lee-Ready 买卖方向分类 ====================
def lee_ready_classify(
    trade_price: float, trade_volume: int,
    bid_price: float, ask_price: float,
    prev_trade_price: float,
) -> int:
    """Lee & Ready (1991) 推断成交方向.

    规则优先级:
        1. 成交价 > mid → 买 (+1)
        2. 成交价 < mid → 卖 (-1)
        3. 成交价 = mid → 看 tick rule (vs 上一笔)

    Returns:
        +1 = 买方主动, -1 = 卖方主动, 0 = 无法判断
    """
    mid = (bid_price + ask_price) / 2
    if trade_price > mid:
        return 1
    if trade_price < mid:
        return -1
    if trade_price > prev_trade_price:
        return 1
    if trade_price < prev_trade_price:
        return -1
    return 0


def trade_direction_classify(
    trades: pd.DataFrame, quotes: pd.DataFrame,
) -> pd.Series:
    """批量用 Lee-Ready 分类.

    Args:
        trades: 含 ['price', 'volume', 'timestamp']
        quotes: 含 ['bid1', 'ask1', 'timestamp'], as-of join

    Returns:
        Series, +1/-1/0
    """
    merged = pd.merge_asof(
        trades.sort_values("timestamp"),
        quotes.sort_values("timestamp"),
        on="timestamp", direction="backward",
    )
    merged["prev_price"] = merged["price"].shift(1)
    merged["prev_price"] = merged["prev_price"].fillna(merged["price"])

    mid = (merged["bid1"] + merged["ask1"]) / 2
    result = pd.Series(0, index=merged.index)
    result[merged["price"] > mid] = 1
    result[merged["price"] < mid] = -1

    # tick rule fallback
    tick_buy = (merged["price"] == mid) & (merged["price"] > merged["prev_price"])
    tick_sell = (merged["price"] == mid) & (merged["price"] < merged["prev_price"])
    result[tick_buy] = 1
    result[tick_sell] = -1
    return result


# ==================== 撤单率 ====================
def cancel_ratio(
    orders: pd.DataFrame,
    window_minutes: int = 5,
) -> pd.Series:
    """按时间窗口滚动撤单率 = 撤单笔数 / (撤单 + 成交) 笔数.

    高撤单率信号:
        - > 0.7: 疑似幌骗 (spoofing)
        - > 0.4: 流动性浅, 实际成本高
        - < 0.2: 健康

    Args:
        orders: 含 ['timestamp', 'tick_type', 'volume']
            tick_type 深圳: '0'=撤单, '1'=市价委托, '2'=限价, '3'=本方最优
            tick_type 上海: 'A'=新增, 'D'=撤单, 'T'=成交
    """
    df = orders.set_index("timestamp").sort_index()
    cancel_mask = df["tick_type"].isin(["0", "D"])
    exec_mask = df["tick_type"].isin(["1", "2", "3", "A", "T"])

    cancels = cancel_mask.rolling(f"{window_minutes}min").sum()
    execs = exec_mask.rolling(f"{window_minutes}min").sum()
    return cancels / (cancels + execs).clip(lower=1)
