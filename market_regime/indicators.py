"""Regime 检测所需的底层指标计算 - 纯 numpy/pandas, 不依赖 qlib."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ==================== 1. 趋势指标 ====================
@dataclass
class TrendResult:
    direction: str          # up / down / flat
    strength: float         # 0-1
    ma_spread: float        # (price - ma200) / ma200
    consec_days: int        # 连续上涨或下跌天数


def compute_trend(index_close: pd.Series,
                  short: int = 20, long: int = 200) -> TrendResult:
    """基于均线系统判断趋势."""
    if len(index_close) < long:
        return TrendResult("flat", 0.0, 0.0, 0)

    ma_s = index_close.rolling(short).mean().iloc[-1]
    ma_l = index_close.rolling(long).mean().iloc[-1]
    price = float(index_close.iloc[-1])
    ma_spread = (price - ma_l) / ma_l if ma_l else 0.0

    # 方向: 严格要求 price > ma_s > ma_l 容易被震荡打断, 改为综合判断
    if price > ma_l and ma_spread > 0.02:
        direction = "up"
    elif price < ma_l and ma_spread < -0.02:
        direction = "down"
    elif price > ma_s > ma_l:
        direction = "up"
    elif price < ma_s < ma_l:
        direction = "down"
    else:
        direction = "flat"

    # 强度: 综合 ma_spread 绝对值 + ma_s vs ma_l 距离
    strength = min(1.0, abs(ma_spread) * 5)

    # 连续涨跌天数
    rets = index_close.diff().dropna().values
    consec = 0
    if len(rets) > 0:
        sign = np.sign(rets[-1])
        for r in reversed(rets):
            if np.sign(r) == sign and sign != 0:
                consec += 1
            else:
                break

    return TrendResult(direction, float(strength),
                       float(ma_spread), int(consec))


# ==================== 2. 波动率 ====================
@dataclass
class VolResult:
    vol_20d: float          # 20日年化波动率
    vol_5d: float           # 5日短期波动
    vol_ratio: float        # 短期/长期
    atr_ratio: float        # ATR/close
    level: str              # low / normal / high / extreme


def compute_volatility(index_df: pd.DataFrame) -> VolResult:
    """index_df 含 close/high/low 列."""
    if len(index_df) < 20:
        return VolResult(0, 0, 1, 0, "normal")

    close = index_df["close"]
    rets = close.pct_change().dropna()
    vol_20 = float(rets.rolling(20).std().iloc[-1] * np.sqrt(252))
    vol_5 = float(rets.rolling(5).std().iloc[-1] * np.sqrt(252))
    vol_ratio = vol_5 / max(vol_20, 1e-9)

    if "high" in index_df.columns and "low" in index_df.columns:
        tr = (index_df["high"] - index_df["low"]).rolling(14).mean().iloc[-1]
        atr_ratio = float(tr / close.iloc[-1]) if close.iloc[-1] else 0
    else:
        atr_ratio = 0.0

    # A股大盘历史年化波动 ~25%
    if vol_20 > 0.45:
        level = "extreme"
    elif vol_20 > 0.32:
        level = "high"
    elif vol_20 < 0.15:
        level = "low"
    else:
        level = "normal"

    return VolResult(vol_20, vol_5, float(vol_ratio), atr_ratio, level)


# ==================== 3. 市场广度 (breadth) ====================
@dataclass
class BreadthResult:
    pct_up: float           # 当日涨幅>0 股票占比
    pct_limit_up: float     # 涨停比例
    pct_limit_down: float   # 跌停比例
    money_effect: str       # 冷/温/热/沸腾
    liquidity_level: str    # low / normal / high


def compute_breadth(
    stocks_daily: pd.DataFrame,
    total_turnover_yi: float | None = None,
) -> BreadthResult:
    """计算市场广度.

    Args:
        stocks_daily: 当日全市场个股的 DataFrame, 至少含 pct_chg 列.
        total_turnover_yi: 两市合计成交额 (亿元).
    """
    if stocks_daily is None or len(stocks_daily) == 0:
        return BreadthResult(0.5, 0.01, 0.01, "温", "normal")

    if "pct_chg" not in stocks_daily.columns:
        return BreadthResult(0.5, 0.01, 0.01, "温", "normal")

    p = stocks_daily["pct_chg"].dropna()
    pct_up = float((p > 0).mean())
    pct_lim_up = float((p > 9.5).mean())
    pct_lim_dn = float((p < -9.5).mean())

    # 赚钱效应
    if pct_up > 0.7 and pct_lim_up > 0.02:
        money = "沸腾"
    elif pct_up > 0.55:
        money = "热"
    elif pct_up > 0.4:
        money = "温"
    else:
        money = "冷"

    # 流动性
    if total_turnover_yi is None:
        liquidity = "normal"
    elif total_turnover_yi > 12000:
        liquidity = "high"
    elif total_turnover_yi < 6000:
        liquidity = "low"
    else:
        liquidity = "normal"

    return BreadthResult(pct_up, pct_lim_up, pct_lim_dn,
                         money, liquidity)


# ==================== 4. 崩盘信号 ====================
def detect_crash(index_close: pd.Series,
                 threshold_1d: float = -0.05,
                 threshold_5d: float = -0.10) -> bool:
    """崩盘: 单日跌幅超阈值 或 5日累计跌幅超阈值."""
    if len(index_close) < 6:
        return False
    today_ret = index_close.pct_change().iloc[-1]
    five_day_ret = index_close.iloc[-1] / index_close.iloc[-6] - 1
    return bool(today_ret <= threshold_1d or five_day_ret <= threshold_5d)


def detect_euphoria(pct_up: float, pct_limit_up: float,
                    vol_20d: float) -> bool:
    """狂热: 高涨停率 + 高波动 + 市场广度极度正向."""
    return (pct_up > 0.75 and pct_limit_up > 0.03 and vol_20d > 0.30)
