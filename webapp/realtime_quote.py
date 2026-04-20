"""实时行情 + 技术指标 (东财直连 fallback 新浪).

绝不编造 "涨 73% 概率" 这种数字. 返回的是:
    - 当前真实价格
    - 基于历史波动的止损/止盈位 (ATR×N)
    - 技术指标 directional bias (不是概率)
    - 多信号一致性 (几个指标看多 / 几个看空)
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import requests


_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Referer": "https://quote.eastmoney.com/",
})


# ==================== 实时报价 ====================
def _secid(code: str) -> str:
    code = str(code).zfill(6)
    return f"1.{code}" if code.startswith("6") else f"0.{code}"


def realtime_quote(code: str, timeout: int = 5) -> dict | None:
    """单只股票实时报价 (东财).

    Returns: {code, name, price, change_pct, volume, amount, high, low, pre_close}
    """
    url = "https://push2.eastmoney.com/api/qt/stock/get"
    params = {
        "secid": _secid(code),
        "ut":    "fa5fd1943c7b386f172d6893dbfba10b",
        "fields": "f43,f44,f45,f46,f47,f48,f57,f58,f60,f62,f168,f169,f170",
    }
    try:
        r = _SESSION.get(url, params=params, timeout=timeout)
        data = r.json().get("data") or {}
    except Exception:
        return None
    if not data:
        return None
    # 东财价格字段是实际价格 × 100
    def _p(f):
        v = data.get(f)
        return float(v) / 100 if isinstance(v, (int, float)) else None
    return {
        "code": str(data.get("f57", code)),
        "name": data.get("f58", ""),
        "price":      _p("f43"),
        "high":       _p("f44"),
        "low":        _p("f45"),
        "open":       _p("f46"),
        "pre_close":  _p("f60"),
        "volume":     data.get("f47"),       # 手
        "amount":     data.get("f48"),       # 元
        "change_pct": float(data.get("f170", 0) or 0) / 100,
        "change_abs": _p("f169"),
        "timestamp":  int(time.time()),
    }


def batch_realtime(codes: list[str]) -> dict[str, dict]:
    """批量实时行情 (东财 clist 接口更高效)."""
    if not codes:
        return {}
    # 用 stock/get 循环, 稳但慢; 也可以直接 clist (一次拉 N 只)
    out = {}
    for c in codes:
        q = realtime_quote(c)
        if q:
            out[c] = q
        time.sleep(0.03)
    return out


# ==================== 技术指标 ====================
def calc_atr(high: pd.Series, low: pd.Series,
             close: pd.Series, period: int = 14) -> float:
    """Average True Range - 波动度量, 用于动态止损."""
    if len(close) < period + 1:
        return 0.0
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])


def calc_rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-12)
    return float(100 - (100 / (1 + rs)).iloc[-1])


def calc_bollinger_pos(close: pd.Series, period: int = 20, k: float = 2) -> float:
    """当前价在布林带中的位置, -1 (下轨) 到 1 (上轨)."""
    ma = close.rolling(period).mean()
    std = close.rolling(period).std()
    if std.iloc[-1] == 0:
        return 0.0
    return float((close.iloc[-1] - ma.iloc[-1]) / (k * std.iloc[-1]))


def calc_macd_signal(close: pd.Series) -> Literal["bull", "bear", "neutral"]:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    if len(dif) < 2:
        return "neutral"
    if dif.iloc[-1] > dea.iloc[-1] and dif.iloc[-2] <= dea.iloc[-2]:
        return "bull"
    if dif.iloc[-1] < dea.iloc[-1] and dif.iloc[-2] >= dea.iloc[-2]:
        return "bear"
    return "neutral"


def calc_ma_position(close: pd.Series) -> dict:
    if len(close) < 60:
        return {"ma5": None, "ma20": None, "ma60": None, "all_above": False}
    last = close.iloc[-1]
    ma5, ma20, ma60 = close.rolling(5).mean().iloc[-1], \
                      close.rolling(20).mean().iloc[-1], \
                      close.rolling(60).mean().iloc[-1]
    return {
        "ma5": float(ma5), "ma20": float(ma20), "ma60": float(ma60),
        "above_ma5":  last > ma5,
        "above_ma20": last > ma20,
        "above_ma60": last > ma60,
        "all_above":  last > ma5 > ma20 > ma60,
        "all_below":  last < ma5 < ma20 < ma60,
    }


@dataclass
class TechnicalSummary:
    code: str
    current_price: float
    atr14: float
    rsi14: float
    boll_pos: float
    macd_signal: str
    ma_info: dict
    # directional bias - 不是概率
    bullish_signals: int
    bearish_signals: int
    signal_detail: list[str]

    @property
    def directional_bias(self) -> str:
        if self.bullish_signals >= 4:
            return "strong_bull"
        if self.bullish_signals >= 3:
            return "lean_bull"
        if self.bearish_signals >= 4:
            return "strong_bear"
        if self.bearish_signals >= 3:
            return "lean_bear"
        return "neutral"

    @property
    def confidence_pct(self) -> float:
        """技术面一致性 - 不是预测概率, 只是指标共振程度."""
        total = self.bullish_signals + self.bearish_signals
        if total == 0:
            return 0.0
        winner = max(self.bullish_signals, self.bearish_signals)
        return winner / 6.0   # 6 个维度最多


def analyze_technicals(df: pd.DataFrame) -> TechnicalSummary:
    """对一只股票的 K 线做全维度技术分析. df 需要 open/high/low/close/volume."""
    close = df["close"]
    high = df["high"]
    low = df["low"]
    last = float(close.iloc[-1])

    atr = calc_atr(high, low, close)
    rsi = calc_rsi(close)
    boll = calc_bollinger_pos(close)
    macd = calc_macd_signal(close)
    ma = calc_ma_position(close)

    bull = 0
    bear = 0
    detail = []

    # 1. 均线位置
    if ma.get("all_above"):
        bull += 1
        detail.append("✓ 均线多头排列 (价格 > MA5 > MA20 > MA60)")
    elif ma.get("all_below"):
        bear += 1
        detail.append("✗ 均线空头排列")
    elif ma.get("above_ma20"):
        bull += 0.5
        detail.append("△ 价格站上 MA20")

    # 2. MACD
    if macd == "bull":
        bull += 1
        detail.append("✓ MACD 金叉")
    elif macd == "bear":
        bear += 1
        detail.append("✗ MACD 死叉")

    # 3. RSI
    if 50 < rsi < 70:
        bull += 1
        detail.append(f"✓ RSI {rsi:.0f} (强势但未超买)")
    elif rsi >= 80:
        bear += 1
        detail.append(f"⚠ RSI {rsi:.0f} 超买, 回调风险")
    elif rsi < 30:
        bull += 0.5
        detail.append(f"△ RSI {rsi:.0f} 超卖, 反弹机会")

    # 4. 布林带位置
    if boll > 0.9:
        bear += 1
        detail.append("⚠ 触及布林上轨, 回归风险")
    elif boll < -0.9:
        bull += 1
        detail.append("△ 触及布林下轨, 反弹概率")

    # 5. 量价关系
    if len(df) >= 20:
        vol_ratio = df["volume"].iloc[-5:].mean() / df["volume"].iloc[-20:].mean()
        price_5d_chg = (close.iloc[-1] / close.iloc[-6] - 1) if len(close) >= 6 else 0
        if vol_ratio > 1.3 and price_5d_chg > 0.03:
            bull += 1
            detail.append(f"✓ 量价齐升 (5日放量 {vol_ratio:.1f}x + 涨 {price_5d_chg:.1%})")
        elif vol_ratio > 1.3 and price_5d_chg < -0.03:
            bear += 1
            detail.append(f"✗ 量增价跌 (恐慌抛售)")
        elif vol_ratio < 0.6 and price_5d_chg > 0.03:
            bear += 0.5
            detail.append(f"⚠ 缩量上涨 (动能不足)")

    # 6. ATR 波动
    atr_ratio = atr / last if last else 0
    if atr_ratio > 0.05:
        detail.append(f"⚠ ATR {atr_ratio:.1%} 波动大, 仓位减半")

    return TechnicalSummary(
        code=str(df.get("code", pd.Series(["?"])).iloc[-1]) if "code" in df.columns else "?",
        current_price=last,
        atr14=atr, rsi14=rsi, boll_pos=boll,
        macd_signal=macd, ma_info=ma,
        bullish_signals=int(bull),
        bearish_signals=int(bear),
        signal_detail=detail,
    )
