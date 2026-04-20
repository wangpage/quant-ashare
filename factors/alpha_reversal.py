"""A股反转+低波因子扩展 - 专治小盘散户市.

学术依据:
    - Jegadeesh 1990: 短期反转 (1-4 周)
    - Jegadeesh-Titman 1993: 12-1 月动量
    - Amihud 2002: 非流动性溢价
    - Ang-Hodrick-Xing-Zhang 2006: 低波异常
    - Han-Hirshleifer-Walden 2022: 社交传播偏差推动 A股反转
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


def _per_stock_advanced(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").set_index("date").copy()
    close = g["close"]
    high, low = g["high"], g["low"]
    vol = g["volume"].astype(float)
    amt = g.get("amount", vol * close).astype(float)
    ret1 = close.pct_change()

    out = pd.DataFrame(index=g.index)

    # ---- 1. 多周期反转 (学术经典) ----
    # 1-4 周反转, 方向: 过去跌的未来涨
    out["REV_1"]  = -ret1
    out["REV_3"]  = -(close / close.shift(3)  - 1)
    out["REV_5"]  = -(close / close.shift(5)  - 1)
    out["REV_10"] = -(close / close.shift(10) - 1)
    out["REV_20"] = -(close / close.shift(20) - 1)

    # ---- 2. 12-1 动量 (Jegadeesh-Titman) ----
    # 过去 252 日收益减去最近 21 日 (避开短反转)
    ret_252 = close / close.shift(252) - 1
    ret_21  = close / close.shift(21)  - 1
    out["MOM12_1"] = ret_252 - ret_21

    # 6-1 动量 (缩短版, 减少数据要求)
    ret_126 = close / close.shift(126) - 1
    out["MOM6_1"] = ret_126 - ret_21

    # ---- 3. Amihud 非流动性 ----
    # |日收益| / 成交额, 越高说明流动性越差 → 未来 premium 越高
    amihud_daily = ret1.abs() / (amt + EPS)
    out["AMIHUD_20"] = amihud_daily.rolling(20).mean()
    out["AMIHUD_60"] = amihud_daily.rolling(60).mean()

    # ---- 4. 低波异常 (Ang-Hodrick) ----
    # 低波动率股票 OOS 夏普更高, 取波动率负号
    out["LOW_VOL_20"] = -ret1.rolling(20).std()
    out["LOW_VOL_60"] = -ret1.rolling(60).std()

    # ---- 5. Idio vol (相对大盘的特异波动) - 简化版 ----
    # 用过去 60 日的残差 std 近似, 与 LOW_VOL 差异互补
    # 此处直接用 HL_SPREAD / close 近似 idio vol
    out["IDIO_VOL_60"] = -((high - low) / (close + EPS)).rolling(60).std()

    # ---- 6. 换手率 z-score (截面异常代理) ----
    # 过去 N 日换手率 (用 vol / float_shares 近似, 这里用 vol ratio)
    vol_ma60 = vol.rolling(60).mean()
    out["TURN_Z_60"] = (vol - vol_ma60) / (vol.rolling(60).std() + EPS)
    # 换手率走向 - 近 5d 比 近 60d
    out["TURN_TREND"] = vol.rolling(5).mean() / (vol_ma60 + EPS) - 1

    # ---- 7. 最大单日涨幅 (MAX, Bali-Cakici-Whitelaw 2011) ----
    # 近 20 日最大 1 日涨幅, 捕获彩票偏好股票 (反向信号)
    out["MAX_RET_20"] = -ret1.rolling(20).max()
    out["MAX_RET_5"]  = -ret1.rolling(5).max()

    # ---- 8. 52周高点距离 ----
    max_252 = close.rolling(252).max()
    out["DIST_52W_HIGH"] = close / (max_252 + EPS) - 1

    # ---- 9. 换手率 × 反转交互项 (高换手叠加反转更强) ----
    out["TURN_X_REV5"] = (vol / (vol_ma60 + EPS) - 1) * (-ret1.rolling(5).sum())

    # ---- 10. Skewness (Harvey-Siddique 2000, 高偏度股票低收益) ----
    out["SKEW_20"] = -ret1.rolling(20).skew()
    out["SKEW_60"] = -ret1.rolling(60).skew()

    return out


def compute_advanced_alpha(daily_df: pd.DataFrame) -> pd.DataFrame:
    daily_df = daily_df.copy()
    daily_df["date"] = pd.to_datetime(daily_df["date"])
    pieces = []
    for code, g in daily_df.groupby("code"):
        f = _per_stock_advanced(g)
        f["code"] = code
        pieces.append(f.reset_index())
    if not pieces:
        return pd.DataFrame()
    full = pd.concat(pieces, ignore_index=True).set_index(["date", "code"]).sort_index()
    return full


ADVANCED_FACTOR_NAMES = [
    "REV_1", "REV_3", "REV_5", "REV_10", "REV_20",
    "MOM12_1", "MOM6_1",
    "AMIHUD_20", "AMIHUD_60",
    "LOW_VOL_20", "LOW_VOL_60", "IDIO_VOL_60",
    "TURN_Z_60", "TURN_TREND",
    "MAX_RET_20", "MAX_RET_5",
    "DIST_52W_HIGH",
    "TURN_X_REV5",
    "SKEW_20", "SKEW_60",
]
