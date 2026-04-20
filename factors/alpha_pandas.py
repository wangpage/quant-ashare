"""A股特化因子 - 纯 pandas 实现, 不依赖 qlib.

把 alpha_ashare.py 的 qlib Expression 翻译成 pandas 向量化,
便于 run_real_research_v2 直接喂 LightGBM.

输入: daily_df 必须有列 code/date/open/close/high/low/volume/amount/pct_chg
输出: feature_df, MultiIndex (date, code), columns = factor names
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


def _per_stock_factors(g: pd.DataFrame) -> pd.DataFrame:
    """对单只股票计算所有因子 (时序)."""
    g = g.sort_values("date").set_index("date").copy()
    close, open_, high, low = g["close"], g["open"], g["high"], g["low"]
    vol = g["volume"].astype(float)
    amt = g.get("amount", vol * close).astype(float)
    ret1 = close.pct_change()

    out = pd.DataFrame(index=g.index)

    # 1. 反转 (A股散户最稳)
    out["REV5"]  = -(close / close.shift(5) - 1)
    out["REV10"] = -(close / close.shift(10) - 1)
    out["REV20"] = -(close / close.shift(20) - 1)

    # 2. 成交量/换手率异动
    vol_ma20 = vol.rolling(20).mean()
    out["TURN_R20"] = vol / (vol_ma20 + EPS)
    out["TURN_R5"]  = vol / (vol.rolling(5).mean() + EPS)
    out["TURN_VOL20"] = vol.rolling(20).std() / (vol_ma20 + EPS)

    # 3. 量价相关
    log_vol = np.log(vol + 1)
    out["CORR_PV10"] = close.rolling(10).corr(log_vol)
    out["CORR_PV20"] = close.rolling(20).corr(log_vol)

    # 4. 波动率
    out["VOL20"] = ret1.rolling(20).std()
    out["VOL5"]  = ret1.rolling(5).std()
    out["VOL_RATIO"] = (out["VOL5"] + EPS) / (out["VOL20"] + EPS)
    out["ATR10"] = (high - low).rolling(10).mean() / (close + EPS)
    out["ATR20"] = (high - low).rolling(20).mean() / (close + EPS)

    # 5. 涨跌停计数 (A股独有)
    is_limit_up = (ret1 > 0.095).astype(float)
    is_limit_dn = (ret1 < -0.095).astype(float)
    out["LIMIT_UP20"] = is_limit_up.rolling(20).sum()
    out["LIMIT_DN20"] = is_limit_dn.rolling(20).sum()
    out["LIMIT_UP5"]  = is_limit_up.rolling(5).sum()

    # 6. 缺口
    out["GAP1"] = open_ / close.shift(1) - 1
    out["GAP5_MA"] = (open_ / close.shift(1) - 1).rolling(5).mean()

    # 7. 均线偏离
    out["MA_DIFF5"]  = close / close.rolling(5).mean()  - 1
    out["MA_DIFF10"] = close / close.rolling(10).mean() - 1
    out["MA_DIFF20"] = close / close.rolling(20).mean() - 1
    out["MA_DIFF60"] = close / close.rolling(60).mean() - 1

    # 8. K线形态
    hl = (high - low).replace(0, np.nan)
    out["BODY_RATIO"] = (close - open_) / (hl + EPS)
    out["UP_SHADOW"]  = (high - np.maximum(close, open_)) / (hl + EPS)
    out["DN_SHADOW"]  = (np.minimum(close, open_) - low)  / (hl + EPS)

    # 9. Bollinger 位置
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    out["BOLL_POS"] = (close - ma20) / (std20 * 2 + EPS)

    # 10. 动量/反转混合
    out["MOM_REV_MIX"] = (close / close.shift(20) - 1) - (close / close.shift(5) - 1)

    # 11. 成交额
    amt_ma20 = amt.rolling(20).mean()
    out["AMT_MOM"] = amt.rolling(5).mean() / (amt_ma20 + EPS)
    out["AMT_VOL"] = amt.rolling(20).std() / (amt_ma20 + EPS)

    # 12. 区间位置
    hh20, ll20 = high.rolling(20).max(), low.rolling(20).min()
    out["POS_IN_RANGE20"] = (close - ll20) / (hh20 - ll20 + EPS)
    hh60, ll60 = high.rolling(60).max(), low.rolling(60).min()
    out["POS_IN_RANGE60"] = (close - ll60) / (hh60 - ll60 + EPS)

    # 13. KDJ 风格
    hh9, ll9 = high.rolling(9).max(), low.rolling(9).min()
    out["KDJ_K"] = (hh9 - close) / (hh9 - ll9 + EPS)

    # 14. 振幅
    out["AMP"] = (high - low) / (close.shift(1) + EPS)
    out["AMP_MA20"] = out["AMP"].rolling(20).mean()

    # 15. 趋势一致性
    out["TREND10"] = np.sign(ret1).rolling(10).sum()

    # 16. 成交量突变
    out["VOL_SPIKE"] = (vol > 2 * vol_ma20).astype(float)

    return out


def compute_pandas_alpha(daily_df: pd.DataFrame) -> pd.DataFrame:
    """批量计算所有股票的因子.

    Returns:
        DataFrame, MultiIndex [date, code], columns=factor names
    """
    daily_df = daily_df.copy()
    daily_df["date"] = pd.to_datetime(daily_df["date"])

    pieces = []
    for code, g in daily_df.groupby("code"):
        f = _per_stock_factors(g)
        f["code"] = code
        pieces.append(f.reset_index())

    if not pieces:
        return pd.DataFrame()

    full = pd.concat(pieces, ignore_index=True)
    full = full.set_index(["date", "code"]).sort_index()
    return full


FACTOR_NAMES = [
    "REV5", "REV10", "REV20",
    "TURN_R20", "TURN_R5", "TURN_VOL20",
    "CORR_PV10", "CORR_PV20",
    "VOL20", "VOL5", "VOL_RATIO", "ATR10", "ATR20",
    "LIMIT_UP20", "LIMIT_DN20", "LIMIT_UP5",
    "GAP1", "GAP5_MA",
    "MA_DIFF5", "MA_DIFF10", "MA_DIFF20", "MA_DIFF60",
    "BODY_RATIO", "UP_SHADOW", "DN_SHADOW",
    "BOLL_POS", "MOM_REV_MIX",
    "AMT_MOM", "AMT_VOL",
    "POS_IN_RANGE20", "POS_IN_RANGE60", "KDJ_K",
    "AMP", "AMP_MA20", "TREND10", "VOL_SPIKE",
]
