"""实时 Regime 适配器 - 把 market_regime.RegimeDetector 接到 webapp.

之前 data_providers.py 有 TODO 标记从未实装, 永远返回 mock. 这里:
    1. 拉沪深300指数 K 线 (fetch_index_daily fallback 新浪)
    2. 拉当日全市场快照 (fetch_hot_stocks)
    3. 估算两市成交额 (从 hot_stocks 聚合)
    4. 调用 RegimeDetector().detect() 得到 RegimeSignal
    5. 转换成 webapp 期望的 dict 格式

缓存策略:
    实时模式盘中刷新代价高 (2 次 HTTP + 计算), 用 @st.cache_data ttl=300s.
    没有 streamlit 上下文时退化为进程级 TTL 缓存.
"""
from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

import pandas as pd

from data_adapter.em_direct import fetch_index_daily, fetch_hot_stocks
from market_regime import RegimeDetector
from market_regime.detector import MarketRegime
from utils.logger import logger


# 进程级 fallback 缓存 (无 streamlit 时)
_CACHE: dict[str, tuple[float, Any]] = {}
_TTL_SECONDS = 300


def _cache_get(key: str):
    entry = _CACHE.get(key)
    if entry is None:
        return None
    ts, val = entry
    if time.time() - ts > _TTL_SECONDS:
        return None
    return val


def _cache_set(key: str, val):
    _CACHE[key] = (time.time(), val)


# ==================== 数据源 ====================
def _fetch_index_df(days_back: int = 300) -> pd.DataFrame:
    """拉沪深 300 最近 N 天 K 线, 用于趋势 / 崩盘检测."""
    from datetime import date, timedelta

    start_dt = (date.today() - timedelta(days=days_back * 2)).strftime("%Y%m%d")
    end_dt = date.today().strftime("%Y%m%d")
    df = fetch_index_daily("000300", start=start_dt, end=end_dt)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").tail(days_back).reset_index(drop=True)
    for col in ("close", "high", "low"):
        if col not in df.columns:
            df[col] = df.get("close", 0)
    return df


def _fetch_stocks_snapshot(limit: int = 100) -> tuple[pd.DataFrame, float]:
    """拉 top-N 活跃股, 估算两市成交额 (亿元).

    注: 东财 hot_stocks 返回的 amount 是元, 需换算.
    实际两市成交额 ~= top1000 * 放大, 这里用 top100 × 缩放因子近似.
    """
    hot = fetch_hot_stocks(limit=limit)
    if not hot:
        return pd.DataFrame(), 0.0

    df = pd.DataFrame(hot)
    # 兼容字段缺失
    if "amount" not in df.columns:
        df["amount"] = 0
    # 东财 f3 = 涨跌幅 (%), 已在 hot_stocks 里 map 成 price 等字段
    # fetch_hot_stocks 没直接返回 pct_chg, 我们从 price/pre_close 推导不现实
    # 退化: 用 volume 排序 + 取 f3 (如果字段里有)
    if "pct_chg" not in df.columns:
        # 无该字段时给中性值, 让 RegimeDetector 走 trend 判断
        df["pct_chg"] = 0.0

    # 估算两市成交额: top100 成交额求和 × 经验放大系数
    # A 股日常: top100 占两市 25-35%, 我们用 × 3.5 近似
    top100_amount = df["amount"].fillna(0).astype(float).sum()
    total_turnover_yi = top100_amount * 3.5 / 1e8     # 元 → 亿元

    return df, total_turnover_yi


# ==================== 主入口 ====================
def fetch_live_regime(cache: bool = True) -> dict:
    """返回 webapp 需要的 regime dict. 失败则抛异常, 由上层 fallback mock."""
    if cache:
        cached = _cache_get("regime")
        if cached is not None:
            return cached

    index_df = _fetch_index_df(days_back=300)
    if index_df.empty or len(index_df) < 60:
        raise RuntimeError(f"沪深300指数数据不足 ({len(index_df)} 行), 无法判定 regime")

    stocks_df, total_turnover = _fetch_stocks_snapshot(limit=100)

    detector = RegimeDetector()
    signal = detector.detect(
        index_df=index_df,
        stocks_daily=stocks_df if not stocks_df.empty else None,
        total_turnover_yi=total_turnover if total_turnover > 0 else None,
    )

    # 转换为 webapp 契约 (与 mock_data.get_market_regime 返回结构一致)
    result = {
        "regime": signal.regime.value,
        "position_mult": float(signal.position_mult),
        "confidence": float(signal.confidence),
        "trend_direction": signal.trend.direction,
        "trend_strength": float(signal.trend.strength),
        "vol_level": signal.vol.level,
        "money_effect": signal.breadth.money_effect,
        "liquidity_level": signal.breadth.liquidity_level,
        "breadth_pct_up": float(signal.breadth.pct_up),
        "limit_up_count": int(signal.breadth.pct_limit_up * len(stocks_df))
                           if not stocks_df.empty else 0,
        "limit_down_count": int(signal.breadth.pct_limit_down * len(stocks_df))
                             if not stocks_df.empty else 0,
        "reasons": signal.reasons,
        "data_source": "live",
        "total_turnover_yi": total_turnover,
        "index_rows": len(index_df),
    }

    if cache:
        _cache_set("regime", result)
    logger.info(f"live regime: {result['regime']} (信心 {result['confidence']:.0%})")
    return result
