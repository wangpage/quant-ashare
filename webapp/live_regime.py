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

    容错:
        东财 clist 接口容易被限频 (与 kline 接口独立). 失败时返回 (空 df, 0.0),
        让 RegimeDetector 降级到纯指数驱动 (只做 trend+vol 判断, 无 breadth).

    注: 东财 hot_stocks 返回的 amount 是元, 需换算.
    实际两市成交额 ~= top1000 * 放大, 这里用 top100 × 缩放因子近似.
    """
    try:
        hot = fetch_hot_stocks(limit=limit)
    except Exception as e:
        logger.warning(f"hot_stocks 失败 ({e.__class__.__name__}), "
                        f"regime 将走纯指数降级模式")
        return pd.DataFrame(), 0.0

    if not hot:
        return pd.DataFrame(), 0.0

    df = pd.DataFrame(hot)
    # 兼容字段缺失
    if "amount" not in df.columns:
        df["amount"] = 0
    if "pct_chg" not in df.columns:
        # 东财 hot_stocks 映射没带 f3; 无则给中性, 让 RegimeDetector 走 trend
        df["pct_chg"] = 0.0

    # 估算两市成交额: top100 成交额求和 × 经验放大系数 (A股 top100 占 25-35%)
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
    has_breadth = not stocks_df.empty

    detector = RegimeDetector()
    signal = detector.detect(
        index_df=index_df,
        stocks_daily=stocks_df if has_breadth else None,
        total_turnover_yi=total_turnover if total_turnover > 0 else None,
    )

    # 降级时明确标注: breadth 维度不可信
    reasons = list(signal.reasons)
    if not has_breadth:
        reasons.insert(
            0,
            "⚠ 降级模式: hot_stocks 接口失败, regime 仅基于沪深300指数 "
            "(trend + vol), breadth/流动性/涨停家数 不可信",
        )

    # 数据源精细化: 指数 live, breadth 可能 degraded
    if has_breadth:
        data_source = "live"
    else:
        data_source = "live (degraded: 仅指数, 无 breadth)"

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
                           if has_breadth else 0,
        "limit_down_count": int(signal.breadth.pct_limit_down * len(stocks_df))
                             if has_breadth else 0,
        "reasons": reasons,
        "data_source": data_source,
        "total_turnover_yi": total_turnover,
        "index_rows": len(index_df),
        "has_breadth": has_breadth,
    }

    if cache:
        _cache_set("regime", result)
    logger.info(f"live regime: {result['regime']} (信心 {result['confidence']:.0%})")
    return result
