"""可交易性掩码 (Tradeable Mask) - 回测中被大量忽视的坑.

A股严苛规则:
    1) 涨停 (≥+9.5%) 时买单无法成交 → 那一天的 label 不能作为样本
    2) 跌停 (≤-9.5%) 时卖单无法成交 → 次日 label 也不能用
    3) 停牌日 / 停牌后 3 日 → 价格失真
    4) 新股上市前 N 日 → 未稳定
    5) 财报前 2 天 / 后 1 天 → 已知信息泄露
    6) 减持 / 增发公告日 → 信息冲击
    7) ST / *ST / 退市整理期 → 流动性陷阱

不过滤这些, 回测 IC 会系统性虚高 (幻方内部估计虚高 20-30%).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def tradeable_mask(
    df: pd.DataFrame,
    limit_up_threshold: float = 0.095,
    limit_down_threshold: float = -0.095,
    min_volume: int = 1,
    min_days_after_ipo: int = 250,
    exclude_st: bool = True,
) -> pd.Series:
    """返回一个 bool Series, True 表示"当日可作为训练/交易样本".

    Args:
        df: 必须含 ['pct_chg', 'volume'] 列, 可选 ['name', 'ipo_days']
    """
    mask = pd.Series(True, index=df.index)

    # 1. 涨跌停 (当日收盘价 ≥ 涨停) 不能买入
    if "pct_chg" in df.columns:
        mask &= df["pct_chg"] < limit_up_threshold * 100
        mask &= df["pct_chg"] > limit_down_threshold * 100

    # 2. 停牌 (成交量 = 0)
    if "volume" in df.columns:
        mask &= df["volume"] >= min_volume

    # 3. 次新股 (上市 < N 天)
    if "ipo_days" in df.columns:
        mask &= df["ipo_days"] >= min_days_after_ipo

    # 4. ST 股
    if exclude_st and "name" in df.columns:
        mask &= ~df["name"].str.contains("ST|退", regex=True, na=False)

    return mask


def event_window_mask(
    df: pd.DataFrame,
    earnings_dates: dict[str, list[str]] | None = None,
    pre_days: int = 2,
    post_days: int = 1,
    other_events: dict[str, list[tuple[str, int, int]]] | None = None,
) -> pd.Series:
    """把特定事件窗口的样本标记为"不可用".

    Args:
        df: 必须含 ['code', 'date'] 列
        earnings_dates: {code: [date strings]}, 财报披露日
        other_events: {code: [(date, pre_days, post_days), ...]}
            如减持公告 (-1, 2), 增发 (-3, 5)

    Returns:
        bool Series, True = 可用
    """
    mask = pd.Series(True, index=df.index)
    if earnings_dates is None and other_events is None:
        return mask

    if "date" not in df.columns or "code" not in df.columns:
        return mask

    df_dates = pd.to_datetime(df["date"])

    if earnings_dates:
        for code, dates in earnings_dates.items():
            stock_idx = df["code"] == code
            for d in dates:
                dt = pd.to_datetime(d)
                window = (df_dates >= dt - pd.Timedelta(days=pre_days)) & \
                         (df_dates <= dt + pd.Timedelta(days=post_days))
                mask &= ~(stock_idx & window)

    if other_events:
        for code, events in other_events.items():
            stock_idx = df["code"] == code
            for d, pre, post in events:
                dt = pd.to_datetime(d)
                window = (df_dates >= dt - pd.Timedelta(days=pre)) & \
                         (df_dates <= dt + pd.Timedelta(days=post))
                mask &= ~(stock_idx & window)

    return mask


def leaky_label_detector(
    features: pd.DataFrame, label: pd.Series,
    threshold: float = 0.3,
) -> list[str]:
    """前视偏差检测: 找出与 label 相关性异常高的特征.

    经验法则:
        - 同期相关性 > 0.3 → 几乎肯定是泄露
        - 先查 feature 是否包含 t+k 信息 (如用到 close[t+1])
    """
    suspicious = []
    for col in features.columns:
        if features[col].dtype.kind not in "if":
            continue
        corr = features[col].corr(label)
        if abs(corr) > threshold:
            suspicious.append(f"{col} (corr={corr:.3f})")
    return suspicious
