"""时区 / 时钟偏差处理.

高频场景下, 数据源服务器时钟与交易所毫秒钟差 50-200 ms 非常常见,
这在分钟级以上无影响, 但做 Level2 tick 量化时会导致:
    1. recv_time - exchange_time 算出负延迟 (时钟跑快)
    2. 多数据源 merge 时 as-of join 错配
    3. 实盘下单时触发交易所拒单

头部机构处理:
    - NTP 同步所有服务器到 PTP 级别 (< 10us)
    - 每小时扫描时钟偏差, 补偿到下游计算
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd


def clock_skew_detector(
    exchange_ts_ms: "np.ndarray | pd.Series",
    local_ts_ms: "np.ndarray | pd.Series",
    sample_limit: int = 10000,
) -> dict:
    """检测本地时钟相对交易所时钟的偏差.

    Args:
        exchange_ts_ms: 交易所发出的时间戳 (ms)
        local_ts_ms: 本地接收时间戳 (ms)
        sample_limit: 最多取多少样本估计

    Returns:
        {mean_skew_ms, p50, p95, p99, max}
        正值 = 本地跑快, 负值 = 本地跑慢
    """
    ex = np.asarray(exchange_ts_ms)[-sample_limit:]
    lo = np.asarray(local_ts_ms)[-sample_limit:]
    diff = lo - ex        # 正常 > 0 (传输延迟)
    return {
        "samples": len(diff),
        "mean_ms": float(np.mean(diff)),
        "p50_ms": float(np.percentile(diff, 50)),
        "p95_ms": float(np.percentile(diff, 95)),
        "p99_ms": float(np.percentile(diff, 99)),
        "max_ms": float(np.max(diff)),
        "neg_count": int((diff < 0).sum()),
        "suggestion": (
            "本地时钟同步异常, 建议 ntpdate"
            if (diff < 0).sum() > 0.05 * len(diff) else
            "正常"
        ),
    }


def align_to_exchange_time(
    df: pd.DataFrame,
    ts_col: str = "local_ts_ms",
    ref_exchange_ts_col: str | None = None,
    offset_ms: float | None = None,
) -> pd.DataFrame:
    """用偏差补偿, 把本地时间对齐到交易所时间.

    Args:
        df: 原始数据
        ts_col: 本地时间列
        ref_exchange_ts_col: 交易所时间列 (有则自动算 offset)
        offset_ms: 手动指定偏差 (优先)
    """
    out = df.copy()
    if offset_ms is None and ref_exchange_ts_col:
        diff = out[ts_col] - out[ref_exchange_ts_col]
        offset_ms = float(diff.median())
    elif offset_ms is None:
        offset_ms = 0.0

    out[f"{ts_col}_aligned"] = out[ts_col] - offset_ms
    return out


def unify_tz_to_shanghai(
    df: pd.DataFrame, date_col: str = "date", tz_input: str = "UTC",
) -> pd.DataFrame:
    """多源数据时区统一到 Asia/Shanghai.

    常见错误: akshare 本地时间 vs CCXT UTC vs Wind 纽约时间混用.
    """
    out = df.copy()
    if date_col not in out.columns:
        return out
    dt = pd.to_datetime(out[date_col])
    if dt.dt.tz is None:
        dt = dt.dt.tz_localize(tz_input)
    out[date_col] = dt.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    return out


def detect_dst_artifacts(df: pd.DataFrame, date_col: str = "date") -> list:
    """检测夏令时切换造成的 1h 错位.

    A股无夏令时, 但美股/港股混数据时会有.
    """
    issues = []
    if date_col not in df.columns:
        return issues
    dt = pd.to_datetime(df[date_col])
    diffs = dt.diff().dt.total_seconds() / 3600
    # 相邻条目差正好 23h 或 25h
    suspects = diffs[(diffs == 23) | (diffs == 25)]
    if len(suspects) > 0:
        issues.append(f"疑似 DST 切换: {len(suspects)} 处")
    return issues
