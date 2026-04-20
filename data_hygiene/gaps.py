"""停牌 / 缺失日处理 - 别偷懒用 ffill.

大众做法:
    df.fillna(method='ffill')    ← 停牌日用前值填充

问题:
    - 停牌期可能有重要公告 (公司重组/被ST), 恢复交易时往往跳空
    - 用 ffill 会让模型把"停牌恢复日"看成"正常涨跌",
      学到一堆诡异"信号"
    - 实盘停牌是不能交易的, 训练里包含会过拟合

正确做法:
    - 停牌日样本标记为 NaN (不参与训练)
    - 停牌恢复后 1-2 天也要屏蔽 (价格失真)
    - 缺失值用 cross-sectional median 或 0 填充 (视 feature 性质)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def find_suspension_days(
    df: pd.DataFrame,
    min_gap_days: int = 1,
) -> pd.DataFrame:
    """找出每只股票的停牌区间.

    Args:
        df: 含 ['code', 'date', 'volume'], volume=0 视为停牌

    Returns:
        DataFrame ['code', 'start', 'end', 'days'], 每一行是一个停牌区间.
    """
    need = {"code", "date", "volume"}
    if not need.issubset(df.columns):
        return pd.DataFrame()

    d = df.sort_values(["code", "date"]).copy()
    d["date"] = pd.to_datetime(d["date"])
    d["is_suspended"] = d["volume"] == 0

    out = []
    for code, g in d.groupby("code"):
        g = g.reset_index(drop=True)
        # 识别连续停牌段
        change = g["is_suspended"].ne(g["is_suspended"].shift()).cumsum()
        for _, seg in g.groupby(change):
            if seg["is_suspended"].iloc[0] and len(seg) >= min_gap_days:
                out.append({
                    "code": code,
                    "start": seg["date"].iloc[0],
                    "end": seg["date"].iloc[-1],
                    "days": len(seg),
                })
    return pd.DataFrame(out)


def gap_aware_fill(
    df: pd.DataFrame,
    price_col: str = "close",
    max_ffill_days: int = 0,
) -> pd.DataFrame:
    """停牌感知的缺失填充.

    策略:
        - 同一股票连续 <= max_ffill_days 个缺失: 可以 ffill (节假日)
        - 连续 > max_ffill_days: 标记为 NaN (停牌), 训练时剔除

    默认 max_ffill_days=0: 除了节假日自动跳过外, 停牌保持 NaN.
    """
    need = {"code", "date", price_col}
    if not need.issubset(df.columns):
        return df.copy()

    d = df.sort_values(["code", "date"]).copy()
    d["date"] = pd.to_datetime(d["date"])

    # 按股票分组处理
    filled_parts = []
    for code, g in d.groupby("code"):
        g = g.reset_index(drop=True)
        # 标记连续 NaN 长度
        is_nan = g[price_col].isna()
        run_id = (~is_nan).cumsum()
        nan_runs = is_nan.groupby(run_id).cumsum()

        # 仅对 ≤ max_ffill_days 做 ffill
        mask_short = nan_runs <= max_ffill_days
        g.loc[mask_short, price_col] = g[price_col].ffill().loc[mask_short]

        filled_parts.append(g)

    return pd.concat(filled_parts, ignore_index=True)


def suspension_recovery_mask(
    df: pd.DataFrame, recovery_buffer_days: int = 2,
) -> pd.Series:
    """标记停牌恢复后 N 个交易日为"不可用".

    Returns:
        bool Series, True = 样本可用.
    """
    need = {"code", "date", "volume"}
    if not need.issubset(df.columns):
        return pd.Series(True, index=df.index)

    d = df.sort_values(["code", "date"]).copy()
    d["date"] = pd.to_datetime(d["date"])
    d["is_suspended"] = d["volume"] == 0

    mask = pd.Series(True, index=d.index)
    for code, g in d.groupby("code"):
        # 找 "停牌 -> 复牌" 的转折点
        recovered = (~g["is_suspended"]) & g["is_suspended"].shift(1).fillna(False)
        recovery_idx = g.index[recovered].tolist()
        for idx in recovery_idx:
            pos = g.index.get_loc(idx)
            buffer = g.index[pos:pos + recovery_buffer_days + 1]
            mask.loc[buffer] = False

    return mask.reindex(df.index).fillna(True)


def cross_sectional_fill(
    feature_df: pd.DataFrame, method: str = "median",
) -> pd.DataFrame:
    """按截面 (同日其他股票) 填充缺失特征.

    Args:
        feature_df: 形状 [dates × stocks] 的单因子矩阵
        method: median / mean / zero

    优于 ffill: 不会把"停牌前值"带到交易日.
    """
    if method == "zero":
        return feature_df.fillna(0)
    if method == "mean":
        return feature_df.fillna(feature_df.mean(axis=1), axis=0)
    if method == "median":
        return feature_df.apply(lambda s: s.fillna(s.median()), axis=1)
    raise ValueError(method)
