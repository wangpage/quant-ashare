"""市场 regime 因子 - 让模型显式知道"当前是牛/熊/震荡"和动量状态.

核心问题(修 leak 后暴露):
    V6 在 2024 年学到的反转策略, 在 2025 牛市完全失效.
    模型没有 regime 感知能力 → 无法切换 mom/rev 极性.

解决: 加 5 个市场级因子, 每日同值广播到所有股票.
    MKT_MOM_60:    大盘 60 日收益率 (顺势 regime)
    MKT_MOM_20:    大盘 20 日收益率 (短期 regime)
    MKT_VOL_60:    大盘 60 日波动率 (高波动 regime)
    MKT_BREADTH_20: 近 20 日涨幅 > 0 的股票占比 (广度)
    MKT_REV_RATIO: MOM_20 / MOM_60 的比例 (反转信号)

预期: 模型学到 "市场涨势强时, 反转因子 → 动量因子"
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_market_regime(daily_df: pd.DataFrame) -> pd.DataFrame:
    """从 kline 计算市场 regime 因子 (universe 等权代替指数).

    Returns:
        DataFrame, MultiIndex (date, code), 每日所有股票 MKT_* 列同值
    """
    df = daily_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["code", "date"])
    df["ret1"] = df.groupby("code")["close"].pct_change()

    # 每日 universe 等权收益 (近似大盘)
    mkt_daily = df.groupby("date")["ret1"].mean()

    # 各期间动量
    cum = (1 + mkt_daily).cumprod()
    mom_20 = cum / cum.shift(20) - 1
    mom_60 = cum / cum.shift(60) - 1

    # 波动率
    vol_60 = mkt_daily.rolling(60).std()

    # 市场广度: 过去 20 日 累计收益 > 0 的股票占比
    ret20_per_stock = df.groupby("code")["ret1"].transform(
        lambda s: (1 + s).rolling(20).apply(np.prod, raw=True) - 1
    )
    df["up_20"] = (ret20_per_stock > 0).astype(int)
    breadth = df.groupby("date")["up_20"].mean()

    # 反转信号: 20日动量 vs 60日动量的比例
    # > 0 且大 = 近期加速, 可能反转; < 0 = 近期疲弱反而
    rev_ratio = mom_20 - mom_60

    regime_raw = pd.DataFrame({
        "MKT_MOM_20":     mom_20,
        "MKT_MOM_60":     mom_60,
        "MKT_VOL_60":     vol_60,
        "MKT_BREADTH_20": breadth,
        "MKT_REV_RATIO":  rev_ratio,
    })

    # 时序 z-score (250 日滚动): 不能走截面 z, 否则同日同值会被打零
    regime_series = pd.DataFrame(index=regime_raw.index, columns=regime_raw.columns)
    for col in regime_raw.columns:
        rolling_mean = regime_raw[col].rolling(250, min_periods=60).mean()
        rolling_std = regime_raw[col].rolling(250, min_periods=60).std()
        regime_series[col] = (
            (regime_raw[col] - rolling_mean) / (rolling_std + 1e-9)
        ).clip(-3, 3)
    regime_series = regime_series.astype(float)

    # 广播到 (date, code) panel
    all_dates = sorted(df["date"].unique())
    all_codes = sorted(df["code"].unique())

    pieces = []
    for col in regime_series.columns:
        # Broadcast: 每个 (date, code) 都用该 date 的 regime 值
        s = regime_series[col].reindex(all_dates)
        wide = pd.DataFrame(
            np.tile(s.values.reshape(-1, 1), (1, len(all_codes))),
            index=pd.DatetimeIndex(all_dates, name="date"),
            columns=pd.Index(all_codes, name="code"),
        )
        long = wide.stack().to_frame(col)
        pieces.append(long)

    out = pd.concat(pieces, axis=1)
    out.index.names = ["date", "code"]
    return out.fillna(0)


REGIME_FACTOR_NAMES = [
    "MKT_MOM_20", "MKT_MOM_60", "MKT_VOL_60",
    "MKT_BREADTH_20", "MKT_REV_RATIO",
]


if __name__ == "__main__":
    from pathlib import Path
    daily = pd.read_parquet(Path(__file__).resolve().parent.parent / "cache" /
                              "kline_20230101_20260420_n500.parquet")
    r = compute_market_regime(daily)
    print(f"shape: {r.shape}")
    print("sample:")
    print(r.xs(r.index.get_level_values("date")[-10], level="date").head(3))
