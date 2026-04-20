"""日内犹豫度因子 - 问题 #6.

核心思想 (时间维度 > 价格维度):
    - 价格在关键位的停留时间 = 犹豫度
    - 久攻不上 → 抛压大;  快速突破 → 强势
    - 冲高回落 vs 低开高走 = 资金真实意图

双版本实现:
    1. compute_proxy_intraday_alpha(daily_df):
       从 daily OHLCV 派生代理指标 - CLV / GAP / 冲高回落次数 / 低开高走
       优点: 覆盖历史完整, 已有 cache 即可
       缺点: 缺失真正的"停留时间"

    2. compute_real_intraday_alpha(minute_df):
       基于真实分钟 K 算停留时间 / 穿越次数 / 尾盘反转
       只对近端窗口有数据 (EM 分钟 K 历史有限)

v5 pipeline 默认用 proxy; 如果已缓存分钟 K, 自动 join real 版.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


# ==================================================================
#  1. Proxy 版 - 基于 daily OHLCV
# ==================================================================
def _per_stock_proxy(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").set_index("date").copy()
    close, open_, high, low = g["close"], g["open"], g["high"], g["low"]
    pre_close = close.shift(1)

    hl_range = (high - low) + EPS
    out = pd.DataFrame(index=g.index)

    # === CLV (Close Location Value) 日内收盘位置 ===
    # (close - low) / (high - low), [0, 1]. 越低 → 尾盘弱 / 被砸
    clv = (close - low) / hl_range
    out["CLV"] = clv                       # 原值
    out["CLV_MA5"] = clv.rolling(5).mean()
    out["CLV_MA20"] = clv.rolling(20).mean()

    # === 冲高回落: 日内高点与收盘的距离 (越大 = 越弱) ===
    reject = (high - close) / hl_range     # [0, 1], 大 → 日内高点被拒
    out["REJECT_FROM_HIGH"] = -reject       # 负号 → 拒绝多 = 弱 = 负信号
    # 20 日内 "冲高回落" 次数 (reject > 0.7)
    out["REJECT_COUNT_20"] = -(reject > 0.7).astype(int).rolling(20).sum()

    # === 低开高走: 开盘低但收盘强 ===
    # GAP_OPEN = (open - pre_close) / pre_close, 负 = 低开
    gap_open = (open_ - pre_close) / (pre_close + EPS)
    # PULL_UP = (close - open) / open, 正 = 日内拉升
    pull_up = (close - open_) / (open_ + EPS)
    # 低开高走组合: gap<0 且 pull_up>0
    low_open_up = ((gap_open < -0.005) & (pull_up > 0.005)).astype(int)
    out["LOW_OPEN_PULL_20"] = low_open_up.rolling(20).sum()

    # === 高开低走: 开盘高但尾盘被砸 (出货信号) ===
    high_open_dn = ((gap_open > 0.005) & (pull_up < -0.005)).astype(int)
    out["HIGH_OPEN_DUMP_20"] = -high_open_dn.rolling(20).sum()

    # === 单日振幅异常: hl_range / close, 高 → 分歧大 ===
    amplitude = hl_range / (close + EPS)
    out["AMP_Z_20"] = ((amplitude - amplitude.rolling(60).mean()) /
                      (amplitude.rolling(60).std() + EPS))

    # === 关键位触及: 过去 20 日冲击 20 日高点但当日未破的次数 ===
    # 代理犹豫度: 触及 20d 高但没收在 20d 高之上 (久攻不上)
    high_20 = high.rolling(20).max()
    close_20_hi = close.rolling(20).max()
    touch_fail = ((high >= high_20.shift(1) * 0.995) &
                  (close < high_20.shift(1) * 1.0)).astype(int)
    out["TOUCH_FAIL_20"] = -touch_fail.rolling(20).sum()

    # === 突破成功: close 收于 20 日高之上 (快速突破 = 强势) ===
    breakout = (close >= high_20.shift(1) * 1.001).astype(int)
    out["BREAKOUT_20"] = breakout.rolling(20).sum()

    # === 尾盘强度 proxy: close / ((high+low)/2) - 1, 越高 = 尾盘推升 ===
    out["LATE_STRENGTH"] = (close / ((high + low) / 2 + EPS) - 1)

    # === Intra-day trend (ret - gap = 盘中贡献), 反映盘中资金意图 ===
    ret1 = close / pre_close - 1
    intra_ret = ret1 - gap_open
    out["INTRA_RET_5"]  = intra_ret.rolling(5).sum()
    out["INTRA_RET_20"] = intra_ret.rolling(20).sum()

    return out


def compute_proxy_intraday_alpha(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Proxy 版 - 基于 daily OHLCV 派生犹豫度.

    输入: daily_df 列 date/code/open/high/low/close
    输出: MultiIndex (date, code) panel
    """
    df = daily_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["code", "date"])
    pieces = []
    for code, g in df.groupby("code"):
        feat = _per_stock_proxy(g)
        feat["code"] = code
        pieces.append(feat.reset_index())
    if not pieces:
        return pd.DataFrame()
    panel = (pd.concat(pieces, ignore_index=True)
               .set_index(["date", "code"]).sort_index())
    return panel


# ==================================================================
#  2. Real 版 - 基于分钟 K (近端数据充足时)
# ==================================================================
def _per_stock_real(g: pd.DataFrame) -> pd.DataFrame:
    """输入: 单股分钟 K (含 date/time/open/close/high/low/volume).
    输出: 每日 1 行, (date, code) 的日内犹豫度特征."""
    g = g.copy()
    g["dt"] = pd.to_datetime(g["date"].astype(str) + " " + g["time"].astype(str))
    g = g.sort_values("dt")
    # 按日分组
    daily_feat = []
    for d, day_df in g.groupby("date"):
        day_df = day_df.reset_index(drop=True)
        if len(day_df) < 5:
            continue
        close = day_df["close"].values
        high = day_df["high"].max()
        low = day_df["low"].min()
        hl = high - low + EPS

        # 日高/日低附近停留分钟数 (±0.2% 区间)
        near_high_mins = ((day_df["high"] >= high * 0.998).sum())
        near_low_mins  = ((day_df["low"]  <= low  * 1.002).sum())
        # 穿越日中位价的次数
        mid = (high + low) / 2
        cross_count = int((np.diff(np.sign(close - mid)) != 0).sum())
        # 尾盘 30 分钟涨跌
        tail = day_df.tail(6) if len(day_df) >= 6 else day_df
        tail_ret = (tail["close"].iloc[-1] / tail["open"].iloc[0] - 1)
        # 最大单分钟成交量占日比
        vol_max_ratio = day_df["volume"].max() / (day_df["volume"].sum() + EPS)
        # 开盘 30 min 占全日量比 (典型开盘抢筹模式)
        head = day_df.head(6)
        vol_head_ratio = head["volume"].sum() / (day_df["volume"].sum() + EPS)

        daily_feat.append({
            "date": d,
            "R_HIGH_DWELL":   near_high_mins / len(day_df),
            "R_LOW_DWELL":    near_low_mins  / len(day_df),
            "R_MID_CROSS":    cross_count,
            "R_TAIL_RET":     tail_ret,
            "R_VOL_PEAK":     vol_max_ratio,
            "R_VOL_HEAD":     vol_head_ratio,
        })
    if not daily_feat:
        return pd.DataFrame()
    out = pd.DataFrame(daily_feat).set_index("date")
    out.index = pd.to_datetime(out.index)
    return out


def compute_real_intraday_alpha(minute_df: pd.DataFrame) -> pd.DataFrame:
    """真实分钟 K 版 - 需 minute_df 含 date/time/code/open/high/low/close/volume."""
    if minute_df.empty:
        return pd.DataFrame()
    df = minute_df.copy()
    pieces = []
    for code, g in df.groupby("code"):
        feat = _per_stock_real(g)
        if feat.empty:
            continue
        feat["code"] = code
        pieces.append(feat.reset_index())
    if not pieces:
        return pd.DataFrame()
    panel = (pd.concat(pieces, ignore_index=True)
               .set_index(["date", "code"]).sort_index())
    return panel


PROXY_INTRADAY_FACTOR_NAMES = [
    "CLV", "CLV_MA5", "CLV_MA20",
    "REJECT_FROM_HIGH", "REJECT_COUNT_20",
    "LOW_OPEN_PULL_20", "HIGH_OPEN_DUMP_20",
    "AMP_Z_20",
    "TOUCH_FAIL_20", "BREAKOUT_20",
    "LATE_STRENGTH",
    "INTRA_RET_5", "INTRA_RET_20",
]
REAL_INTRADAY_FACTOR_NAMES = [
    "R_HIGH_DWELL", "R_LOW_DWELL", "R_MID_CROSS",
    "R_TAIL_RET", "R_VOL_PEAK", "R_VOL_HEAD",
]
INTRADAY_FACTOR_NAMES = PROXY_INTRADAY_FACTOR_NAMES  # 默认只用 proxy


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    cache = Path(__file__).resolve().parent.parent / "cache"
    kl = pd.read_parquet(cache / "kline_20230101_20260420_n200.parquet")
    # 取 20 只股小样本
    codes = kl["code"].unique()[:20]
    sub = kl[kl["code"].isin(codes)]
    panel = compute_proxy_intraday_alpha(sub)
    print(f"proxy 犹豫度因子 shape={panel.shape}")
    print(f"  cols={list(panel.columns)}")
    print(f"  nan 率={panel.isna().mean().mean():.2%}")
    print(f"\n样本:\n{panel.head(3)}")
