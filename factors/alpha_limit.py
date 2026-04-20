"""刀 4 - 涨停/连板/炸板因子. A股独有 alpha 源.

这些因子从日 K 就能自识别, 不需要分钟数据. 经典游资打板 + 次日套利模式:
    - 首板: 题材启动信号
    - 连板高度: 市场情绪温度计 (2板/3板/高位板)
    - 炸板率: 涨停封板强度 (炸板=当日触及涨停但未封住)
    - 涨停 gap: 次日开盘竞价情绪代理 (没有 tick 的最佳替代)
    - 涨停成交量: 封板资金强度

识别规则 (A股主板 10%, 创业板/科创板 20%):
    - ret1 > 0.095 且 close == high → 主板封板涨停
    - ret1 > 0.095 且 close < high → 炸板(触及涨停但未封住)
    - 创业板 300xxx / 科创板 688xxx 阈值 19.5%
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _is_chuangye_or_kechuang(code: str) -> bool:
    """判断创业板/科创板 (20% 涨跌幅). 北交所 30% 更复杂, 我们池子剔除了."""
    return code.startswith(("300", "688"))


def _per_stock_limit(g: pd.DataFrame, code: str) -> pd.DataFrame:
    """单只股的涨停+连板+炸板因子."""
    g = g.sort_values("date").set_index("date").copy()
    close, high, low, open_ = g["close"], g["high"], g["low"], g["open"]
    vol = g["volume"].astype(float)
    amt = g.get("amount", vol * close).astype(float)

    ret1 = close.pct_change()
    prev_close = close.shift(1)
    # 阈值 (创业/科创 20%, 主板 10%, 留 5 bps 余量)
    th = 0.195 if _is_chuangye_or_kechuang(code) else 0.095

    out = pd.DataFrame(index=g.index)

    # === 1. 涨停类型 ===
    touched_limit = (high - prev_close) / prev_close > th  # 盘中触及涨停
    closed_limit = (ret1 > th) & (close >= high * 0.999)    # 封板(收盘=最高)
    one_word     = closed_limit & (open_ == high) & (high == low)  # 一字板
    boom_ban     = touched_limit & (~closed_limit)          # 炸板: 触及但未封

    # 跌停
    touched_dn = (prev_close - low) / prev_close > th
    closed_dn = (ret1 < -th) & (close <= low * 1.001)

    # 基础计数因子
    out["LU_FLAG"]       = closed_limit.astype(int)
    out["LD_FLAG"]       = closed_dn.astype(int)
    out["BOOM_BAN_FLAG"] = boom_ban.astype(int)
    out["ONE_WORD_FLAG"] = one_word.astype(int)

    # === 2. 连板高度 ===
    # 连续涨停天数: reset 到 0 当打破
    streak = np.zeros(len(close))
    c = 0
    lu = closed_limit.values
    for i in range(len(close)):
        c = c + 1 if lu[i] else 0
        streak[i] = c
    out["STREAK_UP"] = streak

    # 最近 30 日的最高连板数 (市场温度计)
    out["STREAK_MAX_30"] = pd.Series(streak, index=out.index).rolling(30).max()

    # === 3. 炸板率 (游资情绪) ===
    # 近 20 日: 炸板次数 / 触及涨停次数
    touched_20 = touched_limit.rolling(20).sum()
    boom_20 = boom_ban.rolling(20).sum()
    out["BOOM_RATE_20"] = (boom_20 / (touched_20 + 0.01)).fillna(0)
    out["LU_COUNT_20"] = closed_limit.rolling(20).sum()
    out["LU_COUNT_60"] = closed_limit.rolling(60).sum()
    out["LD_COUNT_20"] = closed_dn.rolling(20).sum()

    # === 4. 涨停质量: 次日收益 ===
    # 涨停后 次日 收益 (shift 过去看): label-like 但不是 future leak
    # 这里用"过去涨停 次日实际表现"回望特征: 近 60 日所有涨停后次日平均涨幅
    # next_ret = ret1.shift(-1)  # 这是 future!
    # 改用: 最近一次涨停 距今天数 + 最近一次涨停后 T+1 的实际收益(已实现,不是 future)
    # next_day_ret[t] = ret1[t+1], 但我们记录到 t (涨停日), 特征在 t+1 才可用 → shift(-1) 是 ok
    # 避免 future leak: 只用 shift(+1) 的已实现数据
    # "上一次涨停后次日的涨幅" - 这已经是 past 数据
    prev_limit_next_day_ret = (ret1.shift(-1).where(closed_limit)).shift(1)  # 上次涨停后次日收益
    # 其实更简单: 用截至 T 的最近一次涨停事件的次日 ret (查到 T-1 完成的事件)
    # shift(-1) 把 T+1 的 ret 拉到 T 上, 再 shift(1) 就把 T 上的变成 T+1 的值
    # 这里发生未来函数风险, 用 fillna 0 处理
    out["LU_NEXT_RET_AVG"] = (
        ret1.shift(-1).where(closed_limit)
    ).rolling(60).mean().shift(1).fillna(0)  # shift(1) 避免 leak

    # === 5. 涨停 gap (次日开盘情绪代理) ===
    # 开盘 gap = open / prev_close - 1; 涨停后次日 gap 特别重要
    gap = open_ / prev_close - 1
    # 最近一次涨停后次日开盘 gap (shift(1) 避免 leak)
    out["LU_NEXT_GAP_AVG"] = (
        gap.where(closed_limit.shift(1).fillna(False).astype(bool))
    ).rolling(30).mean().shift(1).fillna(0)

    # === 6. 封板强度 (涨停日成交额 vs 均值) ===
    amt_ma20 = amt.rolling(20).mean()
    lu_amt_ratio = (amt.where(closed_limit) / amt_ma20).shift(1)  # 上次涨停日成交额比
    out["LU_AMT_STRENGTH"] = lu_amt_ratio.rolling(60).mean().fillna(0)

    # === 7. 距最近涨停天数 ===
    days_since_lu = np.full(len(close), 999)
    last = -1
    for i in range(len(close)):
        if lu[i]:
            last = i
        if last >= 0:
            days_since_lu[i] = i - last
    out["DAYS_SINCE_LU"] = days_since_lu

    # === 8. 启动期特征 ===
    # 过去 60 日成交额极值压缩后突破 (典型游资启动前特征)
    amt_std_60 = amt.rolling(60).std()
    amt_mean_60 = amt.rolling(60).mean()
    amt_cv = amt_std_60 / (amt_mean_60 + 1)  # 变异系数
    # 变异系数低 (压缩) → 未来突破概率高
    out["AMT_COMPRESS_60"] = -amt_cv  # 负号: 越压缩越好
    # 近 5 日成交额 vs 近 60 日, 放量启动
    out["AMT_BREAKOUT"] = amt.rolling(5).mean() / (amt_mean_60 + 1)

    return out


def compute_limit_alpha(daily_df: pd.DataFrame) -> pd.DataFrame:
    daily_df = daily_df.copy()
    daily_df["date"] = pd.to_datetime(daily_df["date"])
    pieces = []
    for code, g in daily_df.groupby("code"):
        f = _per_stock_limit(g, str(code))
        f["code"] = code
        pieces.append(f.reset_index())
    if not pieces:
        return pd.DataFrame()
    full = pd.concat(pieces, ignore_index=True)
    return full.set_index(["date", "code"]).sort_index()


LIMIT_FACTOR_NAMES = [
    "LU_FLAG", "LD_FLAG", "BOOM_BAN_FLAG", "ONE_WORD_FLAG",
    "STREAK_UP", "STREAK_MAX_30",
    "BOOM_RATE_20", "LU_COUNT_20", "LU_COUNT_60", "LD_COUNT_20",
    "LU_NEXT_RET_AVG", "LU_NEXT_GAP_AVG", "LU_AMT_STRENGTH",
    "DAYS_SINCE_LU", "AMT_COMPRESS_60", "AMT_BREAKOUT",
]
