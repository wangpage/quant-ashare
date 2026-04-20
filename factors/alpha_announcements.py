"""公告事件因子 - 一次 load 同时支撑 #1 时间结构 / #2 失败事件 / #3 股东减持.

#1 时间结构: 发布时段 (盘前/盘中/盘后/深夜), 周几, burst 密度, 周五晚规避信号
#2 失败事件: 终止重组/撤回定增/问询函回复 → T+1~T+5 错杀修复
#3 股东减持微结构: 大宗 vs 集中竞价, 节奏, 参与方类型 (控股 / ESOP / VC)

输入:
    ann_df: data_adapter.announcements.fetch_announcements_range 返回的宽表
    trading_dates: 交易日索引 (作为 panel 坐标轴)

输出:
    MultiIndex (date, code), ~20+ factors, 0 表示"无事件"
"""
from __future__ import annotations

import sys
from pathlib import Path

# 顶级 import 保护: 被当脚本跑时允许从仓库根 import
if __name__ == "__main__" or "data_adapter" not in sys.modules:
    _ROOT = Path(__file__).resolve().parent.parent
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from data_adapter.announcements import classify_title

EPS = 1e-12


def _classify_all(ann_df: pd.DataFrame) -> pd.DataFrame:
    """把 title 展开成 flag 列."""
    flags = ann_df["title"].apply(classify_title)
    fdf = pd.DataFrame(list(flags))
    return pd.concat([ann_df.reset_index(drop=True), fdf], axis=1)


# ------------------------------------------------------------------
#  每股每日汇总
# ------------------------------------------------------------------
def _daily_aggregate(ann_df: pd.DataFrame) -> pd.DataFrame:
    """把 multi-publication-per-day 汇总到 (date, code) 粒度."""
    df = _classify_all(ann_df)
    if df.empty:
        return pd.DataFrame()

    # 使用 publish_date (真实发布日), 盘后公告落在下一交易日影响, 由下游处理
    df["pdate"] = df["publish_date"]

    agg = df.groupby(["pdate", "code"]).agg(
        # #1 时间结构
        n_ann=("title", "count"),
        min_slot=("publish_slot", "min"),
        max_slot=("publish_slot", "max"),
        avg_slot=("publish_slot", "mean"),
        earliest_hour=("publish_hour", "min"),
        latest_hour=("publish_hour", "max"),
        weekday=("publish_weekday", "first"),
        # #2 失败事件
        n_fail=("is_fail", "sum"),
        n_terminate=("is_terminate", "sum"),
        n_withdraw=("is_withdraw", "sum"),
        n_inquiry=("is_inquiry", "sum"),
        # #3 股东行为
        n_reduce=("is_reduce", "sum"),
        n_increase=("is_increase", "sum"),
        n_block=("is_block_trade", "sum"),
        n_cent=("is_cent_compet", "sum"),
        n_esop=("is_esop", "sum"),
        n_control=("is_control", "sum"),
    ).reset_index().rename(columns={"pdate": "date"})
    agg["date"] = pd.to_datetime(agg["date"])
    return agg


# ------------------------------------------------------------------
#  对齐到 trading_dates - 盘后公告算下一交易日
# ------------------------------------------------------------------
def _align_to_trading(agg: pd.DataFrame,
                      trading_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """盘后公告 (slot>=3, 15:00 后) 的影响归到下一交易日.

    简化做法: 对于 max_slot>=3 的记录, date 推到下一交易日.
    这样可以避免 lookahead (不会把收盘后公告用在当日策略上).
    """
    agg = agg.copy()
    td_sorted = trading_dates.sort_values()
    # 盘后的 flag
    post_close = agg["max_slot"] >= 3
    # 找每个 date 的下一 trading date
    def _next_td(d):
        idx = td_sorted.searchsorted(d, side="right")
        return td_sorted[idx] if idx < len(td_sorted) else pd.NaT
    agg.loc[post_close, "date"] = agg.loc[post_close, "date"].apply(_next_td)
    # 合并同一 (date, code) 可能已存在当日盘中公告
    agg = agg.dropna(subset=["date"])
    merged = agg.groupby(["date", "code"]).agg({
        "n_ann": "sum",
        "min_slot": "min", "max_slot": "max", "avg_slot": "mean",
        "earliest_hour": "min", "latest_hour": "max",
        "weekday": "first",
        "n_fail": "sum", "n_terminate": "sum", "n_withdraw": "sum", "n_inquiry": "sum",
        "n_reduce": "sum", "n_increase": "sum",
        "n_block": "sum", "n_cent": "sum",
        "n_esop": "sum", "n_control": "sum",
    }).reset_index()
    return merged


# ------------------------------------------------------------------
#  Rolling feature builder: 对齐到 (trading_dates × all_codes) panel
# ------------------------------------------------------------------
def _build_panel(events: pd.DataFrame,
                 trading_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """对每个出现过事件的股票, 生成 rolling panel."""
    if events.empty:
        return pd.DataFrame()
    pieces = []
    for code, g in events.groupby("code"):
        g = g.sort_values("date").set_index("date")
        g = g.reindex(trading_dates, fill_value=0)
        out = pd.DataFrame(index=trading_dates)

        # === #1 时间结构 ===
        out["ANN_COUNT_5"]     = g["n_ann"].rolling(5).sum()
        out["ANN_COUNT_20"]    = g["n_ann"].rolling(20).sum()
        # 盘后密集度 (slot>=3 的次数) / 总公告数
        post_cnt = (g["max_slot"] >= 3).astype(int) * g["n_ann"]
        out["ANN_POST_RATIO_20"] = (post_cnt.rolling(20).sum() /
                                    (g["n_ann"].rolling(20).sum() + EPS))
        # 深夜公告 (slot=4) 20 日次数 (高 → 紧急事件 → 偏利空)
        night_cnt = (g["max_slot"] == 4).astype(int) * (g["n_ann"] > 0)
        out["ANN_NIGHT_20"] = -night_cnt.rolling(20).sum()
        # 周五晚发布密度 (weekday==4 且 slot>=3)
        friday_night = ((g["weekday"] == 4) & (g["max_slot"] >= 3)).astype(int)
        out["ANN_FRIDAY_NIGHT_20"] = -friday_night.rolling(20).sum()  # 负: 规避
        # burst: 连续 3+ 日有公告
        has_ann = (g["n_ann"] > 0).astype(int)
        # 简易 burst 代理: 过去 5 日累计公告数 >= 3
        out["ANN_BURST_5"] = has_ann.rolling(5).sum()

        # === #2 失败事件 ===
        out["FAIL_FLAG_5"]  = (g["n_fail"].rolling(5).sum() > 0).astype(int)
        out["FAIL_FLAG_20"] = (g["n_fail"].rolling(20).sum() > 0).astype(int)
        out["INQUIRY_20"]   = g["n_inquiry"].rolling(20).sum()
        out["TERMINATE_20"] = g["n_terminate"].rolling(20).sum()
        out["WITHDRAW_20"]  = g["n_withdraw"].rolling(20).sum()
        # 失败事件后的恢复窗 (T+1 ~ T+10 是反弹期; 值 = 近 10 日失败计数, 越高越像错杀机会)
        out["FAIL_RECOVERY_10"] = g["n_fail"].rolling(10).sum()

        # === #3 股东行为 ===
        out["REDUCE_20"]   = -g["n_reduce"].rolling(20).sum()    # 减持 = 利空
        out["INCREASE_20"] =  g["n_increase"].rolling(20).sum()  # 增持 = 利多
        out["BLOCK_20"]    = g["n_block"].rolling(20).sum()      # 大宗交易 (可能有溢价承接)
        out["CENT_COMPET_20"] = -g["n_cent"].rolling(20).sum()   # 集中竞价减持 = 隐性出货
        out["ESOP_20"]     = -g["n_esop"].rolling(20).sum()      # ESOP 减持 (情绪影响)
        out["CONTROL_REDUCE_20"] = -((g["n_control"] > 0) &
                                     (g["n_reduce"] > 0)).astype(int).rolling(20).sum()
        # 减持节奏: 连续减持天数 (T 时刻算) - 大股东"每天卖一点" 模式
        is_reducing = (g["n_reduce"] > 0).astype(int)
        grp = (is_reducing != is_reducing.shift()).cumsum()
        out["REDUCE_STREAK"] = -(is_reducing * is_reducing.groupby(grp).cumcount().add(1))
        # 净增减持 (近 60 日)
        out["NET_HOLDING_60"] = (g["n_increase"] - g["n_reduce"]).rolling(60).sum()

        out["code"] = code
        pieces.append(out.reset_index().rename(columns={"index": "date"}))

    if not pieces:
        return pd.DataFrame()
    panel = (pd.concat(pieces, ignore_index=True)
               .set_index(["date", "code"]).sort_index())
    return panel.fillna(0)


# ------------------------------------------------------------------
#  主入口
# ------------------------------------------------------------------
def compute_announcement_alpha(
    ann_df: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """一次性算三族公告因子.

    Args:
        ann_df: data_adapter.announcements.fetch_announcements_range 返回
        trading_dates: 交易日
    """
    if ann_df.empty:
        return pd.DataFrame()
    agg = _daily_aggregate(ann_df)
    agg = _align_to_trading(agg, trading_dates)
    return _build_panel(agg, trading_dates)


ANNOUNCEMENT_FACTOR_NAMES = [
    # #1 时间结构
    "ANN_COUNT_5", "ANN_COUNT_20",
    "ANN_POST_RATIO_20", "ANN_NIGHT_20",
    "ANN_FRIDAY_NIGHT_20", "ANN_BURST_5",
    # #2 失败事件
    "FAIL_FLAG_5", "FAIL_FLAG_20",
    "INQUIRY_20", "TERMINATE_20", "WITHDRAW_20",
    "FAIL_RECOVERY_10",
    # #3 股东
    "REDUCE_20", "INCREASE_20", "BLOCK_20",
    "CENT_COMPET_20", "ESOP_20", "CONTROL_REDUCE_20",
    "REDUCE_STREAK", "NET_HOLDING_60",
]


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from data_adapter.announcements import fetch_announcements_day

    # 小样本测试
    df = fetch_announcements_day("2025-04-17")
    print(f"公告原始: {len(df)} 条")
    # 构造 publish_time 等字段 (fetch_announcements_day 只返回 raw, 需在 _range 里补)
    # 这里手动加
    df["publish_time"] = pd.to_datetime(
        df["eiTime"].str.replace(r":\d+$", "", regex=True), errors="coerce")
    df["publish_date"] = df["publish_time"].dt.normalize()
    df["publish_hour"] = df["publish_time"].dt.hour
    df["publish_minute"] = df["publish_time"].dt.minute
    df["publish_weekday"] = df["publish_time"].dt.weekday
    def _slot(h):
        if pd.isna(h): return 3
        if h < 9: return 0
        if h < 12: return 1
        if h < 13: return 2
        if h < 15: return 1
        if h < 22: return 3
        return 4
    df["publish_slot"] = df["publish_hour"].apply(_slot)

    trading_dates = pd.DatetimeIndex(pd.date_range("2025-04-01", "2025-04-30", freq="B"))
    panel = compute_announcement_alpha(df, trading_dates)
    print(f"因子 panel shape={panel.shape}")
    print(f"  cols={list(panel.columns)}")
    # 非零看一下
    nz = panel[panel["ANN_COUNT_5"] > 0].head(5)
    print("\n有事件样本:")
    print(nz.head())
