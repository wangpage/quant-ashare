"""席位网络因子 - 从龙虎榜原始缓存挖掘游资圈 / 机构主导 / 一日游模式.

关键发现 (通过统计 lhb cache 解码):
    BUY_SEAT_NEW / SELL_SEAT_NEW 是 5 位字符串编码, 对应龙虎榜买/卖方前 5 个席位:
        '1' = 普通营业部 (游资/散户集中地, ~88%)
        '3' = 机构专用席位 (~12%)
        '*' = 缺失 (<0.3%)
    因此 BUY_SEAT_NEW = "11311" 表示 5 席位中第 3 个是机构, 其余 4 个是游资.

非共识洞察:
    1. 相同 pattern 重复出现 = "游资固定军团回归", 稳定性好
    2. pattern 每次都不同 = "散资接力 / 一日游", 次日高概率回落
    3. 多股共享同一 pattern = 同一游资圈在扫货 → 板块异动前兆
    4. 机构槽位数 + EXPLANATION 类型组合 → 买方性质量化

EXPLANATION 关键原因 (10+ 类):
    '涨幅偏离 7%'  (short-term breakout)
    '连续三个交易日涨幅 20%'  (multi-day squeeze)
    '日涨幅 15%' (ST 类型)
    '日换手率 20%/30%'  (流动性事件, 非价格)
    '跌幅偏离 7%'  (潜在抄底 / 避雷)

用法:
    from factors.seat_network import compute_seat_alpha, SEAT_FACTOR_NAMES
    feat = compute_seat_alpha(lhb_df, trading_dates)
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

EPS = 1e-12


# ------------------------------------------------------------------
#  底层: 单行 LHB 解码
# ------------------------------------------------------------------
def _count_digit(code: str, digit: str) -> int:
    """SEAT_NEW 编码里某个数字出现次数. None/nan 返回 0."""
    if not isinstance(code, str):
        return 0
    return code.count(digit)


def _is_price_up_reason(expl: str) -> int:
    if not isinstance(expl, str):
        return 0
    return int(("涨幅" in expl) and ("跌幅" not in expl))


def _is_price_down_reason(expl: str) -> int:
    if not isinstance(expl, str):
        return 0
    return int("跌幅" in expl)


def _is_turnover_reason(expl: str) -> int:
    if not isinstance(expl, str):
        return 0
    return int("换手率" in expl)


def _is_streak_reason(expl: str) -> int:
    if not isinstance(expl, str):
        return 0
    return int(("连续" in expl) and ("交易日" in expl))


_JIGOU_CNT_RE = re.compile(r"(\d+)家机构")


def _parse_jigou_count(expl: str) -> int:
    """从 EXPLAIN 文本抽取"N家机构"的 N. 买入正/卖出负 无法区分, 只统计总."""
    if not isinstance(expl, str):
        return 0
    m = _JIGOU_CNT_RE.search(expl)
    return int(m.group(1)) if m else 0


# ------------------------------------------------------------------
#  预处理: 把原始 LHB 流水转为"每股每日"汇总的 seat 事件表
# ------------------------------------------------------------------
def _preprocess_lhb(lhb_df: pd.DataFrame) -> pd.DataFrame:
    if lhb_df.empty:
        return pd.DataFrame()

    df = lhb_df.copy()
    df["TRADE_DATE"] = pd.to_datetime(df["TRADE_DATE"])
    df["code"] = df["SECURITY_CODE"].astype(str).str.zfill(6)

    # 席位编码衍生
    df["buy_seat"]  = df["BUY_SEAT_NEW"].astype(str)
    df["sell_seat"] = df["SELL_SEAT_NEW"].astype(str)
    df["n_jigou_buy"]  = df["buy_seat"].apply(lambda s: _count_digit(s, "3"))
    df["n_jigou_sell"] = df["sell_seat"].apply(lambda s: _count_digit(s, "3"))
    df["n_retail_buy"]  = df["buy_seat"].apply(lambda s: _count_digit(s, "1"))
    df["n_retail_sell"] = df["sell_seat"].apply(lambda s: _count_digit(s, "1"))

    # EXPLAIN / EXPLANATION 语义
    df["explain_jigou_cnt"] = df.get("EXPLAIN", pd.Series()).apply(_parse_jigou_count)
    reason = df.get("EXPLANATION", pd.Series([""] * len(df)))
    df["is_up_reason"]       = reason.apply(_is_price_up_reason)
    df["is_down_reason"]     = reason.apply(_is_price_down_reason)
    df["is_turnover_reason"] = reason.apply(_is_turnover_reason)
    df["is_streak_reason"]   = reason.apply(_is_streak_reason)

    # 净买入占流通市值比 (信号强度)
    if "BILLBOARD_NET_AMT" in df.columns and "FREE_MARKET_CAP" in df.columns:
        df["nb_ratio"] = (pd.to_numeric(df["BILLBOARD_NET_AMT"], errors="coerce") /
                          (pd.to_numeric(df["FREE_MARKET_CAP"], errors="coerce") + 1))
    else:
        df["nb_ratio"] = 0.0

    # 同一天同一股可能多次上榜 (不同 EXPLANATION), 汇总
    grp = df.groupby(["TRADE_DATE", "code"]).agg(
        buy_seat_primary=("buy_seat", "first"),
        sell_seat_primary=("sell_seat", "first"),
        n_jigou_buy=("n_jigou_buy", "max"),
        n_jigou_sell=("n_jigou_sell", "max"),
        n_retail_buy=("n_retail_buy", "max"),
        n_retail_sell=("n_retail_sell", "max"),
        explain_jigou_cnt=("explain_jigou_cnt", "max"),
        is_up_reason=("is_up_reason", "max"),
        is_down_reason=("is_down_reason", "max"),
        is_turnover_reason=("is_turnover_reason", "max"),
        is_streak_reason=("is_streak_reason", "max"),
        nb_ratio=("nb_ratio", "sum"),
        n_listings=("code", "size"),
    ).reset_index()
    grp = grp.rename(columns={"TRADE_DATE": "date"})
    return grp


# ------------------------------------------------------------------
#  席位共现图 (day-level, rolling 60d 窗口)
# ------------------------------------------------------------------
def _build_pattern_cooccurrence(events: pd.DataFrame,
                                window_days: int = 60) -> dict:
    """对每天, 计算"与本股共享相同 buy_seat pattern 的其他股数".

    做法: 对每个 date, 取过去 window_days 的事件, 构造
            pattern -> set(codes). 然后该股当日的 co-occur 规模 =
            max over 其 buy_seat_primary 对应的 set 大小 - 1.

    Returns:
        dict: (date, code) -> cooccur count
    """
    if events.empty:
        return {}
    events = events.sort_values("date")
    dates = events["date"].sort_values().unique()

    result: dict = {}
    from collections import defaultdict, deque
    # 维护一个滑动窗口, 按日期进入/退出
    pattern_to_codes: dict = defaultdict(lambda: defaultdict(int))   # pattern -> code -> count
    event_queue: deque = deque()                                      # (date, pattern, code)

    # 按日期分组迭代
    events_by_date = events.groupby("date", sort=True)
    for dt, day_events in events_by_date:
        # 先把窗口外的事件移除
        cutoff = dt - pd.Timedelta(days=window_days)
        while event_queue and event_queue[0][0] < cutoff:
            old_dt, old_pat, old_code = event_queue.popleft()
            pattern_to_codes[old_pat][old_code] -= 1
            if pattern_to_codes[old_pat][old_code] <= 0:
                del pattern_to_codes[old_pat][old_code]
                if not pattern_to_codes[old_pat]:
                    del pattern_to_codes[old_pat]
        # 先在加入今天之前查询 (避免自引用)
        for _, r in day_events.iterrows():
            pat = r["buy_seat_primary"]
            code = r["code"]
            co = len(pattern_to_codes.get(pat, {}))
            result[(dt, code)] = co
        # 然后把今天加入窗口
        for _, r in day_events.iterrows():
            pat = r["buy_seat_primary"]
            code = r["code"]
            event_queue.append((dt, pat, code))
            pattern_to_codes[pat][code] += 1
    return result


# ------------------------------------------------------------------
#  对每股构建时序特征 (rolling) 然后展成 panel
# ------------------------------------------------------------------
def _per_stock_seat_features(g: pd.DataFrame,
                             trading_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """g: 单股 events (已汇总), 按 date 排序."""
    g = g.sort_values("date").set_index("date")
    # 展到 trading_dates 上
    g = g.reindex(trading_dates)

    out = pd.DataFrame(index=trading_dates)

    # 事件 flag (60 日内是否上榜)
    has_event = g["n_listings"].fillna(0)
    out["SEAT_FLAG_10"] = (has_event.rolling(10).sum() > 0).astype(int)
    out["SEAT_FLAG_60"] = (has_event.rolling(60).sum() > 0).astype(int)

    # 机构净席位 (买 - 卖), 60 日窗口
    njb = g["n_jigou_buy"].fillna(0)
    njs = g["n_jigou_sell"].fillna(0)
    out["SEAT_NET_JIGOU_60"] = (njb - njs).rolling(60).sum()

    # 纯游资接力 (11111) 最近次数 → 一日游风险
    is_pure_hotmoney = ((g["buy_seat_primary"] == "11111") &
                       (g["n_jigou_buy"] == 0)).fillna(False).astype(int)
    out["SEAT_PURE_HOTMONEY_20"] = is_pure_hotmoney.rolling(20).sum()

    # 机构主导 (买方机构 >= 3)
    is_inst_lead = (g["n_jigou_buy"] >= 3).fillna(False).astype(int)
    out["SEAT_INST_LEAD_20"] = is_inst_lead.rolling(20).sum()

    # pattern 多样性 / 固定性 (rolling 60 日; object dtype 手动滚窗)
    from collections import Counter, deque
    patterns_list = g["buy_seat_primary"].tolist()
    diversity, persistence = [], []
    window: deque = deque()
    counter: Counter = Counter()
    for p in patterns_list:
        is_real = isinstance(p, str) and p not in ("nan", "None")
        window.append(p if is_real else None)
        if is_real:
            counter[p] += 1
        if len(window) > 60:
            old = window.popleft()
            if old is not None:
                counter[old] -= 1
                if counter[old] <= 0:
                    del counter[old]
        total = sum(counter.values())
        diversity.append(len(counter))
        persistence.append(max(counter.values()) / total if total > 0 else np.nan)
    out["SEAT_DIVERSITY_60"]   = pd.Series(diversity, index=g.index)
    out["SEAT_PERSISTENCE_60"] = pd.Series(persistence, index=g.index)

    # 上榜原因类型特征 (过去 20 日)
    out["SEAT_UP_REASON_20"]    = g["is_up_reason"].fillna(0).rolling(20).sum()
    out["SEAT_DOWN_REASON_20"]  = g["is_down_reason"].fillna(0).rolling(20).sum()
    out["SEAT_TURN_REASON_20"]  = g["is_turnover_reason"].fillna(0).rolling(20).sum()
    out["SEAT_STREAK_REASON_20"] = g["is_streak_reason"].fillna(0).rolling(20).sum()

    # 净买入强度 (60 日累积)
    out["SEAT_NB_RATIO_60"] = g["nb_ratio"].fillna(0).rolling(60).sum()

    return out


# ------------------------------------------------------------------
#  主入口
# ------------------------------------------------------------------
def compute_seat_alpha(
    lhb_df: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """从原始龙虎榜缓存构造席位网络因子面板.

    Args:
        lhb_df: 原始 LHB parquet (含 BUY_SEAT_NEW/SELL_SEAT_NEW/EXPLAIN 等列)
        trading_dates: 交易日索引 (用于 reindex, 填充无事件日)

    Returns:
        MultiIndex (date, code) 的因子面板, NaN 表示该股从未上榜
    """
    events = _preprocess_lhb(lhb_df)
    if events.empty:
        return pd.DataFrame()

    # 共现图 (全集级别)
    cooccur = _build_pattern_cooccurrence(events, window_days=60)

    # 逐股时序
    pieces = []
    for code, g in events.groupby("code"):
        feat = _per_stock_seat_features(g, trading_dates)
        feat["code"] = code
        pieces.append(feat.reset_index().rename(columns={"index": "date"}))
    if not pieces:
        return pd.DataFrame()
    panel = pd.concat(pieces, ignore_index=True).set_index(["date", "code"])

    # 共现图因子: 仅事件日有值, reindex 后 forward-fill 到下次事件前的 20 天
    cooccur_series = pd.Series({k: v for k, v in cooccur.items()}, name="SEAT_COOCCUR_60")
    if not cooccur_series.empty:
        cooccur_series.index = pd.MultiIndex.from_tuples(
            cooccur_series.index, names=["date", "code"])
        panel = panel.join(cooccur_series, how="left")
        # forward fill 20 天 (事件后的衰减)
        panel["SEAT_COOCCUR_60"] = (panel.groupby(level="code")["SEAT_COOCCUR_60"]
                                    .transform(lambda s: s.ffill(limit=20)))

    # 空值填 0 (从未上榜的股票 = 无事件信号)
    panel = panel.fillna(0).sort_index()

    return panel


SEAT_FACTOR_NAMES = [
    "SEAT_FLAG_10", "SEAT_FLAG_60",
    "SEAT_NET_JIGOU_60",
    "SEAT_PURE_HOTMONEY_20", "SEAT_INST_LEAD_20",
    "SEAT_DIVERSITY_60", "SEAT_PERSISTENCE_60",
    "SEAT_UP_REASON_20", "SEAT_DOWN_REASON_20",
    "SEAT_TURN_REASON_20", "SEAT_STREAK_REASON_20",
    "SEAT_NB_RATIO_60",
    "SEAT_COOCCUR_60",
]


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import time

    cache = Path(__file__).resolve().parent.parent / "cache"
    lhb = pd.read_parquet(cache / "lhb_20230101_20260420.parquet")
    kl = pd.read_parquet(cache / "kline_20230101_20260420_n200.parquet")
    print(f"lhb: {lhb.shape}, kline: {kl.shape}")

    trading_dates = pd.DatetimeIndex(sorted(pd.to_datetime(kl["date"]).unique()))
    print(f"交易日: {len(trading_dates)}")

    t0 = time.time()
    panel = compute_seat_alpha(lhb, trading_dates)
    print(f"耗时 {time.time()-t0:.1f}s, shape={panel.shape}")
    print(f"  cols={list(panel.columns)}")
    print(f"  nnz 占比: {(panel != 0).mean().to_dict()}")
    # 采样验证
    if not panel.empty:
        sample = panel[panel["SEAT_FLAG_10"] > 0].head(5)
        print("\n有事件样本:")
        print(sample)
