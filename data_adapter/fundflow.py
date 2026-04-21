"""主力资金流适配器 - "大鳄进出" 的日频精华.

字段 (akshare stock_individual_fund_flow):
    - 超大单净流入 (>100 万单笔) - 机构/大户
    - 大单净流入 (20-100 万) - 中等资金
    - 中单 (5-20 万) - 活跃散户
    - 小单 (<5 万) - 散户
    - 净占比 = 净流入 / 当日成交额

学术/行业证据:
    - 超大单净流入 5 日累计 vs 未来 5 日收益的 IC ≈ 0.04-0.08
    - 超大单 + 股价下跌 = 机构吸筹 (经典反转信号)
    - 小单大量流出 + 超大单流入 = "韭菜割完了"
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import akshare as ak


def fetch_fund_flow_one(code: str, retries: int = 2) -> pd.DataFrame:
    """拉单只股近 ~120 日资金流. 返回字段清理后的英文列名."""
    market = "sh" if code.startswith("6") else "sz"
    for i in range(retries):
        try:
            df = ak.stock_individual_fund_flow(stock=code, market=market)
            if df.empty:
                return pd.DataFrame()
            df = df.rename(columns={
                "日期": "date", "收盘价": "close", "涨跌幅": "pct_chg",
                "主力净流入-净额": "main_net",
                "主力净流入-净占比": "main_net_pct",
                "超大单净流入-净额": "sup_large_net",
                "超大单净流入-净占比": "sup_large_net_pct",
                "大单净流入-净额": "large_net",
                "大单净流入-净占比": "large_net_pct",
                "中单净流入-净额": "medium_net",
                "中单净流入-净占比": "medium_net_pct",
                "小单净流入-净额": "small_net",
                "小单净流入-净占比": "small_net_pct",
            })
            df["code"] = str(code).zfill(6)
            df["date"] = pd.to_datetime(df["date"])
            # 单位: 净额 元, 占比 %
            return df
        except Exception as e:
            if i == retries - 1:
                return pd.DataFrame()
            time.sleep(1.0)
    return pd.DataFrame()


def bulk_fetch_fund_flow(
    codes: list[str], sleep_ms: int = 400,
    cache_path: Path | None = None,
) -> pd.DataFrame:
    """批量拉资金流 + 缓存. 注意 akshare 有限速."""
    if cache_path and cache_path.exists():
        df = pd.read_parquet(cache_path)
        print(f"  fund_flow 缓存命中 {len(df)} 行")
        return df

    rows = []
    fail = []
    t0 = time.time()
    for i, c in enumerate(codes):
        df = fetch_fund_flow_one(c)
        if df.empty:
            fail.append(c)
        else:
            rows.append(df)
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(codes) - i - 1)
            print(f"  进度 {i+1}/{len(codes)}  成功 {len(rows)}  "
                  f"失败 {len(fail)}  ETA {eta:.0f}s")
        time.sleep(sleep_ms / 1000)

    print(f"  完成 成功={len(rows)} 失败={len(fail)} 耗时 {time.time()-t0:.0f}s")
    if not rows:
        return pd.DataFrame()
    full = pd.concat(rows, ignore_index=True)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        full.to_parquet(cache_path)
        print(f"  缓存已写 {cache_path}")
    return full


def build_fundflow_features(
    ff_df: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """把资金流日志 → 每日每股的因子面板.

    核心 12 因子:
        SUPER_NET_5/20/60:   超大单 5/20/60 日累计净流入占比
        LARGE_NET_5/20:      大单
        SMART_MONEY_5/20:    超大+大 (聪明钱)
        RETAIL_OUT_5/20:     中+小 反向 (散户跑路)
        SUP_NET_Z_20:        超大单净流入 20 日 z-score (突变信号)
        BIG_VS_SMALL:        (超大+大) - (中+小) 占比差
        DIV_ACCUM:           连续流入天数 (持续性)
    """
    if ff_df.empty:
        return pd.DataFrame()

    df = ff_df.copy()
    df["super_pct"]  = df["sup_large_net_pct"]
    df["large_pct"]  = df["large_net_pct"]
    df["medium_pct"] = df["medium_net_pct"]
    df["small_pct"]  = df["small_net_pct"]
    df["smart_pct"]  = df["super_pct"] + df["large_pct"]
    df["retail_pct"] = df["medium_pct"] + df["small_pct"]

    # 宽表 (date × code) 用于滚动
    def pivot(col):
        w = df.pivot_table(index="date", columns="code",
                            values=col, fill_value=0)
        return w.reindex(trading_dates, fill_value=0)

    sup = pivot("super_pct")
    lar = pivot("large_pct")
    sma = pivot("smart_pct")
    ret = pivot("retail_pct")

    feat = {
        "SUPER_NET_5":  sup.rolling(5, min_periods=3).sum(),
        "SUPER_NET_20": sup.rolling(20, min_periods=10).sum(),
        "SUPER_NET_60": sup.rolling(60, min_periods=30).sum(),
        "LARGE_NET_5":  lar.rolling(5, min_periods=3).sum(),
        "LARGE_NET_20": lar.rolling(20, min_periods=10).sum(),
        "SMART_MONEY_5":  sma.rolling(5, min_periods=3).sum(),
        "SMART_MONEY_20": sma.rolling(20, min_periods=10).sum(),
        "RETAIL_OUT_5":  -ret.rolling(5, min_periods=3).sum(),
        "RETAIL_OUT_20": -ret.rolling(20, min_periods=10).sum(),
    }
    # 超大单突变 z-score (20 日)
    sup_mean = sup.rolling(20, min_periods=10).mean()
    sup_std = sup.rolling(20, min_periods=10).std()
    feat["SUP_NET_Z_20"] = ((sup - sup_mean) / (sup_std + 1e-9)).clip(-3, 3)
    # 大资金 vs 散户差 5 日
    feat["BIG_VS_SMALL_5"] = sma.rolling(5).sum() - ret.rolling(5).sum()
    # 超大单连续流入天数
    pos_days = (sup > 0).astype(int)
    feat["SUP_DAYS_5"] = pos_days.rolling(5, min_periods=3).sum()

    pieces = []
    for name, wide in feat.items():
        long = wide.stack().to_frame(name)
        pieces.append(long)
    out = pd.concat(pieces, axis=1)
    out.index.names = ["date", "code"]
    return out.fillna(0)


FUNDFLOW_FACTOR_NAMES = [
    "SUPER_NET_5", "SUPER_NET_20", "SUPER_NET_60",
    "LARGE_NET_5", "LARGE_NET_20",
    "SMART_MONEY_5", "SMART_MONEY_20",
    "RETAIL_OUT_5", "RETAIL_OUT_20",
    "SUP_NET_Z_20", "BIG_VS_SMALL_5", "SUP_DAYS_5",
]


if __name__ == "__main__":
    df = fetch_fund_flow_one("300750")
    print(f"宁德时代资金流: {len(df)} 行")
    print(df[["date", "close", "main_net_pct",
              "sup_large_net_pct", "large_net_pct", "small_net_pct"]].tail(5))
