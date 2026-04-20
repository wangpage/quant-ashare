"""高管/大股东增减持数据 - 东财 datacenter.

强 alpha 源:
    - 减持 = 负面信号 (高管看空)
    - 增持 = 正面信号 (高管看多)
    - 成本均价 vs 市价: 增持成本线支撑效应

接口: RPT_EXECUTIVE_HOLD_DETAILS
关键字段:
    CHANGE_DATE: 变动日期
    SECURITY_CODE: 股票代码
    CHANGE_AMOUNT: 变动金额(元, 负数=减持)
    CHANGE_SHARES: 变动股数(负数=减持)
    CHANGE_RATIO: 占总股本比例
    AVERAGE_PRICE: 均价
    HOLD_TYPE: 持股类型 (个人/机构)
    PERSON_NAME / POSITION_NAME: 高管名 / 职位
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

_HDRS = {"User-Agent": "Mozilla/5.0", "Referer": "https://data.eastmoney.com/"}
_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"


def fetch_insider_page(start: str, end: str,
                        page: int = 1, page_size: int = 200) -> list[dict]:
    """拉一页增减持数据. date 格式 2024-12-01."""
    params = {
        "sortColumns": "CHANGE_DATE,SECURITY_CODE",
        "sortTypes": "-1,1",
        "pageSize": page_size, "pageNumber": page,
        "reportName": "RPT_EXECUTIVE_HOLD_DETAILS",
        "columns": "ALL", "source": "WEB",
        "filter": f"(CHANGE_DATE>='{start}')(CHANGE_DATE<='{end}')",
    }
    try:
        r = requests.get(_URL, params=params, headers=_HDRS, timeout=15)
        d = r.json()
        return d.get("result", {}).get("data", []) or []
    except Exception as e:
        print(f"  insider page {page} 失败: {e}")
        return []


def fetch_insider_range(start: str, end: str,
                         cache_path: Path | None = None) -> pd.DataFrame:
    """拉区间增减持, 自动分页 + 缓存."""
    if cache_path and cache_path.exists():
        df = pd.read_parquet(cache_path)
        print(f"  insider 缓存命中 {len(df)} 条")
        return df

    s = f"{start[:4]}-{start[4:6]}-{start[6:]}"
    e = f"{end[:4]}-{end[4:6]}-{end[6:]}"

    rows = []
    page = 1
    while True:
        chunk = fetch_insider_page(s, e, page)
        if not chunk:
            break
        rows.extend(chunk)
        if page % 10 == 0:
            print(f"  insider 分页 {page} 累计 {len(rows)}")
        if len(chunk) < 200:
            break
        page += 1
        time.sleep(0.25)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["CHANGE_DATE"] = pd.to_datetime(df["CHANGE_DATE"])
    df["code"] = df["SECURITY_CODE"].astype(str).str.zfill(6)
    for c in ["CHANGE_AMOUNT", "CHANGE_SHARES", "CHANGE_RATIO", "AVERAGE_PRICE",
              "BEGIN_HOLD_NUM", "END_HOLD_NUM"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path)
        print(f"  insider 缓存已写 {cache_path}")
    return df


def build_insider_features(
    insider_df: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """面板因子: 每日每股的增减持信号.

    INSIDER_NET_AMT_20:  20 日净增减持金额 (亿元)
    INSIDER_NET_RATIO_60: 60 日累计净变动占流通股比例
    INSIDER_REDUCE_FLAG_30: 近 30 日是否有减持事件
    INSIDER_ADD_FLAG_30:    近 30 日是否有增持事件
    INSIDER_SENTIMENT_20:   20 日加权情绪分 (减持多=负, 增持多=正)
    """
    if insider_df.empty:
        return pd.DataFrame()

    df = insider_df.copy()
    df["date"] = df["CHANGE_DATE"]
    # 方向: 减持为负, 增持为正
    df["sign"] = df["CHANGE_AMOUNT"].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    df["abs_amt_yi"] = df["CHANGE_AMOUNT"].abs() / 1e8

    daily = df.groupby(["date", "code"]).agg(
        net_amt_yi=("CHANGE_AMOUNT", lambda s: s.sum() / 1e8),
        net_ratio=("CHANGE_RATIO", "sum"),
        n_add=("sign", lambda s: int((s > 0).sum())),
        n_reduce=("sign", lambda s: int((s < 0).sum())),
    ).reset_index()

    def pivot(col, fillv=0):
        return daily.pivot_table(index="date", columns="code",
                                  values=col, fill_value=fillv
                                  ).reindex(trading_dates, fill_value=fillv)

    net_amt = pivot("net_amt_yi")
    net_ratio = pivot("net_ratio")
    n_add = pivot("n_add")
    n_reduce = pivot("n_reduce")

    feat = {
        "INSIDER_NET_AMT_20":   net_amt.rolling(20).sum(),
        "INSIDER_NET_AMT_60":   net_amt.rolling(60).sum(),
        "INSIDER_NET_RATIO_60": net_ratio.rolling(60).sum(),
        "INSIDER_ADD_30":       (n_add.rolling(30).sum() > 0).astype(int),
        "INSIDER_REDUCE_30":    (n_reduce.rolling(30).sum() > 0).astype(int),
        "INSIDER_SENTIMENT_20": (n_add - n_reduce).rolling(20).sum(),
    }

    pieces = []
    for name, wide in feat.items():
        long = wide.stack().to_frame(name)
        pieces.append(long)
    out = pd.concat(pieces, axis=1)
    out.index.names = ["date", "code"]
    return out.fillna(0)


INSIDER_FACTOR_NAMES = [
    "INSIDER_NET_AMT_20", "INSIDER_NET_AMT_60", "INSIDER_NET_RATIO_60",
    "INSIDER_ADD_30", "INSIDER_REDUCE_30", "INSIDER_SENTIMENT_20",
]


if __name__ == "__main__":
    df = fetch_insider_range("20240101", "20240131")
    print(f"样本: {len(df)} 条")
    if len(df):
        print(df[["CHANGE_DATE", "code", "CHANGE_AMOUNT", "CHANGE_RATIO",
                  "POSITION_NAME", "CHANGE_REASON"]].head(10))
