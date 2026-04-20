"""东财分钟 K adapter - 支撑 #6 分时犹豫度因子.

接口与 em_direct.fetch_daily 同源, 差别在 klt 参数:
    1 = 1 分钟 (仅最近几天)
    5 = 5 分钟 (最近 ~1-2 月)
    15 = 15 分钟
    30 = 30 分钟 (较长历史)
    60 = 60 分钟 (最长历史)

建议:
    - 研究/回测: klt=30 (8 根/天, 覆盖 1+ 年, 信号足够)
    - 实盘近端: klt=5 (48 根/天)

输出列: date, time, open, close, high, low, volume, amount, code
"""
from __future__ import annotations

import time
from typing import Optional

import pandas as pd
import requests


_HDRS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}


def _em_secid(code: str) -> str:
    code = str(code).zfill(6)
    return f"1.{code}" if code.startswith("6") else f"0.{code}"


def fetch_minute_kline(
    code: str, klt: int = 30,
    start: str = "20240101", end: str = "20261231",
    adjust: str = "qfq", retries: int = 2, timeout: int = 10,
) -> pd.DataFrame:
    """拉单只股票分钟 K.

    Args:
        klt: 1/5/15/30/60
        adjust: qfq/hfq/空
    """
    fqt_map = {"qfq": 1, "hfq": 2, "": 0}
    fqt = fqt_map.get(adjust, 1)
    secid = _em_secid(code)

    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "secid": secid,
        "ut": "fa5fd1943c7b386f172d6893dbfba10b",
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": klt, "fqt": fqt,
        "beg": start, "end": end, "lmt": 10000,
    }
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=_HDRS, timeout=timeout)
            data = r.json()
            if not data.get("data") or not data["data"].get("klines"):
                return pd.DataFrame()
        except Exception:
            if attempt == retries - 1:
                return pd.DataFrame()
            time.sleep(2 ** attempt)
            continue
        rows = []
        for line in data["data"]["klines"]:
            parts = line.split(",")
            # datetime 字段: "2024-12-01 09:35" 格式
            dt = parts[0]
            if " " in dt:
                d, t = dt.split(" ", 1)
            else:
                d, t = dt, "00:00"
            rows.append({
                "date": d, "time": t,
                "open": float(parts[1]), "close": float(parts[2]),
                "high": float(parts[3]), "low": float(parts[4]),
                "volume": int(parts[5]), "amount": float(parts[6]),
                "pct_chg": float(parts[8]) if len(parts) > 8 else 0.0,
            })
        df = pd.DataFrame(rows)
        df["code"] = str(code).zfill(6)
        return df
    return pd.DataFrame()


def bulk_fetch_minutes(
    codes: list[str], klt: int = 30,
    start: str = "20240101", end: str = "20261231",
    sleep_ms: int = 60, progress: bool = True,
) -> pd.DataFrame:
    """批量拉分钟 K.

    Note: 30min × 500 股 × 1 年 ≈ 96 万行, parquet 压缩后 ~20MB.
    """
    rows = []
    n = len(codes)
    for i, c in enumerate(codes):
        df = fetch_minute_kline(c, klt=klt, start=start, end=end)
        if not df.empty:
            rows.append(df)
        if progress and (i + 1) % 20 == 0:
            print(f"  分钟 K 进度 {i+1}/{n}  累计 {sum(len(x) for x in rows)} 行")
        time.sleep(sleep_ms / 1000)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    df = fetch_minute_kline("300750", klt=30, start="20250101", end="20260420")
    print(f"300750 30min K: {df.shape}")
    if not df.empty:
        print(df.tail(3))
        print(f"  日期范围: {df['date'].min()} → {df['date'].max()}")
        print(f"  每日 K 数: {df.groupby('date').size().median()}")
