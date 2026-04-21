"""分钟线数据适配器 - 东财 push2his.

东财 5min/15min/30min K 线接口:
    - 单次 lmt 10000, 但 5min 实际只返回最近 ~30 天
    - 分段请求: 每段 30 天, 多次拉 + 合并
    - 字段: date/open/close/high/low/volume/amount/pct/change/...
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
_HDRS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Referer": "https://quote.eastmoney.com/",
}


def _secid(code: str) -> str:
    code = str(code).zfill(6)
    return f"1.{code}" if code.startswith("6") else f"0.{code}"


def fetch_minute_segment(
    code: str, klt: int, beg: str, end: str,
    retries: int = 3, timeout: int = 12,
) -> pd.DataFrame:
    """单段分钟线. klt: 5/15/30/60. date YYYYMMDD."""
    params = {
        "secid": _secid(code),
        "ut": "fa5fd1943c7b386f172d6893dbfba10b",
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": klt, "fqt": 1, "beg": beg, "end": end, "lmt": 10000,
    }
    for i in range(retries):
        try:
            r = requests.get(_URL, params=params, headers=_HDRS, timeout=timeout)
            data = r.json().get("data", {})
            klines = data.get("klines") or []
            if not klines:
                return pd.DataFrame()
            rows = []
            for line in klines:
                p = line.split(",")
                rows.append({
                    "datetime": p[0],
                    "open": float(p[1]), "close": float(p[2]),
                    "high": float(p[3]), "low": float(p[4]),
                    "volume": int(float(p[5])), "amount": float(p[6]),
                    "amplitude": float(p[7]), "pct_chg": float(p[8]),
                    "chg_amt": float(p[9]), "turnover": float(p[10]),
                })
            df = pd.DataFrame(rows)
            df["code"] = str(code).zfill(6)
            return df
        except Exception:
            if i == retries - 1:
                return pd.DataFrame()
            time.sleep(2 ** i * 0.5)
    return pd.DataFrame()


def fetch_minute_range(
    code: str, klt: int, start: str, end: str,
    segment_days: int = 25, sleep_ms: int = 50,
) -> pd.DataFrame:
    """拉长区间分钟线, 自动分段. 5min 每次限 30 天, 分段 25 天留余量."""
    s = pd.Timestamp(start[:4] + "-" + start[4:6] + "-" + start[6:])
    e = pd.Timestamp(end[:4] + "-" + end[4:6] + "-" + end[6:])
    segs = []
    cur = s
    while cur < e:
        seg_end = min(cur + pd.Timedelta(days=segment_days), e)
        beg_s = cur.strftime("%Y%m%d")
        end_s = seg_end.strftime("%Y%m%d")
        df = fetch_minute_segment(code, klt, beg_s, end_s)
        if not df.empty:
            segs.append(df)
        cur = seg_end + pd.Timedelta(days=1)
        time.sleep(sleep_ms / 1000)
    if not segs:
        return pd.DataFrame()
    full = pd.concat(segs, ignore_index=True).drop_duplicates(
        subset=["datetime", "code"]
    )
    full["datetime"] = pd.to_datetime(full["datetime"])
    return full.sort_values("datetime").reset_index(drop=True)


def bulk_fetch_minute(
    codes: list[str], klt: int, start: str, end: str,
    sleep_ms: int = 40, progress: bool = True,
    segment_days: int = 25,
) -> pd.DataFrame:
    """批量拉 分钟线. klt=5 推荐."""
    all_data = []
    fail = []
    t0 = time.time()
    for i, c in enumerate(codes):
        df = fetch_minute_range(c, klt, start, end,
                                segment_days=segment_days, sleep_ms=sleep_ms)
        if df.empty:
            fail.append(c)
        else:
            all_data.append(df)
        if progress and (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(codes) - i - 1)
            print(f"  进度 {i+1}/{len(codes)}  成功 {len(all_data)}  "
                  f"失败 {len(fail)}  ETA {eta:.0f}s")
    print(f"  完成 成功={len(all_data)} 失败={len(fail)} 耗时 {time.time()-t0:.0f}s")
    if not all_data:
        return pd.DataFrame()
    return pd.concat(all_data, ignore_index=True)


if __name__ == "__main__":
    df = fetch_minute_range("300750", 5, "20260101", "20260320")
    print(f"宁德时代 5min: {len(df)} 根")
    print(df.head())
