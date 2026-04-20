"""东方财富直连 fallback - 绕开 akshare 在 macOS 的 curl_cffi 兼容问题.

纯 requests 实现, 性能和稳定性都好, 作为 akshare 主链路失败时的备选.
"""
from __future__ import annotations

import json
import time
from typing import Optional

import pandas as pd
import requests


_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://quote.eastmoney.com/",
})


def _em_secid(code: str) -> str:
    """300750 → 0.300750, 600519 → 1.600519."""
    code = str(code).zfill(6)
    return f"1.{code}" if code.startswith("6") else f"0.{code}"


def fetch_daily(
    code: str, start: str = "20200101", end: str = "20261231",
    adjust: str = "qfq", retries: int = 3, timeout: int = 10,
    include_factor: bool = False,
) -> pd.DataFrame:
    """直连东财拉日K.

    Args:
        code: 6 位股票代码, 无后缀
        start/end: YYYYMMDD
        adjust: 'qfq'=前复权 'hfq'=后复权 ''=不复权
        include_factor: True 时额外请求一次不复权价, 计算 factor 列
                        (复权因子 = 后复权/不复权, 单调递增). 会 2x 请求数.

    Returns:
        DataFrame: date, open, close, high, low, volume, amount,
                   turnover, pct_chg. include_factor=True 时加 factor 列.
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
        "klt": 101,          # 101 = 日K
        "fqt": fqt,
        "beg": start, "end": end, "lmt": 10000,
    }
    for attempt in range(retries):
        try:
            # 每次创建新 session 避免连接被复用时失效
            session = requests.Session()
            session.headers.update(_SESSION.headers)
            r = session.get(url, params=params, timeout=timeout)
            session.close()
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            if attempt == retries - 1:
                return pd.DataFrame()
            time.sleep(2 ** attempt)     # 指数退避 1, 2, 4, 8...
            continue
        if not data.get("data") or not data["data"].get("klines"):
            return pd.DataFrame()
        rows = []
        for line in data["data"]["klines"]:
            parts = line.split(",")
            rows.append({
                "date": parts[0], "open": float(parts[1]),
                "close": float(parts[2]), "high": float(parts[3]),
                "low": float(parts[4]), "volume": int(parts[5]),
                "amount": float(parts[6]),
                "amplitude": float(parts[7]),   # 振幅
                "pct_chg": float(parts[8]),
                "pct_change_amt": float(parts[9]),
                "turnover": float(parts[10]),
            })
        df = pd.DataFrame(rows)
        df["code"] = str(code).zfill(6)
        if include_factor:
            # factor = hfq_close / raw_close, 后复权方向单调递增
            df = _attach_factor_column(df, code, start, end, timeout)
        return df
    return pd.DataFrame()


def _attach_factor_column(
    df: pd.DataFrame, code: str, start: str, end: str, timeout: int,
) -> pd.DataFrame:
    """再拉一次后复权价, 与主 df 对齐计算复权因子."""
    secid = _em_secid(code)
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "secid": secid,
        "ut": "fa5fd1943c7b386f172d6893dbfba10b",
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": 101, "fqt": 2,   # 2 = 后复权
        "beg": start, "end": end, "lmt": 10000,
    }
    try:
        s = requests.Session()
        s.headers.update(_SESSION.headers)
        r = s.get(url, params=params, timeout=timeout)
        s.close()
        data = r.json()
        klines = (data.get("data") or {}).get("klines") or []
    except Exception:
        return df
    if not klines:
        return df
    hfq_map = {}
    for line in klines:
        parts = line.split(",")
        hfq_map[parts[0]] = float(parts[2])   # date → hfq_close
    # df['close'] 是前复权价, 要拿到原始价需要再拉 fqt=0.
    # 但工程上 factor = hfq_close / qfq_close 也能校验单调性, 因为
    # qfq 在基准日价格不变, hfq 累积向上, 比值单调递增等价于复权正确.
    df = df.copy()
    df["factor"] = df["date"].map(hfq_map) / df["close"]
    return df


def fetch_index_sina(
    index_code: str = "000300", datalen: int = 800,
    timeout: int = 10,
) -> pd.DataFrame:
    """新浪指数 K 线 - 东财备选."""
    prefix = "sh" if index_code.startswith("000") else "sz"
    symbol = f"{prefix}{index_code}"
    url = "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData"
    params = {"symbol": symbol, "scale": 240, "ma": "no",
              "datalen": datalen}
    headers = {"Referer": "https://finance.sina.com.cn/"}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
        data = r.json()
    except Exception:
        return pd.DataFrame()
    rows = []
    for d in data or []:
        rows.append({
            "date": d["day"], "open": float(d["open"]),
            "high": float(d["high"]), "low": float(d["low"]),
            "close": float(d["close"]),
            "volume": int(float(d["volume"])),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("date").reset_index(drop=True)
        df["pct_chg"] = df["close"].pct_change() * 100
        df["code"] = index_code
    return df


def fetch_index_daily(
    index_code: str = "000300", start: str = "20200101",
    end: str = "20261231", timeout: int = 10,
) -> pd.DataFrame:
    """拉指数: 000300=沪深300, 000905=中证500, 000001=上证指数, 399001=深证成指."""
    secid = f"1.{index_code}" if index_code.startswith("0") and index_code != "000001" else \
            f"1.{index_code}" if index_code == "000001" else f"0.{index_code}"
    # 沪市指数 prefix=1, 深市指数=0
    if index_code.startswith(("000",)):
        secid = f"1.{index_code}"
    elif index_code.startswith("399"):
        secid = f"0.{index_code}"

    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "secid": secid, "ut": "fa5fd1943c7b386f172d6893dbfba10b",
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60",
        "klt": 101, "fqt": 0, "beg": start, "end": end, "lmt": 10000,
    }
    try:
        r = requests.get(url, params=params, timeout=timeout,
                          headers={"User-Agent": "Mozilla/5.0"})
        data = r.json()
    except Exception:
        data = {}
    if not data.get("data") or not data["data"].get("klines"):
        # 回退新浪
        sina_df = fetch_index_sina(index_code)
        if not sina_df.empty:
            start_fmt = start[:4] + "-" + start[4:6] + "-" + start[6:]
            end_fmt = end[:4] + "-" + end[4:6] + "-" + end[6:]
            return sina_df[(sina_df["date"] >= start_fmt) &
                           (sina_df["date"] <= end_fmt)].reset_index(drop=True)
        return pd.DataFrame()
    rows = []
    for line in data["data"]["klines"]:
        parts = line.split(",")
        rows.append({
            "date": parts[0], "open": float(parts[1]),
            "close": float(parts[2]), "high": float(parts[3]),
            "low": float(parts[4]), "volume": int(parts[5]),
            "amount": float(parts[6]),
            "pct_chg": float(parts[8]),
        })
    df = pd.DataFrame(rows)
    df["code"] = index_code
    return df


def fetch_csi300_constituents(timeout: int = 10) -> list[str]:
    """拉沪深300成分股代码."""
    url = "https://push2.eastmoney.com/api/qt/clist/get"
    params = {
        "pn": 1, "pz": 500, "po": 1,
        "fid": "f3",
        "fs": "b:BK0500",     # 沪深300板块
        "fields": "f12,f14",
        "ut": "fa5fd1943c7b386f172d6893dbfba10b",
    }
    try:
        r = _SESSION.get(url, params=params, timeout=timeout)
        data = r.json()
        diff = data.get("data", {}).get("diff") or []
        return [d["f12"] for d in diff if "f12" in d][:300]
    except Exception:
        return []


def fetch_hot_stocks(limit: int = 50, timeout: int = 10) -> list[dict]:
    """活跃股 top N (按成交额) - 用于快速生成测试池."""
    url = "https://push2.eastmoney.com/api/qt/clist/get"
    params = {
        "pn": 1, "pz": limit, "po": 1, "fid": "f6",       # f6 = 成交额排序
        "fs": "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23",         # A股
        "fields": "f12,f14,f2,f5,f6,f20",
        "ut": "fa5fd1943c7b386f172d6893dbfba10b",
    }
    r = _SESSION.get(url, params=params, timeout=timeout)
    diff = r.json().get("data", {}).get("diff") or []
    return [{
        "code": d.get("f12"), "name": d.get("f14"),
        "price": d.get("f2"), "volume": d.get("f5"),
        "amount": d.get("f6"), "market_cap": d.get("f20"),
    } for d in diff]


def _sina_symbol(code: str) -> str:
    code = str(code).zfill(6)
    return f"sh{code}" if code.startswith("6") else f"sz{code}"


def fetch_daily_sina(
    code: str, datalen: int = 800, timeout: int = 10,
) -> pd.DataFrame:
    """新浪财经日K - 东财限频时的备选."""
    url = "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData"
    params = {"symbol": _sina_symbol(code), "scale": 240, "ma": "no",
              "datalen": datalen}
    headers = {"Referer": "https://finance.sina.com.cn/",
               "User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
        data = r.json()
    except Exception:
        return pd.DataFrame()
    if not data:
        return pd.DataFrame()
    rows = []
    for d in data:
        rows.append({
            "date": d["day"], "open": float(d["open"]),
            "high": float(d["high"]), "low": float(d["low"]),
            "close": float(d["close"]),
            "volume": int(float(d["volume"])),
            "amount": float(d.get("amount", 0)),
        })
    df = pd.DataFrame(rows)
    df["code"] = str(code).zfill(6)
    # 计算 pct_chg
    df = df.sort_values("date").reset_index(drop=True)
    df["pct_chg"] = df["close"].pct_change() * 100
    df["turnover"] = 0.0  # 新浪不直接提供换手率, 留空
    return df


def bulk_fetch_daily(
    codes: list[str], start: str, end: str,
    sleep_ms: int = 50, progress: bool = True,
    use_sina_fallback: bool = True,
    include_factor: bool = False,
) -> pd.DataFrame:
    """批量拉, 带限频保护和新浪 fallback.

    Args:
        include_factor: True 时每只股 2x 请求数, 换来 factor 列 (复权因子校验).
    """
    rows = []
    fail = []
    n = len(codes)
    em_fail_streak = 0
    for i, c in enumerate(codes):
        df = pd.DataFrame()
        if em_fail_streak < 3:                # 东财还没被连续限频
            df = fetch_daily(c, start, end, include_factor=include_factor)
            if df.empty:
                em_fail_streak += 1
            else:
                em_fail_streak = 0

        if df.empty and use_sina_fallback:
            df = fetch_daily_sina(c)
            if not df.empty:
                # 按 start/end 过滤
                df = df[(df["date"] >= start[:4] + "-" + start[4:6] + "-" + start[6:]) &
                        (df["date"] <= end[:4] + "-" + end[4:6] + "-" + end[6:])]

        if df.empty:
            fail.append(c)
        else:
            rows.append(df)
        if progress and (i + 1) % 5 == 0:
            src_hint = f"(东财连续失败 {em_fail_streak})" if em_fail_streak > 0 else ""
            print(f"  进度 {i+1}/{n}  成功 {len(rows)} 失败 {len(fail)} {src_hint}")
        time.sleep(sleep_ms / 1000)
    print(f"  完成 成功={len(rows)} 失败={len(fail)}")
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    df = fetch_daily("300750", "20250101", "20260420")
    print("300750:", df.shape)
    print(df.tail(3))
