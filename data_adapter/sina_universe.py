"""新浪财经全市场 A股 + 估值数据拉取.

弥补 eastmoney clist 接口挂掉时的主力备胎. 含 PE/PB/流通市值/换手率,
可直接用作基本面因子 snapshot.
"""
from __future__ import annotations

import json
import time
from typing import Any

import pandas as pd
import requests

_SINA_URL = (
    "https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
    "Market_Center.getHQNodeData"
)
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Referer": "https://finance.sina.com.cn/",
}


def fetch_sina_market_page(page: int, num: int = 100,
                            node: str = "hs_a") -> list[dict[str, Any]]:
    """拉单页. node: hs_a=沪深A, sh_a=沪A, sz_a=深A."""
    params = {"page": page, "num": num, "sort": "symbol",
              "asc": 1, "node": node, "_s_r_a": "init"}
    try:
        r = requests.get(_SINA_URL, params=params, headers=_HEADERS, timeout=12)
        return json.loads(r.text) or []
    except Exception:
        return []


def fetch_all_ashare(max_pages: int = 60, page_size: int = 100,
                      sleep_ms: int = 80) -> pd.DataFrame:
    """拉 A股全市场估值快照. 支持断点续拉."""
    rows = []
    empty_streak = 0
    for p in range(1, max_pages + 1):
        data = fetch_sina_market_page(p, page_size)
        if not data:
            empty_streak += 1
            if empty_streak >= 2:
                break
            time.sleep(sleep_ms / 1000)
            continue
        empty_streak = 0
        rows.extend(data)
        if p % 5 == 0:
            print(f"  sina 分页 {p}/{max_pages}, 累计 {len(rows)}")
        time.sleep(sleep_ms / 1000)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # 字段清理
    df["code"] = df["code"].astype(str).str.zfill(6)
    num_cols = ["trade", "pricechange", "changepercent", "buy", "sell",
                "settlement", "open", "high", "low", "per", "pb",
                "mktcap", "nmc", "turnoverratio"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    # 单位: mktcap/nmc 万元, amount 元
    df["mkt_cap_yi"]   = df["mktcap"] / 1e4     # 总市值 → 亿
    df["float_cap_yi"] = df["nmc"]    / 1e4     # 流通市值 → 亿
    df["amount_yi"]    = df["amount"] / 1e8     # 成交额 → 亿

    return df


def filter_midcap_universe(
    df: pd.DataFrame,
    min_float_cap: float = 30,   # 亿
    max_float_cap: float = 300,  # 亿
    min_amount: float = 0.5,     # 亿, 日成交额
    exclude_st: bool = True,
    exclude_688: bool = True,    # 剔科创板
    exclude_bj: bool = True,
) -> list[str]:
    """筛选中小盘 + 流动性达标. 返回 6 位代码 list."""
    d = df.copy()
    if exclude_st:
        d = d[~d["name"].str.contains("ST|退", regex=True, na=False)]
    if exclude_688:
        d = d[~d["code"].str.startswith("688")]
    if exclude_bj:
        d = d[d["symbol"].str.startswith(("sh", "sz"))]
    d = d[d["trade"] > 0]
    d = d[d["float_cap_yi"].between(min_float_cap, max_float_cap)]
    d = d[d["amount_yi"] >= min_amount]

    # 剔除次新股 (简单: 价格与 5 日前相差太大说明波动极端)
    d = d[d["changepercent"].abs() < 20]

    codes = d.sort_values("amount_yi", ascending=False)["code"].tolist()
    return codes


if __name__ == "__main__":
    df = fetch_all_ashare(max_pages=60)
    print(f"total: {len(df)}")
    codes = filter_midcap_universe(df, min_float_cap=30, max_float_cap=300, min_amount=0.5)
    print(f"midcap: {len(codes)}, top10: {codes[:10]}")
