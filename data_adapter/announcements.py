"""东财公告中心 adapter - 支撑 #1 时间结构 / #2 失败事件 / #3 股东减持 三族因子.

接口: https://np-anotice-stock.eastmoney.com/api/security/ann
    分页: page_size (<=50) + page_index
    排序: sr=-1 (倒序, 最新先)
    时间: begin_time / end_time (YYYY-MM-DD)
    过滤: ann_type=A (A股)

关键字段:
    eiTime       真实发布时间 (YYYY-MM-DD HH:MM:SS:ms), 不是 notice_date
    codes[0].stock_code  股票代码
    title / title_ch     公告标题
    columns              公告分类列表 (column_code + column_name)

缓存: parquet (date / code / time / title / columns_str)
"""
from __future__ import annotations

import time
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import requests

_URL = "https://np-anotice-stock.eastmoney.com/api/security/ann"
_HDRS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}


def fetch_announcements_day(
    date: str, page_size: int = 100, max_pages: int = 100,
    timeout: int = 15, sleep_ms: int = 200,
) -> pd.DataFrame:
    """拉指定一天的全部 A股公告.

    Args:
        date: YYYY-MM-DD, begin_time = end_time = 该日
    """
    rows = []
    for page in range(1, max_pages + 1):
        params = {
            "sr": -1, "page_size": page_size, "page_index": page,
            "ann_type": "A", "client_source": "web",
            "begin_time": date, "end_time": date,
        }
        try:
            r = requests.get(_URL, params=params, headers=_HDRS, timeout=timeout)
            d = r.json()
        except Exception as e:
            print(f"    page{page} 失败: {e}")
            break
        items = (d.get("data") or {}).get("list") or []
        if not items:
            break
        for it in items:
            codes = it.get("codes") or []
            if not codes:
                continue
            stock_code = codes[0].get("stock_code")
            if not stock_code or not str(stock_code).isdigit():
                continue
            columns = it.get("columns") or []
            rows.append({
                "date": date,
                "code": str(stock_code).zfill(6),
                "eiTime": it.get("eiTime"),
                "title": it.get("title") or "",
                "columns_str": "|".join(c.get("column_name", "") for c in columns),
                "art_code": it.get("art_code"),
            })
        if len(items) < page_size:
            break
        time.sleep(sleep_ms / 1000)
    return pd.DataFrame(rows)


def fetch_announcements_range(
    start: str, end: str,
    codes_filter: set | None = None,
    cache_path: Path | None = None,
    skip_weekends: bool = True,
) -> pd.DataFrame:
    """拉区间公告. start/end = YYYYMMDD.

    Args:
        codes_filter: 如提供, 仅保留这些股票的公告 (减小 I/O)
        cache_path: parquet 缓存
        skip_weekends: 跳过周六日 (仍可能有公告但极少)
    """
    if cache_path and cache_path.exists():
        df = pd.read_parquet(cache_path)
        print(f"  公告缓存命中 {len(df)} 条 ({cache_path.name})")
        return df

    start_dt = datetime.strptime(start, "%Y%m%d")
    end_dt = datetime.strptime(end, "%Y%m%d")
    all_rows = []
    day = start_dt
    while day <= end_dt:
        if skip_weekends and day.weekday() >= 5:
            day += timedelta(days=1)
            continue
        ds = day.strftime("%Y-%m-%d")
        df = fetch_announcements_day(ds)
        if codes_filter is not None and not df.empty:
            df = df[df["code"].isin(codes_filter)]
        if not df.empty:
            all_rows.append(df)
        if day.day == 1:                           # 月初打点
            print(f"  {ds} 累计 {sum(len(x) for x in all_rows)} 条")
        day += timedelta(days=1)

    if not all_rows:
        return pd.DataFrame()
    out = pd.concat(all_rows, ignore_index=True)
    # 解析 eiTime → pandas datetime + 衍生字段
    out["publish_time"] = pd.to_datetime(
        out["eiTime"].str.replace(r":\d+$", "", regex=True),
        errors="coerce",
    )
    out["publish_date"] = out["publish_time"].dt.normalize()
    out["publish_hour"] = out["publish_time"].dt.hour
    out["publish_minute"] = out["publish_time"].dt.minute
    out["publish_weekday"] = out["publish_time"].dt.weekday
    # 发布时段分类: 0=盘前 1=盘中 2=午间 3=盘后 4=深夜
    def _slot(h):
        if h < 9: return 0
        if h < 12: return 1
        if h < 13: return 2
        if h < 15: return 1
        if h < 22: return 3
        return 4
    out["publish_slot"] = out["publish_hour"].apply(_slot)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(cache_path)
        print(f"  公告缓存已写 {cache_path}")
    return out


def fetch_announcements_by_stock(
    stock_code: str, start: str, end: str,
    page_size: int = 100, max_pages: int = 100,
    timeout: int = 15, sleep_ms: int = 100,
) -> pd.DataFrame:
    """按单只股票拉区间公告 (服务端 stock_list 参数过滤, 比按日拉快 20x).

    Args:
        stock_code: 6 位代码
        start/end: YYYYMMDD
    """
    s = f"{start[:4]}-{start[4:6]}-{start[6:]}"
    e = f"{end[:4]}-{end[4:6]}-{end[6:]}"
    rows = []
    for page in range(1, max_pages + 1):
        params = {
            "sr": -1, "page_size": page_size, "page_index": page,
            "ann_type": "A", "client_source": "web",
            "stock_list": stock_code,
            "begin_time": s, "end_time": e,
        }
        try:
            r = requests.get(_URL, params=params, headers=_HDRS, timeout=timeout)
            d = r.json()
        except Exception:
            break
        items = (d.get("data") or {}).get("list") or []
        if not items:
            break
        for it in items:
            codes = it.get("codes") or []
            if not codes:
                continue
            columns = it.get("columns") or []
            rows.append({
                "code": stock_code.zfill(6),
                "eiTime": it.get("eiTime"),
                "title": it.get("title") or "",
                "columns_str": "|".join(c.get("column_name", "") for c in columns),
                "art_code": it.get("art_code"),
            })
        if len(items) < page_size:
            break
        time.sleep(sleep_ms / 1000)
    return pd.DataFrame(rows)


def fetch_announcements_bulk_by_codes(
    codes: list[str], start: str, end: str,
    cache_path: Path | None = None,
    progress: bool = True, sleep_ms: int = 150,
) -> pd.DataFrame:
    """对股票池批量拉公告. 每股单次请求 (stock_list 参数).

    对 500 股 × 3 年 ≈ 5-10 分钟, 首次建缓存后增量可秒级.
    """
    if cache_path and cache_path.exists():
        df = pd.read_parquet(cache_path)
        print(f"  公告缓存命中 {len(df)} 条 ({cache_path.name})")
        return df

    all_rows = []
    n = len(codes)
    for i, c in enumerate(codes):
        df = fetch_announcements_by_stock(c, start, end)
        if not df.empty:
            all_rows.append(df)
        if progress and (i + 1) % 20 == 0:
            print(f"  进度 {i+1}/{n}  累计 {sum(len(x) for x in all_rows)} 条")
        time.sleep(sleep_ms / 1000)

    if not all_rows:
        return pd.DataFrame()
    out = pd.concat(all_rows, ignore_index=True)
    # 派生时间字段
    out["publish_time"] = pd.to_datetime(
        out["eiTime"].str.replace(r":\d+$", "", regex=True), errors="coerce")
    out["publish_date"] = out["publish_time"].dt.normalize()
    out["publish_hour"] = out["publish_time"].dt.hour
    out["publish_minute"] = out["publish_time"].dt.minute
    out["publish_weekday"] = out["publish_time"].dt.weekday
    def _slot(h):
        if pd.isna(h): return 3
        if h < 9: return 0
        if h < 12: return 1
        if h < 13: return 2
        if h < 15: return 1
        if h < 22: return 3
        return 4
    out["publish_slot"] = out["publish_hour"].apply(_slot)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(cache_path)
        print(f"  公告缓存已写 {cache_path} ({len(out)} 条)")
    return out


# ------------------------------------------------------------------
#  关键词分类 (供下游因子用)
# ------------------------------------------------------------------
# #2 失败事件关键词
FAIL_KEYWORDS = [
    "终止重大资产重组", "终止筹划重大资产", "终止收购", "终止资产",
    "撤回定增", "撤回申请", "撤回发行", "撤回材料",
    "不予核准", "否决", "未通过",
    "问询函", "关注函",
    "终止", "中止",
]

# #3 股东减持关键词 + 区分方式
SHAREHOLDER_REDUCE_KW = ["减持计划", "减持股份", "减持公告", "减持进展", "减持结果"]
SHAREHOLDER_INCREASE_KW = ["增持计划", "增持股份", "增持公告"]
BLOCK_TRADE_KW = ["大宗交易"]
CENT_COMPET_KW = ["集中竞价"]
ESOP_KW = ["员工持股", "限制性股票", "股权激励"]
CONTROL_HOLDER_KW = ["控股股东", "实际控制人"]


def classify_title(title: str) -> dict:
    """从标题抽取语义 flags (用于 #2/#3 因子)."""
    if not isinstance(title, str):
        return {}
    t = title.lower()
    out = {
        "is_fail": any(kw in title for kw in FAIL_KEYWORDS),
        "is_reduce": any(kw in title for kw in SHAREHOLDER_REDUCE_KW),
        "is_increase": any(kw in title for kw in SHAREHOLDER_INCREASE_KW),
        "is_block_trade": any(kw in title for kw in BLOCK_TRADE_KW),
        "is_cent_compet": any(kw in title for kw in CENT_COMPET_KW),
        "is_esop": any(kw in title for kw in ESOP_KW),
        "is_control": any(kw in title for kw in CONTROL_HOLDER_KW),
        "is_inquiry": "问询函" in title or "关注函" in title,
        "is_terminate": "终止" in title,
        "is_withdraw": "撤回" in title,
    }
    return out


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    # 小规模验证: 拉一个交易日
    df = fetch_announcements_day("2025-04-17")
    print(f"2025-04-17 公告 {len(df)} 条")
    print(df[["code", "eiTime", "title", "columns_str"]].head())
    print()
    # 分类
    df["flags"] = df["title"].apply(classify_title)
    fail_n = df["flags"].apply(lambda d: d.get("is_fail", False)).sum()
    reduce_n = df["flags"].apply(lambda d: d.get("is_reduce", False)).sum()
    print(f"失败类事件: {fail_n} 条, 减持类: {reduce_n} 条")
    if fail_n > 0:
        idx = df["flags"].apply(lambda d: d.get("is_fail", False))
        print("\n失败类样例:")
        print(df[idx][["code", "eiTime", "title"]].head(5).to_string())
