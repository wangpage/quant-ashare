"""龙虎榜 (Longhubang) 数据适配器 - A股特色 alpha 源.

东财 datacenter 接口稳定, 字段含:
    - TRADE_DATE / SECURITY_CODE / CHANGE_RATE / TURNOVERRATE
    - BILLBOARD_NET_AMT: 龙虎榜净买入
    - EXPLAIN: 上榜原因 (含"机构专用"/游资营业部名)
    - D1/D5/D10/D20/D30 后续 adj 收益

用途: 识别"被游资盯上"+"机构进驻"的股票, 经典事件驱动 alpha.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

_HDRS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Referer": "https://data.eastmoney.com/",
}
_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"


def fetch_lhb_page(start_date: str, end_date: str,
                   page: int = 1, page_size: int = 200) -> list[dict]:
    """拉一页龙虎榜. date 格式 2024-12-20."""
    params = {
        "sortColumns": "TRADE_DATE,SECURITY_CODE",
        "sortTypes": "-1,1",
        "pageSize": page_size, "pageNumber": page,
        "reportName": "RPT_DAILYBILLBOARD_DETAILSNEW",
        "columns": "ALL", "source": "WEB",
        "filter": f"(TRADE_DATE>='{start_date}')(TRADE_DATE<='{end_date}')",
    }
    try:
        r = requests.get(_URL, params=params, headers=_HDRS, timeout=15)
        d = r.json()
        if not d.get("success") and not d.get("result"):
            return []
        return d.get("result", {}).get("data", []) or []
    except Exception as e:
        print(f"  lhb page {page} 失败: {e}")
        return []


def fetch_lhb_range(start: str, end: str,
                    cache_path: Path | None = None) -> pd.DataFrame:
    """拉指定区间龙虎榜. 自动分页 + 缓存.

    Args:
        start/end: YYYYMMDD
        cache_path: 可选缓存, 跳过重复请求
    """
    s = f"{start[:4]}-{start[4:6]}-{start[6:]}"
    e = f"{end[:4]}-{end[4:6]}-{end[6:]}"

    if cache_path and cache_path.exists():
        df = pd.read_parquet(cache_path)
        print(f"  lhb 缓存命中 {len(df)} 条")
        return df

    rows = []
    page = 1
    while True:
        chunk = fetch_lhb_page(s, e, page)
        if not chunk:
            break
        rows.extend(chunk)
        print(f"  lhb 分页 {page} 累计 {len(rows)}")
        if len(chunk) < 200:
            break
        page += 1
        time.sleep(0.3)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["TRADE_DATE"] = pd.to_datetime(df["TRADE_DATE"])
    df["code"] = df["SECURITY_CODE"].astype(str).str.zfill(6)

    # 数值列
    num_cols = ["BILLBOARD_NET_AMT", "BILLBOARD_BUY_AMT", "BILLBOARD_SELL_AMT",
                "DEAL_AMOUNT_RATIO", "TURNOVERRATE", "CHANGE_RATE",
                "FREE_MARKET_CAP", "CLOSE_PRICE",
                "D1_CLOSE_ADJCHRATE", "D5_CLOSE_ADJCHRATE",
                "D10_CLOSE_ADJCHRATE"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 衍生字段
    df["is_jigou"] = df["EXPLAIN"].str.contains("机构", na=False).astype(int)
    # 净买入占流通市值比 (龙虎榜强度)
    df["nb_ratio"] = df["BILLBOARD_NET_AMT"] / (df["FREE_MARKET_CAP"] + 1)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path)
        print(f"  lhb 缓存已写 {cache_path}")

    return df


def _load_taxonomy() -> pd.DataFrame | None:
    """加载 Qwen 标注的龙虎榜 taxonomy (B2 产物)."""
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "cache" / "lhb_taxonomy.parquet"
    if p.exists():
        return pd.read_parquet(p).set_index("template")
    return None


def _normalize_expl(s: str) -> str:
    import re
    if not isinstance(s, str):
        return "UNKNOWN"
    s = re.sub(r"\d+\.?\d*%", "X%", s)
    s = re.sub(r"成功率X%", "成功率", s)
    s = re.sub(r"\d+家", "N家", s)
    return s.strip() or "UNKNOWN"


def build_lhb_features(
    lhb_df: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """把龙虎榜流水转成每日每股特征面板.

    基础因子 (不依赖 taxonomy):
        LHB_FLAG_5 / LHB_FLAG_10 / LHB_NB_20 / LHB_JIGOU_20 / LHB_COUNT_60

    Qwen 标注因子 (需 cache/lhb_taxonomy.parquet):
        LHB_DIR_20: 20 日方向加权分 (sum of direction × quality)
        LHB_JIGOU_BUY_20 / LHB_JIGOU_SELL_20
        LHB_YOUZI_BUY_20 / LHB_YOUZI_SELL_20
        LHB_SMART_MONEY_20: 机构+游资净方向 (过滤散户噪声)
        LHB_QUALITY_20:    加权平均 quality
    """
    if lhb_df.empty:
        return pd.DataFrame()

    # 基础聚合
    daily = lhb_df.groupby(["TRADE_DATE", "code"]).agg(
        net_amt=("BILLBOARD_NET_AMT", "sum"),
        nb_ratio=("nb_ratio", "sum"),
        is_jigou=("is_jigou", "max"),
        count=("code", "size"),
    ).reset_index().rename(columns={"TRADE_DATE": "date"})

    flag_daily = (daily.pivot_table(
        index="date", columns="code", values="count", fill_value=0
    ) > 0).astype(int).reindex(trading_dates, fill_value=0)
    nb_daily = daily.pivot_table(
        index="date", columns="code", values="nb_ratio", fill_value=0
    ).reindex(trading_dates, fill_value=0)
    jg_daily = daily.pivot_table(
        index="date", columns="code", values="is_jigou", fill_value=0
    ).reindex(trading_dates, fill_value=0)

    feat = {
        "LHB_FLAG_5":   flag_daily.rolling(5).max(),
        "LHB_FLAG_10":  flag_daily.rolling(10).max(),
        "LHB_NB_20":    nb_daily.rolling(20).sum(),
        "LHB_JIGOU_20": jg_daily.rolling(20).sum(),
        "LHB_COUNT_60": flag_daily.rolling(60).sum(),
    }

    # === B2: Qwen taxonomy 因子 ===
    tax = _load_taxonomy()
    if tax is not None:
        lhb2 = lhb_df.copy()
        lhb2["templ"] = lhb2["EXPLAIN"].apply(_normalize_expl)
        lhb2 = lhb2.merge(
            tax[["direction", "player", "quality"]],
            left_on="templ", right_index=True, how="left",
        )
        lhb2["direction"] = lhb2["direction"].fillna(0)
        lhb2["player"] = lhb2["player"].fillna("未知")
        lhb2["quality"] = lhb2["quality"].fillna(2).astype(float)

        # dir × quality 加权分
        lhb2["dir_q"] = lhb2["direction"] * lhb2["quality"]

        def pivot(flag_col: str, fillv=0):
            agg = lhb2.groupby(["TRADE_DATE", "code"])[flag_col].sum().reset_index()
            agg = agg.rename(columns={"TRADE_DATE": "date"})
            return agg.pivot_table(index="date", columns="code",
                                    values=flag_col, fill_value=fillv
                                    ).reindex(trading_dates, fill_value=fillv)

        dir_q_daily = pivot("dir_q")

        # 机构买入/卖出
        jb = lhb2[(lhb2["player"] == "机构") & (lhb2["direction"] > 0)]
        js = lhb2[(lhb2["player"] == "机构") & (lhb2["direction"] < 0)]
        yb = lhb2[(lhb2["player"] == "游资") & (lhb2["direction"] > 0)]
        ys = lhb2[(lhb2["player"] == "游资") & (lhb2["direction"] < 0)]

        def count_pivot(sub: pd.DataFrame) -> pd.DataFrame:
            if sub.empty:
                return pd.DataFrame(0, index=trading_dates,
                                     columns=flag_daily.columns)
            g = sub.groupby(["TRADE_DATE", "code"]).size().reset_index(name="n")
            g = g.rename(columns={"TRADE_DATE": "date"})
            return g.pivot_table(index="date", columns="code",
                                  values="n", fill_value=0
                                  ).reindex(trading_dates, fill_value=0)

        jb_d = count_pivot(jb)
        js_d = count_pivot(js)
        yb_d = count_pivot(yb)
        ys_d = count_pivot(ys)

        # 对齐 columns (有些股可能只在一部分 sub 中出现)
        all_cols = flag_daily.columns
        for name, frame in [("jb", jb_d), ("js", js_d), ("yb", yb_d), ("ys", ys_d),
                             ("dir_q", dir_q_daily)]:
            if name == "jb":
                jb_d = frame.reindex(columns=all_cols, fill_value=0)
            elif name == "js":
                js_d = frame.reindex(columns=all_cols, fill_value=0)
            elif name == "yb":
                yb_d = frame.reindex(columns=all_cols, fill_value=0)
            elif name == "ys":
                ys_d = frame.reindex(columns=all_cols, fill_value=0)
            else:
                dir_q_daily = frame.reindex(columns=all_cols, fill_value=0)

        feat["LHB_DIR_20"]         = dir_q_daily.rolling(20).sum()
        feat["LHB_JIGOU_BUY_20"]   = jb_d.rolling(20).sum()
        feat["LHB_JIGOU_SELL_20"]  = js_d.rolling(20).sum()
        feat["LHB_YOUZI_BUY_20"]   = yb_d.rolling(20).sum()
        feat["LHB_YOUZI_SELL_20"]  = ys_d.rolling(20).sum()
        feat["LHB_SMART_MONEY_20"] = (
            jb_d.rolling(20).sum() + yb_d.rolling(20).sum()
            - js_d.rolling(20).sum() - ys_d.rolling(20).sum()
        )
        # 加权平均 quality = dir_q / count (除零保护)
        ct_daily = pivot("direction").abs()  # 有方向的次数
        feat["LHB_QUALITY_20"] = (
            dir_q_daily.rolling(20).sum().abs()
            / (ct_daily.rolling(20).sum() + 1e-9)
        )

    pieces = []
    for name, wide in feat.items():
        long = wide.stack().to_frame(name)
        pieces.append(long)
    out = pd.concat(pieces, axis=1)
    out.index.names = ["date", "code"]
    return out.fillna(0)


# 基础龙虎榜因子 (B1)
LHB_FACTOR_NAMES = [
    "LHB_FLAG_5", "LHB_FLAG_10", "LHB_NB_20",
    "LHB_JIGOU_20", "LHB_COUNT_60",
]
# B2 Qwen 加工因子
LHB_B2_FACTOR_NAMES = [
    "LHB_DIR_20", "LHB_JIGOU_BUY_20", "LHB_JIGOU_SELL_20",
    "LHB_YOUZI_BUY_20", "LHB_YOUZI_SELL_20",
    "LHB_SMART_MONEY_20", "LHB_QUALITY_20",
]


if __name__ == "__main__":
    df = fetch_lhb_range("20240101", "20240131")
    print(f"拉取 {len(df)} 条, 涉及 {df['code'].nunique()} 只")
    print(df[["TRADE_DATE", "code", "CHANGE_RATE", "BILLBOARD_NET_AMT",
              "D1_CLOSE_ADJCHRATE", "D5_CLOSE_ADJCHRATE", "EXPLAIN"]].head(10))
