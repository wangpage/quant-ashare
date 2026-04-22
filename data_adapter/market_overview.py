"""市场全景 fetcher: 大盘指数 + 北向资金 + 板块涨幅榜.

给 analyst 层提供"盘后收盘"的宏观上下文。每个 fetch 失败/返回空返回 None,
不抛异常, 让分析师简报可以降级输出"⚠️ 大盘数据暂缺"。

优先用新浪接口(更稳), 东财接口作备选。

落盘格式: cache/market_overview_{YYYY-MM-DD}.parquet (可选,失败不写)
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import akshare as ak
import pandas as pd

from utils.config import PROJECT_ROOT
from utils.logger import logger

_CACHE = PROJECT_ROOT / "cache"
_CACHE.mkdir(exist_ok=True)


def fetch_index_snapshot() -> Optional[pd.DataFrame]:
    """三大指数今日收盘 + 涨跌幅 (新浪 API, 更稳定).

    Returns:
        DataFrame[code, name, close, pct_chg] 或 None.
    """
    try:
        df = ak.stock_zh_index_spot_sina()
        wanted = {
            "sh000001": "上证指数",
            "sz399001": "深证成指",
            "sz399006": "创业板指",
        }
        hit = df[df["代码"].astype(str).isin(wanted.keys())]
        if hit.empty:
            logger.warning("market_overview: 三大指数一个都没匹配到")
            return None
        result = []
        for _, row in hit.iterrows():
            code = str(row["代码"])
            result.append({
                "code": code,
                "name": wanted[code],
                "close": float(row.get("最新价", 0) or 0),
                "pct_chg": float(row.get("涨跌幅", 0) or 0),
            })
        return pd.DataFrame(result)
    except Exception as e:
        logger.warning(f"market_overview.fetch_index_snapshot 失败: {e}")
        return None


def fetch_northbound_flow() -> Optional[dict]:
    """北向资金今日净流入(亿元).

    东财收盘汇总接口,数据在交易日内抽风时常返回 0 或空,
    那种情况返回 None 让简报标注"未更新"。

    Returns:
        {"net_inflow_yi": float, "date": str} 或 None.
    """
    try:
        df = ak.stock_hsgt_fund_flow_summary_em()
        if df is None or df.empty:
            return None
        north = df[df["资金方向"].astype(str) == "北向"]
        if north.empty:
            return None
        # 成交净买额是"亿元"为单位
        total = 0.0
        for v in north["成交净买额"]:
            if pd.notna(v):
                total += float(v)
        # 收盘前或未更新都会是 0.0, 此时不误导用户
        if abs(total) < 1e-6:
            logger.info("market_overview: 北向成交净买额为 0, 视作未更新")
            return None
        return {
            "net_inflow_yi": round(total, 2),
            "date": datetime.now().strftime("%Y-%m-%d"),
        }
    except Exception as e:
        logger.warning(f"market_overview.fetch_northbound_flow 失败: {e}")
        return None


def fetch_sector_rank_top5() -> Optional[pd.DataFrame]:
    """行业板块涨幅榜 Top 5 (sina 优先, em 备选).

    Returns:
        DataFrame[name, pct_chg] 或 None.
    """
    # 新浪行业
    try:
        df = ak.stock_sector_spot(indicator="新浪行业")
        if df is not None and not df.empty:
            # 涨跌幅列在 sina 接口是字符串 "10.000", 转 float
            df = df.copy()
            df["涨跌幅_f"] = pd.to_numeric(df["涨跌幅"], errors="coerce")
            df = df.dropna(subset=["涨跌幅_f"])
            top = df.nlargest(5, "涨跌幅_f")
            return pd.DataFrame({
                "name": top["板块"].astype(str).values,
                "pct_chg": top["涨跌幅_f"].round(2).values,
            })
    except Exception as e:
        logger.warning(f"market_overview.sina 板块失败: {e}, 尝试东财")

    # 东财备选
    try:
        df = ak.stock_board_industry_name_em()
        if df is None or df.empty:
            return None
        top = df.nlargest(5, "涨跌幅")
        return pd.DataFrame({
            "name": top["板块名称"].astype(str).values,
            "pct_chg": top["涨跌幅"].astype(float).round(2).values,
        })
    except Exception as e:
        logger.warning(f"market_overview.em 板块失败: {e}")
        return None


def fetch_all(persist: bool = True) -> dict:
    """一次性拉齐三项, 返回 dict. 单项失败只在对应 key 返回 None。

    Args:
        persist: 是否落盘 cache/market_overview_{date}.parquet

    Returns:
        {"indices": DataFrame|None, "northbound": dict|None, "sectors": DataFrame|None, "date": str}
    """
    result = {
        "indices": fetch_index_snapshot(),
        "northbound": fetch_northbound_flow(),
        "sectors": fetch_sector_rank_top5(),
        "date": datetime.now().strftime("%Y-%m-%d"),
    }
    if persist:
        try:
            _persist(result)
        except Exception as e:
            logger.warning(f"market_overview.persist 失败: {e}")
    return result


def _persist(snap: dict) -> None:
    date = snap["date"]
    rows = []
    if snap["indices"] is not None:
        for _, r in snap["indices"].iterrows():
            rows.append({"kind": "index", "name": r["name"], "code": r["code"],
                         "value": r["close"], "pct_chg": r["pct_chg"]})
    if snap["sectors"] is not None:
        for _, r in snap["sectors"].iterrows():
            rows.append({"kind": "sector", "name": r["name"], "code": "",
                         "value": None, "pct_chg": r["pct_chg"]})
    if snap["northbound"] is not None:
        rows.append({"kind": "northbound", "name": "北向净流入", "code": "",
                     "value": snap["northbound"]["net_inflow_yi"], "pct_chg": None})
    if not rows:
        return
    df = pd.DataFrame(rows)
    path = _CACHE / f"market_overview_{date}.parquet"
    df.to_parquet(path, index=False)


if __name__ == "__main__":
    snap = fetch_all(persist=False)
    print("indices:", snap["indices"])
    print("northbound:", snap["northbound"])
    print("sectors:", snap["sectors"])
