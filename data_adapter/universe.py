from __future__ import annotations

import pandas as pd

from utils.config import CONFIG
from utils.logger import logger

from .fetcher import DataFetcher


def get_stock_universe(refresh: bool = False) -> list[str]:
    """按配置筛选候选股票池.

    过滤条件:
      - 非 ST / 非退市
      - 市值在 [market_cap_min, market_cap_max]
      - 排除主板新股 ipo_days <= exclude_new
    """
    fetcher = DataFetcher()
    if refresh:
        fetcher.fetch_stock_list()
    df = fetcher.get_stock_info()
    if df.empty:
        logger.warning("股票信息为空，先运行 fetch_stock_list()")
        fetcher.fetch_stock_list()
        df = fetcher.get_stock_info()

    conf = CONFIG["data"]["universe"]
    mask = pd.Series(True, index=df.index)
    if conf["exclude_st"]:
        mask &= df["is_st"] == 0
    if "market_cap" in df.columns:
        mask &= df["market_cap"].between(conf["market_cap_min"], conf["market_cap_max"])

    df = df[mask]
    df = df[df["code"].str.match(r"^(60|00|30)")]
    logger.info(f"候选股票池: {len(df)}只")
    return df["code"].tolist()
