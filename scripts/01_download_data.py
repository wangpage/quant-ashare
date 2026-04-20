"""Step 1: 从 akshare 拉取全市场日K线, 存入 SQLite."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_adapter import DataFetcher, get_stock_universe
from utils.logger import logger


def main(full: bool = False, codes: list[str] | None = None):
    fetcher = DataFetcher()

    # 刷新股票信息
    fetcher.fetch_stock_list()

    # 指数 - 回测基准
    fetcher.fetch_index("sh000300")
    fetcher.fetch_index("sh000905")
    fetcher.fetch_index("sh000016")

    if codes:
        target = codes
    elif full:
        target = get_stock_universe(refresh=True)
    else:
        # 默认只拉沪深300成分股, 便于快速跑通
        import akshare as ak
        try:
            df = ak.index_stock_cons_csindex(symbol="000300")
            col = "成分券代码" if "成分券代码" in df.columns else df.columns[0]
            target = df[col].astype(str).tolist()
        except Exception as e:
            logger.warning(f"沪深300 拉取失败, 回退到全市场: {e}")
            target = get_stock_universe(refresh=True)

    logger.info(f"目标股票数: {len(target)}")
    fetcher.fetch_daily_bars(target)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="全市场, 否则只拉沪深300")
    ap.add_argument("--codes", nargs="*", help="指定股票代码")
    args = ap.parse_args()
    main(full=args.full, codes=args.codes)
