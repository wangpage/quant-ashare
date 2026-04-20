from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import akshare as ak
import pandas as pd
from tqdm import tqdm

from utils.config import CONFIG, PROJECT_ROOT
from utils.logger import logger


class DataFetcher:
    """akshare 数据抓取 + SQLite 本地缓存。

    表结构:
      daily_bars(code, date, open, high, low, close, volume, amount, turnover, pct_chg)
      stock_info(code, name, industry, market_cap, list_date, is_st)
      index_bars(code, date, open, high, low, close, volume, amount)
    """

    def __init__(self, db_path: str | Path | None = None):
        db_path = Path(db_path) if db_path else PROJECT_ROOT / CONFIG["data"]["db_path"].lstrip("./")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._init_schema()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self):
        with self._conn() as c:
            c.executescript(
                """
                CREATE TABLE IF NOT EXISTS daily_bars (
                    code TEXT,
                    date TEXT,
                    open REAL, high REAL, low REAL, close REAL,
                    volume REAL, amount REAL, turnover REAL, pct_chg REAL,
                    PRIMARY KEY (code, date)
                );
                CREATE INDEX IF NOT EXISTS idx_daily_date ON daily_bars(date);
                CREATE INDEX IF NOT EXISTS idx_daily_code ON daily_bars(code);

                CREATE TABLE IF NOT EXISTS stock_info (
                    code TEXT PRIMARY KEY,
                    name TEXT,
                    industry TEXT,
                    market_cap REAL,
                    list_date TEXT,
                    is_st INTEGER DEFAULT 0,
                    updated_at TEXT
                );

                CREATE TABLE IF NOT EXISTS index_bars (
                    code TEXT,
                    date TEXT,
                    open REAL, high REAL, low REAL, close REAL,
                    volume REAL, amount REAL,
                    PRIMARY KEY (code, date)
                );
                """
            )

    # -------- 股票池 --------
    def fetch_stock_list(self) -> pd.DataFrame:
        logger.info("拉取A股股票列表...")
        df = ak.stock_zh_a_spot_em()
        df = df[["代码", "名称", "总市值"]].rename(
            columns={"代码": "code", "名称": "name", "总市值": "market_cap"}
        )
        df["is_st"] = df["name"].str.contains("ST|退", regex=True, na=False).astype(int)
        df["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            info = ak.stock_individual_info_em  # placeholder
        except Exception:
            pass
        with self._conn() as conn:
            df.to_sql("stock_info_tmp", conn, if_exists="replace", index=False)
            conn.execute(
                """
                INSERT OR REPLACE INTO stock_info(code, name, market_cap, is_st, updated_at)
                SELECT code, name, market_cap, is_st, updated_at FROM stock_info_tmp
                """
            )
            conn.execute("DROP TABLE stock_info_tmp")
        logger.info(f"股票列表更新完成，共{len(df)}只")
        return df

    def get_stock_info(self) -> pd.DataFrame:
        with self._conn() as c:
            return pd.read_sql("SELECT * FROM stock_info", c)

    # -------- 日K线 --------
    def _fetch_daily_one(self, code: str, start: str, end: str) -> pd.DataFrame:
        try:
            df = ak.stock_zh_a_hist(
                symbol=code, period="daily",
                start_date=start.replace("-", ""), end_date=end.replace("-", ""),
                adjust="qfq",
            )
        except Exception as e:
            logger.warning(f"[{code}] 拉取失败: {e}")
            return pd.DataFrame()
        if df.empty:
            return df
        df = df.rename(
            columns={
                "日期": "date", "开盘": "open", "最高": "high", "最低": "low",
                "收盘": "close", "成交量": "volume", "成交额": "amount",
                "换手率": "turnover", "涨跌幅": "pct_chg",
            }
        )
        df["code"] = code
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        cols = ["code", "date", "open", "high", "low", "close",
                "volume", "amount", "turnover", "pct_chg"]
        return df[cols]

    def fetch_daily_bars(
        self,
        codes: Iterable[str],
        start: str | None = None,
        end: str | None = None,
        sleep: float = 0.1,
    ) -> None:
        start = start or CONFIG["data"]["start_date"]
        end = end or CONFIG["data"]["end_date"]
        codes = list(codes)
        logger.info(f"开始拉取日K线: {len(codes)}只 [{start} -> {end}]")
        success = fail = 0
        for code in tqdm(codes, desc="daily bars"):
            df = self._fetch_daily_one(code, start, end)
            if df.empty:
                fail += 1
                continue
            with self._conn() as c:
                df.to_sql("daily_bars", c, if_exists="append", index=False)
            success += 1
            time.sleep(sleep)
        logger.info(f"完成: 成功{success} 失败{fail}")

    def get_daily_bars(
        self,
        codes: Iterable[str] | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        sql = "SELECT * FROM daily_bars WHERE 1=1"
        params = []
        if codes is not None:
            placeholders = ",".join(["?"] * len(list(codes)))
            codes = list(codes)
            sql += f" AND code IN ({placeholders})"
            params.extend(codes)
        if start:
            sql += " AND date >= ?"
            params.append(start)
        if end:
            sql += " AND date <= ?"
            params.append(end)
        sql += " ORDER BY code, date"
        with self._conn() as c:
            df = pd.read_sql(sql, c, params=params)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
        return df

    # -------- 指数 --------
    def fetch_index(self, code: str = "sh000300", start: str | None = None,
                    end: str | None = None) -> pd.DataFrame:
        start = start or CONFIG["data"]["start_date"]
        end = end or CONFIG["data"]["end_date"]
        logger.info(f"拉取指数 {code}")
        try:
            df = ak.stock_zh_index_daily(symbol=code)
        except Exception as e:
            logger.error(f"指数拉取失败: {e}")
            return pd.DataFrame()
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df["code"] = code
        df = df[(df["date"] >= start) & (df["date"] <= end)]
        cols = ["code", "date", "open", "high", "low", "close", "volume"]
        for c in ["amount"]:
            if c not in df.columns:
                df[c] = 0.0
        df = df[cols + ["amount"]]
        with self._conn() as c:
            df.to_sql("index_bars", c, if_exists="append", index=False)
        return df

    def get_index(self, code: str = "sh000300") -> pd.DataFrame:
        with self._conn() as c:
            df = pd.read_sql("SELECT * FROM index_bars WHERE code = ? ORDER BY date",
                             c, params=[code])
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
        return df


if __name__ == "__main__":
    f = DataFetcher()
    f.fetch_stock_list()
