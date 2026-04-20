"""akshare SQLite -> qlib bin 格式转换器.

qlib 需要的目录结构:
  qlib_data/
    ├── calendars/day.txt            # 交易日历
    ├── instruments/all.txt          # 全市场股票代码
    │                 csi300.txt     # 沪深300
    └── features/
         sh600000/
             ├── open.day.bin
             ├── close.day.bin
             └── ...

qlib 的 .bin 格式:
  首 4 字节: float32 起始索引 (日历中的位置)
  之后: float32 数组 (对齐日历)
"""
from __future__ import annotations

import struct
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.config import CONFIG, PROJECT_ROOT
from utils.logger import logger

from .fetcher import DataFetcher


class AkshareToQlibConverter:
    FIELDS = ["open", "close", "high", "low", "volume", "amount", "factor"]

    def __init__(self, qlib_dir: str | Path | None = None):
        self.qlib_dir = Path(qlib_dir) if qlib_dir else PROJECT_ROOT / "qlib_data"
        self.qlib_dir.mkdir(parents=True, exist_ok=True)
        (self.qlib_dir / "calendars").mkdir(exist_ok=True)
        (self.qlib_dir / "instruments").mkdir(exist_ok=True)
        (self.qlib_dir / "features").mkdir(exist_ok=True)
        self.fetcher = DataFetcher()

    @staticmethod
    def _qlib_code(code: str) -> str:
        """'600000' -> 'sh600000', '000001' -> 'sz000001', '300750' -> 'sz300750'."""
        if code.startswith("6"):
            return f"sh{code}"
        return f"sz{code}"

    def build_calendar(self, df: pd.DataFrame) -> pd.DatetimeIndex:
        """全市场交易日历 (取所有股票的日期并集)."""
        cal = pd.to_datetime(df["date"].unique())
        cal = cal.sort_values()
        cal_str = cal.strftime("%Y-%m-%d")
        (self.qlib_dir / "calendars" / "day.txt").write_text(
            "\n".join(cal_str), encoding="utf-8"
        )
        logger.info(f"交易日历: {len(cal)}个交易日 [{cal_str[0]} -> {cal_str[-1]}]")
        return cal

    def build_instruments(self, codes: list[str], name: str = "all"):
        """写 instruments/{name}.txt, 格式: code\tstart\tend"""
        start = CONFIG["data"]["start_date"]
        end = CONFIG["data"]["end_date"]
        lines = [f"{self._qlib_code(c)}\t{start}\t{end}" for c in codes]
        (self.qlib_dir / "instruments" / f"{name}.txt").write_text(
            "\n".join(lines), encoding="utf-8"
        )
        logger.info(f"instruments/{name}.txt: {len(codes)}只")

    def _write_bin(self, series: pd.Series, out_path: Path,
                   calendar: pd.DatetimeIndex):
        """写 qlib .bin 文件 (float32)."""
        series = series.copy()
        series.index = pd.to_datetime(series.index)
        aligned = series.reindex(calendar)
        start_idx = aligned.first_valid_index()
        if start_idx is None:
            return
        start_pos = calendar.get_loc(start_idx)
        data = aligned.loc[start_idx:].values.astype(np.float32)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(struct.pack("<f", float(start_pos)))
            f.write(data.tobytes())

    def convert_all(self, codes: list[str] | None = None):
        logger.info("开始转换 akshare -> qlib bin")
        df = self.fetcher.get_daily_bars(codes=codes)
        if df.empty:
            logger.error("数据库为空，先运行 scripts/01_download_data.py")
            return

        df = df.dropna(subset=["close"])
        df["factor"] = 1.0

        calendar = self.build_calendar(df)
        all_codes = df["code"].unique().tolist()
        self.build_instruments(all_codes, "all")

        for code, g in tqdm(df.groupby("code"), desc="writing bins"):
            qlib_code = self._qlib_code(code)
            g = g.sort_values("date").set_index("date")
            for field in self.FIELDS:
                if field not in g.columns:
                    continue
                out = self.qlib_dir / "features" / qlib_code / f"{field}.day.bin"
                self._write_bin(g[field], out, calendar)

        logger.info(f"转换完成: {self.qlib_dir}")

    def build_csi300_instruments(self):
        """从 akshare 拉取沪深300成分股."""
        import akshare as ak
        try:
            df = ak.index_stock_cons_csindex(symbol="000300")
            codes = df["成分券代码"].tolist() if "成分券代码" in df.columns \
                else df.iloc[:, 0].astype(str).tolist()
        except Exception as e:
            logger.warning(f"拉取沪深300失败: {e}")
            return
        self.build_instruments(codes, "csi300")


if __name__ == "__main__":
    AkshareToQlibConverter().convert_all()
