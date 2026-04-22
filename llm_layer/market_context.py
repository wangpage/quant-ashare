"""为 radar_analyst.deep_analyze 生成市场上下文文本块.

目的: 用纯文本传达"图"能传达的信息, 让 Opus 推理"是否已 price-in".
输入 code, 从 cache/ 目录下的 parquet 文件读 kline + 指数数据,
输出 ~200 字的单行上下文 blob.

数据源优先级 (后覆盖前):
  1. cache/kline_*_n500.parquet          — 500 股票 universe, 最近 3 年
  2. cache/watchlist_kline_YYYYMMDD.parquet — watchlist 最新日频 (含今日前)
指数: cache/v4_idx_*.parquet (HS300 code=000300).

找不到数据时返回 "无历史数据缓存" 降级文本, 不触发 live akshare 调用.
模块级懒加载 + mtime 感知重载, 首次加载后常驻内存.
"""
from __future__ import annotations

import glob
import threading
from pathlib import Path
from typing import Any

import pandas as pd

from utils.config import PROJECT_ROOT
from utils.logger import logger


_CACHE_DIR = PROJECT_ROOT / "cache"

_lock = threading.Lock()
_stock_df: pd.DataFrame | None = None
_index_df: pd.DataFrame | None = None
_loaded_signature: tuple | None = None  # (stock_mtimes, idx_mtimes)


def _latest(glob_pat: str) -> Path | None:
    matches = sorted(_CACHE_DIR.glob(glob_pat), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def _gather_signature() -> tuple:
    universe = _latest("kline_*_n500.parquet")
    watchlist = _latest("watchlist_kline_*.parquet")
    idx = _latest("v4_idx_*.parquet")
    return (
        (universe, universe.stat().st_mtime) if universe else None,
        (watchlist, watchlist.stat().st_mtime) if watchlist else None,
        (idx, idx.stat().st_mtime) if idx else None,
    )


def _ensure_loaded() -> None:
    global _stock_df, _index_df, _loaded_signature
    sig = _gather_signature()
    with _lock:
        if sig == _loaded_signature and _stock_df is not None:
            return

        frames = []
        for entry in (sig[0], sig[1]):  # universe 先, watchlist 覆盖
            if entry is None:
                continue
            path, _ = entry
            try:
                df = pd.read_parquet(path)
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                frames.append(df)
                logger.info(f"market_context 加载 {path.name} rows={len(df)}")
            except Exception as e:
                logger.warning(f"读 {path} 失败: {e}")

        if not frames:
            _stock_df = pd.DataFrame()
        else:
            merged = pd.concat(frames, ignore_index=True, sort=False)
            # 同 (code, date) 取最新文件(watchlist)那条: 后到的 drop 前面
            merged = merged.sort_values(["code", "date"]).drop_duplicates(
                subset=["code", "date"], keep="last"
            ).reset_index(drop=True)
            _stock_df = merged

        if sig[2] is not None:
            try:
                idx = pd.read_parquet(sig[2][0])
                if "date" in idx.columns:
                    idx["date"] = pd.to_datetime(idx["date"])
                _index_df = idx
                logger.info(f"market_context 加载指数 {sig[2][0].name} rows={len(idx)}")
            except Exception as e:
                logger.warning(f"读指数失败: {e}")
                _index_df = pd.DataFrame()
        else:
            _index_df = pd.DataFrame()

        _loaded_signature = sig


def _stock_bars(code: str, tail: int = 260) -> pd.DataFrame:
    _ensure_loaded()
    if _stock_df is None or _stock_df.empty:
        return pd.DataFrame()
    df = _stock_df[_stock_df["code"] == code].sort_values("date")
    if df.empty:
        return df
    return df.tail(tail).reset_index(drop=True)


def _index_return_20d() -> float | None:
    _ensure_loaded()
    if _index_df is None or _index_df.empty:
        return None
    df = _index_df[_index_df["code"].isin(["000300", "sh000300"])].sort_values("date")
    if len(df) < 21:
        return None
    tail = df.tail(21)
    return round(float((tail["close"].iloc[-1] / tail["close"].iloc[0] - 1) * 100), 2)


def compute_indicators(code: str) -> dict:
    """计算个股技术面指标. 返回字典, 缺失字段为 None."""
    out: dict[str, Any] = {
        "code": code,
        "bars_in_cache": 0,
        "close": None,
        "ret_5d_pct": None,
        "ret_20d_pct": None,
        "ret_60d_pct": None,
        "pos_in_250d_range": None,
        "pos_in_60d_range": None,
        "vol_ratio_5d": None,
        "amount_yi_latest": None,
        "amount_rank_in_20d": None,
        "excess_vs_hs300_20d_pct": None,
        "consec_up_days": 0,
        "at_near_250d_high": False,
    }
    df = _stock_bars(code)
    if df.empty:
        return out

    out["bars_in_cache"] = len(df)
    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    volume = df["volume"].to_numpy()
    amount = df["amount"].to_numpy() if "amount" in df.columns else None
    pct = df["pct_chg"].to_numpy() if "pct_chg" in df.columns else None

    out["close"] = round(float(close[-1]), 3)

    def _ret(n: int) -> float | None:
        if len(close) <= n or close[-1 - n] == 0:
            return None
        return round((close[-1] / close[-1 - n] - 1) * 100, 2)

    out["ret_5d_pct"] = _ret(5)
    out["ret_20d_pct"] = _ret(20)
    out["ret_60d_pct"] = _ret(60)

    if len(df) >= 60:
        lo60, hi60 = float(low[-60:].min()), float(high[-60:].max())
        if hi60 > lo60:
            out["pos_in_60d_range"] = round((close[-1] - lo60) / (hi60 - lo60), 2)

    if len(df) >= 120:
        window = min(250, len(df))
        lo, hi = float(low[-window:].min()), float(high[-window:].max())
        if hi > lo:
            out["pos_in_250d_range"] = round((close[-1] - lo) / (hi - lo), 2)
        out["at_near_250d_high"] = bool(close[-1] >= hi * 0.97)

    if len(volume) >= 6:
        avg5 = float(volume[-6:-1].mean())
        if avg5 > 0:
            out["vol_ratio_5d"] = round(float(volume[-1]) / avg5, 2)

    if amount is not None and len(amount) >= 1:
        out["amount_yi_latest"] = round(float(amount[-1]) / 1e8, 2)
        if len(amount) >= 20:
            recent = amount[-20:]
            rank = int((recent >= amount[-1]).sum())
            out["amount_rank_in_20d"] = rank

    idx_ret = _index_return_20d()
    if out["ret_20d_pct"] is not None and idx_ret is not None:
        out["excess_vs_hs300_20d_pct"] = round(out["ret_20d_pct"] - idx_ret, 2)

    if pct is not None:
        cnt = 0
        for v in pct[::-1]:
            if v is not None and v > 0:
                cnt += 1
            else:
                break
        out["consec_up_days"] = cnt

    return out


def build_context_text(code: str) -> str:
    """把指标拼成 Opus 友好的短文本 blob."""
    if not code:
        return "无代码, 无市场上下文"
    ind = compute_indicators(code)
    if ind["bars_in_cache"] == 0:
        return f"{code} 无本地历史数据缓存 (非 universe / watchlist, 跳过技术面)."

    parts = [f"{code}"]
    if ind["close"] is not None:
        parts.append(f"收盘{ind['close']}")

    ret_parts = []
    for label, key in [("5日", "ret_5d_pct"), ("20日", "ret_20d_pct"),
                        ("60日", "ret_60d_pct")]:
        if ind[key] is not None:
            ret_parts.append(f"{label}{ind[key]:+.1f}%")
    if ret_parts:
        parts.append("收益:" + "/".join(ret_parts))

    if ind["excess_vs_hs300_20d_pct"] is not None:
        parts.append(f"20日超额HS300 {ind['excess_vs_hs300_20d_pct']:+.1f}%")

    pos_parts = []
    if ind["pos_in_250d_range"] is not None:
        pos_parts.append(f"250日区间{int(ind['pos_in_250d_range']*100)}%")
    if ind["pos_in_60d_range"] is not None:
        pos_parts.append(f"60日区间{int(ind['pos_in_60d_range']*100)}%")
    if pos_parts:
        parts.append("位置:" + "/".join(pos_parts))

    if ind["at_near_250d_high"]:
        parts.append("接近250日高点")

    vol_parts = []
    if ind["vol_ratio_5d"]:
        vol_parts.append(f"量比{ind['vol_ratio_5d']:.2f}x")
    if ind["amount_yi_latest"]:
        tail = ""
        if ind["amount_rank_in_20d"] and ind["amount_rank_in_20d"] <= 3:
            tail = f"(近20日TOP{ind['amount_rank_in_20d']})"
        vol_parts.append(f"成交{ind['amount_yi_latest']:.1f}亿{tail}")
    if vol_parts:
        parts.append("量能:" + "/".join(vol_parts))

    if ind["consec_up_days"] >= 3:
        parts.append(f"连涨{ind['consec_up_days']}日")

    return " | ".join(parts)


def refresh() -> None:
    """强制丢缓存, 下次调用触发重新加载. daemon 可在每日早盘前调一次."""
    global _stock_df, _index_df, _loaded_signature
    with _lock:
        _stock_df = None
        _index_df = None
        _loaded_signature = None
