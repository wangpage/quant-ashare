"""板块动量因子 — 基于相关性的"伪板块"版本.

动机:
    真·行业板块接口不稳 (东财 push2 CDN 被屏蔽), 而热点轮动每天都在发生.
    我们不需要精确的 SW/申万行业映射, 只需要捕捉"和它一起走的那批股最近强不强".

方法:
    对每只目标 code, 在大 universe 里找相关性 top-K 的邻居, 邻居最近 lookback 日
    平均收益 = SECTOR_MOM_{lookback}. 邻居收益方差 = SECTOR_DISP (越大越不像板块).

因子:
    SECTOR_MOM_5   +  邻居 5 日累计涨跌均值
    SECTOR_MOM_1   +  邻居昨日涨跌均值 (隔夜延续性)
    SECTOR_DISP_5  -  邻居 5 日累计涨跌标准差 (越大越不是真板块)

Universe 选择:
    - 默认用 cache/kline_*_n500.parquet (500 只大盘股)
    - 也允许传入自定义 universe DataFrame (同字段 code/date/close)

边界:
    - 若一只 code 在 universe 中不存在, 算出的是"它和 universe 里最相似者"的动量
    - 相关窗口 corr_window 默认 60 日, 至少要求 40 日有效重叠
    - 邻居数 n_neighbors=15; 邻居与自己相关系数 <0.2 的直接剔除 (避免噪声邻居)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _align_returns(kline: pd.DataFrame,
                    as_of: pd.Timestamp | None = None,
                    window: int = 60) -> pd.DataFrame:
    """把长表 [code,date,close] pivot 成日收益宽表 [date × code].

    只保留 as_of 前 window 日的数据, 用于算相关和最近收益.
    """
    df = kline[["code", "date", "close"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    if as_of is None:
        as_of = df["date"].max()
    else:
        as_of = pd.Timestamp(as_of)
    df = df[df["date"] <= as_of]

    wide = df.pivot_table(index="date", columns="code", values="close",
                           aggfunc="last").sort_index()
    # 只保留最近 window+10 天 (算收益要多 1 天)
    wide = wide.tail(window + 10)
    ret = wide.pct_change().dropna(how="all")
    # 丢掉缺失太多的 code (>=50% 日期缺失)
    ret = ret.loc[:, ret.notna().mean() >= 0.5]
    return ret.tail(window)


def _corr_matrix(ret_wide: pd.DataFrame,
                  min_overlap: int = 40) -> pd.DataFrame:
    """pairwise Pearson corr. 要求 overlap >= min_overlap 日."""
    n = ret_wide.count()  # 每只股有多少个非 NaN 日
    corr = ret_wide.corr(min_periods=min_overlap)
    return corr


def compute_sector_momentum(
    target_kline: pd.DataFrame,
    universe_kline: pd.DataFrame | None = None,
    as_of: pd.Timestamp | None = None,
    lookback: int = 5,
    n_neighbors: int = 15,
    corr_window: int = 60,
    min_corr: float = 0.2,
    min_overlap: int = 40,
) -> pd.DataFrame:
    """对每只 target code 计算 SECTOR_MOM_{1,5} + SECTOR_DISP_5.

    Args:
        target_kline: watchlist 的日 K, 长表 code/date/close
        universe_kline: 大盘 universe 日 K, 同字段. None 则用 target 自身
        as_of:        算到哪天为止 (含). 默认 target_kline 最大日
        lookback:     邻居近期涨跌的计算窗口 (日)
        n_neighbors:  取相关 top-k 的 k
        corr_window:  算相关用多少日收益
        min_corr:     邻居必须 |corr| >= 此值, 否则不算"同群"
        min_overlap:  最少多少日重叠才算相关可信

    Returns:
        DataFrame indexed by code, columns: SECTOR_MOM_1, SECTOR_MOM_5,
        SECTOR_DISP_5, n_peers, mean_peer_corr
    """
    if universe_kline is None:
        universe_kline = target_kline
    else:
        # 合并: universe 不含 target 的代码也要加进来供相关计算
        extra_codes = set(target_kline["code"]) - set(universe_kline["code"])
        if extra_codes:
            add = target_kline[target_kline["code"].isin(extra_codes)]
            universe_kline = pd.concat([universe_kline, add], ignore_index=True)

    # 收益宽表
    ret = _align_returns(universe_kline, as_of=as_of, window=corr_window)
    if ret.shape[0] < min_overlap:
        raise ValueError(f"ret 有效日仅 {ret.shape[0]} < min_overlap {min_overlap}")

    target_codes = sorted(set(target_kline["code"]))
    # 对 target 子集算相关 (N×M)
    target_codes_in_wide = [c for c in target_codes if c in ret.columns]
    if not target_codes_in_wide:
        raise ValueError("target 代码都不在 universe 中")

    # 全量 corr 矩阵, 然后切 target 列 (pandas corrwith 旧版不支持 min_periods)
    full_corr = ret.corr(min_periods=min_overlap)
    corr = full_corr[target_codes_in_wide]  # rows=universe, cols=target

    # 对每个 target code, 近期涨跌宽表 (lookback 日累计收益)
    lb_ret = ret.tail(lookback).sum()  # Series: code -> lookback 日累计
    last_day_ret = ret.tail(1).iloc[0]  # 最近一日收益

    out = []
    for code in target_codes:
        if code not in corr.columns:
            out.append({
                "code": code, "SECTOR_MOM_1": np.nan, "SECTOR_MOM_5": np.nan,
                "SECTOR_DISP_5": np.nan, "n_peers": 0, "mean_peer_corr": np.nan,
            })
            continue

        col = corr[code].drop(code, errors="ignore")
        col = col.dropna()
        col = col[col.abs() >= min_corr]
        top = col.abs().sort_values(ascending=False).head(n_neighbors)
        peers = top.index.tolist()
        if not peers:
            out.append({
                "code": code, "SECTOR_MOM_1": np.nan, "SECTOR_MOM_5": np.nan,
                "SECTOR_DISP_5": np.nan, "n_peers": 0, "mean_peer_corr": np.nan,
            })
            continue

        # 加权: 正相关邻居 +, 负相关邻居 - (反向同步也是信号)
        signed_corr = col.loc[peers]
        peer_lb = lb_ret.reindex(peers)
        peer_1 = last_day_ret.reindex(peers)

        # 加权均值: 相关越强权重越大
        w = signed_corr.abs()
        w = w / w.sum()
        # 正相关邻居贡献原 sign, 负相关邻居反 sign 贡献
        mom_5 = float((peer_lb * np.sign(signed_corr) * w).sum())
        mom_1 = float((peer_1 * np.sign(signed_corr) * w).sum())
        disp_5 = float(peer_lb.std())

        out.append({
            "code": code,
            "SECTOR_MOM_1": mom_1 * 100,    # 百分比
            "SECTOR_MOM_5": mom_5 * 100,
            "SECTOR_DISP_5": disp_5 * 100,
            "n_peers": len(peers),
            "mean_peer_corr": float(signed_corr.mean()),
        })

    return pd.DataFrame(out).set_index("code")


def load_universe_kline(cache_dir: Path,
                         prefer_n: int = 500) -> pd.DataFrame | None:
    """加载 cache 里最新的大盘 universe K 线."""
    files = sorted(cache_dir.glob(f"kline_*_n{prefer_n}.parquet"))
    if not files:
        return None
    df = pd.read_parquet(files[-1])
    # 保留必要字段
    df = df[["code", "date", "close"]].copy()
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ---------- CLI 自测 ----------
if __name__ == "__main__":
    import sys
    ROOT = Path(__file__).resolve().parent.parent
    CACHE = ROOT / "cache"

    print("加载 watchlist kline...")
    wl = sorted(CACHE.glob("watchlist_kline_*.parquet"))
    if not wl:
        print("❌ 无 watchlist kline 缓存"); sys.exit(1)
    target = pd.read_parquet(wl[-1])
    target["code"] = target["code"].astype(str).str.zfill(6)
    target["date"] = pd.to_datetime(target["date"])
    print(f"  target: {len(target)} 行, {target['code'].nunique()} 只")

    print("加载 universe kline...")
    universe = load_universe_kline(CACHE)
    if universe is not None:
        print(f"  universe: {len(universe)} 行, {universe['code'].nunique()} 只")

    print("\n计算 sector momentum (as_of = 2026-04-21)...")
    sm = compute_sector_momentum(
        target, universe,
        as_of=pd.Timestamp("2026-04-21"),
        lookback=5, n_neighbors=15, corr_window=60,
    )
    print(sm.head(10))
    print("\n统计:")
    print(sm.describe())
