"""结构/指数/沉寂类因子 - 纯离线, 基于 daily OHLCV + 可选指数序列.

覆盖非共识 alpha 源:
    #5 跌停结构: 一字/开板/封单节奏 (比涨停更有信息密度)
    #7 指数掩护: 逆指数行为 = 真实强弱
    #8 沉寂/压缩: 无交易数据, 压缩弹簧效应

设计:
    - 输入: daily_df (必需) + index_df (可选; 不传则跳过 #7 家族)
    - 输出: MultiIndex (date, code), columns = factor names
    - 签名与 alpha_pandas.compute_pandas_alpha 对齐, 直接 join 到 feat_combo
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12
LIMIT_PCT = 0.095          # 跌停/涨停阈值, 留一点缓冲
DOWN_PCT = -LIMIT_PCT


# ------------------------------------------------------------------
#  #5  跌停结构
# ------------------------------------------------------------------
def _limit_down_features(g: pd.DataFrame) -> pd.DataFrame:
    """单股时序: 跌停结构系列."""
    close = g["close"]
    open_, high, low = g["open"], g["high"], g["low"]
    vol = g["volume"].astype(float)
    pre_close = close.shift(1)

    ret1 = close.pct_change()
    is_ld = ret1 < DOWN_PCT
    # 一字跌停: open==low==close 且收跌停
    is_one_word = (open_ == low) & (open_ == close) & is_ld
    # 开板跌停: 当日曾触及跌停线 (low <= pre*0.905) 但收盘未跌停
    down_line = pre_close * (1 + DOWN_PCT)
    is_opened_ld = (low <= down_line * 1.001) & (ret1 > DOWN_PCT)

    # 封单/放量程度: 跌停日成交量 vs 20 日均量
    vol_ma20 = vol.rolling(20).mean()
    vol_ratio_ld = pd.Series(np.where(is_ld, vol / (vol_ma20 + EPS), np.nan),
                             index=g.index)

    # 跌停日振幅 (开板程度代理): (high-low)/pre_close, 一字=0, 反复开板=大
    amp_ld = pd.Series(np.where(is_ld, (high - low) / (pre_close + EPS), np.nan),
                       index=g.index)

    out = pd.DataFrame(index=g.index)
    # 20 日一字跌停次数 (越多 → 越弱)
    out["LD_ONE_WORD_20"] = -is_one_word.rolling(20).sum()     # 负号: 高=强势
    # 20 日开板跌停次数 (日内抄底, 中性偏多)
    out["LD_OPENED_20"] = is_opened_ld.rolling(20).sum()
    # 20 日跌停日平均振幅 (高 → 开板多 → 有承接 → 偏多)
    out["LD_AMP_MEAN_20"] = amp_ld.rolling(20).mean()
    # 20 日跌停日成交量倍数均值 (高 → 放量 → 抛压真实; 低 → 无量跌停, 反弹概率高)
    out["LD_VOL_RATIO_20"] = -vol_ratio_ld.rolling(20).mean()  # 负号: 无量跌停 → 反弹
    # 当前连续跌停天数 (值越大, 越可能继续杀)
    streak = is_ld.astype(int)
    out["LD_STREAK"] = -(streak * (streak.groupby((streak != streak.shift()).cumsum())
                                    .cumcount() + 1))
    # 60 日最大连跌停天数 (历史暴跌伤疤, 长期利空)
    # 先算每段连跌长度, 再 rolling max
    grp = (is_ld != is_ld.shift()).cumsum()
    seg_len = is_ld.astype(int) * is_ld.groupby(grp).cumcount().add(1)
    out["LD_MAX_STREAK_60"] = -seg_len.rolling(60).max()
    # 跌停后 5 日累积反弹 (埋伏抄底代理): 仅在 T-1 刚跌停的样本上算 T..T+4 收益
    ld_shift = is_ld.astype(bool).shift(fill_value=False)
    post_ld_ret = ret1.where(ld_shift, np.nan).rolling(5).sum()
    out["LD_POST_REBOUND_5"] = post_ld_ret

    return out


# ------------------------------------------------------------------
#  #8  沉寂 / 压缩
# ------------------------------------------------------------------
def _compression_features(g: pd.DataFrame) -> pd.DataFrame:
    close = g["close"]
    high, low = g["high"], g["low"]
    vol = g["volume"].astype(float)
    ret1 = close.pct_change()

    out = pd.DataFrame(index=g.index)
    # 成交量 60 日分位 (低 → 沉寂)
    out["VOL_PCTILE_60"] = -vol.rolling(60).rank(pct=True)     # 负号: 低分位 → 正信号
    # 连续沉寂天数 (vol < 60 日 30 分位)
    pct30 = vol.rolling(60).quantile(0.3)
    is_silent = (vol < pct30).astype(int)
    grp = (is_silent != is_silent.shift()).cumsum()
    silent_streak = is_silent * is_silent.groupby(grp).cumcount().add(1)
    out["SILENT_STREAK"] = silent_streak
    # 振幅压缩: 20 日 (max-min) / 60 日 (max-min) 的比
    range20 = high.rolling(20).max() - low.rolling(20).min()
    range60 = high.rolling(60).max() - low.rolling(60).min()
    out["RANGE_COMPRESSION"] = -(range20 / (range60 + EPS))    # 低比值 → 压缩 → 正
    # Bollinger 宽度压缩 (收盘 20 日 std / 60 日均 std)
    std20 = ret1.rolling(20).std()
    std60 = ret1.rolling(60).std()
    out["BOLL_SQUEEZE"] = -(std20 / (std60 + EPS))
    # 复合沉寂打分: 低成交 × 低波动 × 低换手, 值越高越"压缩"
    turn_pct = vol.rolling(60).rank(pct=True)
    vol_pct = ret1.abs().rolling(60).rank(pct=True)
    amp_pct = ((high - low) / (close + EPS)).rolling(60).rank(pct=True)
    out["SILENCE_SCORE"] = (1 - turn_pct) * (1 - vol_pct) * (1 - amp_pct)
    # 成交量趋势斜率 (最近 10 日线性回归斜率, 负 → 递减 → 接近爆发)
    x = np.arange(10)
    def _slope(y):
        if np.isnan(y).any():
            return np.nan
        return np.polyfit(x, y, 1)[0]
    out["VOL_SLOPE_10"] = -vol.rolling(10).apply(_slope, raw=True)

    return out


# ------------------------------------------------------------------
#  #7  指数掩护 (需要 index_df)
# ------------------------------------------------------------------
def _index_cover_features(g: pd.DataFrame, idx_ret: pd.Series) -> pd.DataFrame:
    """idx_ret: index pct_change, index 与 g 的 date 对齐."""
    close = g["close"]
    ret1 = close.pct_change()
    # 对齐
    ir = idx_ret.reindex(g.index)

    out = pd.DataFrame(index=g.index)
    # 20 日 beta (协方差 / 方差)
    cov20 = ret1.rolling(60).cov(ir)
    var60 = ir.rolling(60).var()
    beta = cov20 / (var60 + EPS)
    out["BETA_INDEX_60"] = beta

    # 逆指数: 指数涨但个股跌 (20 日比例)
    contra = ((ir > 0) & (ret1 < 0)).astype(int).rolling(20).mean()
    cover  = ((ir < 0) & (ret1 > 0)).astype(int).rolling(20).mean()
    # 逆指数强势 = cover - contra (指数跌我涨的天数多, 指数涨我跌的天数少)
    out["INDEX_COVER_20"] = cover - contra
    # 相对强度 (20 日超额)
    out["REL_RET_20"] = close.pct_change(20) - ir.rolling(20).sum()
    out["REL_RET_5"]  = close.pct_change(5)  - ir.rolling(5).sum()
    # Jensen alpha: 20 日累积 (ret - beta*idx_ret)
    resid = ret1 - beta * ir
    out["ALPHA_RESID_20"] = resid.rolling(20).sum()
    # 抗跌指标: 指数跌时, 个股跌幅均值 / 指数跌幅均值 (值 < 1 抗跌, >1 更脆)
    idx_down = ir < 0
    def _down_ratio(mask_window):
        return None  # 用 apply 实现更清晰, 此处省
    # 简化版: 过去 20 日, 指数负收益日个股收益和 / 指数收益和
    down_stock = ret1.where(idx_down).rolling(20).sum()
    down_idx   = ir.where(idx_down).rolling(20).sum()
    out["DOWN_PROTECT_20"] = -(down_stock / (down_idx.abs() + EPS))  # 值高 → 抗跌

    return out


# ------------------------------------------------------------------
#  主入口
# ------------------------------------------------------------------
def compute_microstructure_alpha(
    daily_df: pd.DataFrame,
    index_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """离线结构因子总入口.

    Args:
        daily_df: 列 date/code/open/high/low/close/volume (必需)
        index_df: 可选, 列 date/close (指数收盘价, 用于计算 #7)

    Returns:
        MultiIndex (date, code) 的因子面板
    """
    df = daily_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["code", "date"])

    # 指数 return 序列
    idx_ret = None
    if index_df is not None and not index_df.empty:
        idx = index_df.copy()
        idx["date"] = pd.to_datetime(idx["date"])
        idx = idx.sort_values("date").set_index("date")
        idx_ret = idx["close"].pct_change()

    pieces = []
    for code, g in df.groupby("code"):
        g = g.set_index("date")
        # 三族因子
        ld   = _limit_down_features(g)
        comp = _compression_features(g)
        cols = [ld, comp]
        if idx_ret is not None:
            cov = _index_cover_features(g, idx_ret)
            cols.append(cov)
        feat = pd.concat(cols, axis=1)
        feat["code"] = code
        pieces.append(feat.reset_index())

    if not pieces:
        return pd.DataFrame()
    full = (pd.concat(pieces, ignore_index=True)
              .set_index(["date", "code"]).sort_index())
    return full


LIMIT_DOWN_FACTOR_NAMES = [
    "LD_ONE_WORD_20", "LD_OPENED_20", "LD_AMP_MEAN_20",
    "LD_VOL_RATIO_20", "LD_STREAK", "LD_MAX_STREAK_60",
    "LD_POST_REBOUND_5",
]
COMPRESSION_FACTOR_NAMES = [
    "VOL_PCTILE_60", "SILENT_STREAK", "RANGE_COMPRESSION",
    "BOLL_SQUEEZE", "SILENCE_SCORE", "VOL_SLOPE_10",
]
INDEX_COVER_FACTOR_NAMES = [
    "BETA_INDEX_60", "INDEX_COVER_20",
    "REL_RET_20", "REL_RET_5", "ALPHA_RESID_20",
    "DOWN_PROTECT_20",
]
MICROSTRUCTURE_FACTOR_NAMES = (LIMIT_DOWN_FACTOR_NAMES +
                               COMPRESSION_FACTOR_NAMES +
                               INDEX_COVER_FACTOR_NAMES)


if __name__ == "__main__":
    # 自检
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    cache = Path(__file__).resolve().parent.parent / "cache"
    kl_path = next(iter(cache.glob("kline_*_n*.parquet")), None)
    assert kl_path, "无 kline 缓存, 先跑 v3"
    daily = pd.read_parquet(kl_path)
    # 取小样本验证
    codes = daily["code"].unique()[:10]
    sub = daily[daily["code"].isin(codes)]
    print(f"输入: {sub.shape} 行, {len(codes)} 只")
    feat = compute_microstructure_alpha(sub)
    print(f"输出无指数: shape={feat.shape}")
    print(f"  cols={list(feat.columns)[:8]}...")
    # 带指数
    from data_adapter.em_direct import fetch_index_daily
    idx = fetch_index_daily("000905", "20230101", "20260420")   # 中证500 中小盘
    print(f"指数 {idx.shape}")
    feat2 = compute_microstructure_alpha(sub, index_df=idx)
    print(f"输出带指数: shape={feat2.shape}, cols 数={len(feat2.columns)}")
    # nan 统计
    print(f"  nan ratio={feat2.isna().mean().mean():.2%}")
    print(f"  sample:\n{feat2.head(3)}")
