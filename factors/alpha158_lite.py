"""Alpha158-lite - qlib Alpha158 的纯 pandas 实现 (不依赖 pyqlib).

## 为什么要 "lite"

qlib 官方 Alpha158 有 158 个因子 + 依赖 qlib 的 Expression Engine. 我们的
回测链路直接消费东财 DataFrame, 为了:
    1. 不强绑 qlib 数据生态 (装包要 > 500 MB)
    2. 在小股票池 (10-100 只) 上快速验证
    3. 方便学习 Alpha158 的构造思路

实现了其中 ~30 个核心因子, 足以验证 "换玩具因子 → IC 显著" 的飞跃.

## 因子清单

| 类别 | 因子 |
|---|---|
| K 线形态 | KMID, KLEN, KUP, KLOW, KSFT |
| 归一化价 | OPEN, HIGH, LOW |
| 动量 / 反转 | ROC{5,10,20,60}, MA{5,10,20,60}, BETA{5,20} |
| 波动率 | STD{5,20}, MAX{20}, MIN{20} |
| 技术指标 | RSV{5,20}, QTLU{20}, QTLD{20}, RSQR{20} |
| 成交量 | VMA{5,20}, VSTD{20}, VSUMP{20}, WVMA{5}|
| 量价关系 | CORR{10,20}, CORD{10} |

## 产出

MultiIndex 列 (factor, code), index = date, 与 ResearchPipeline 的
_stage_features 格式兼容.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.where(b.abs() > 1e-12, np.nan)


def _rolling_beta(y: pd.Series, window: int) -> pd.Series:
    """y vs 时间 t 的滚动回归斜率 (类似 linear regression slope)."""
    def _beta(arr):
        if np.isnan(arr).any():
            return np.nan
        x = np.arange(len(arr))
        sx = x.mean()
        sy = arr.mean()
        cov = ((x - sx) * (arr - sy)).sum()
        var = ((x - sx) ** 2).sum()
        return cov / var if var > 1e-12 else np.nan
    return y.rolling(window, min_periods=max(3, window // 2)).apply(_beta, raw=True)


def _rolling_rsqr(y: pd.Series, window: int) -> pd.Series:
    """线性回归 R²."""
    def _r2(arr):
        if np.isnan(arr).any() or len(arr) < 3:
            return np.nan
        x = np.arange(len(arr))
        sx = x.mean()
        sy = arr.mean()
        cov = ((x - sx) * (arr - sy)).sum()
        var_x = ((x - sx) ** 2).sum()
        var_y = ((arr - sy) ** 2).sum()
        if var_x < 1e-12 or var_y < 1e-12:
            return np.nan
        r = cov / np.sqrt(var_x * var_y)
        return r * r
    return y.rolling(window, min_periods=max(3, window // 2)).apply(_r2, raw=True)


def compute_alpha158_lite(
    df: pd.DataFrame,
    windows: tuple = (5, 10, 20, 60),
) -> pd.DataFrame:
    """为单只股票算一组 Alpha158-lite 因子.

    Args:
        df: 单票 DataFrame, index = date, 含 [open, high, low, close, volume]
        windows: 滚动窗口

    Returns:
        DataFrame (index=date, columns=factor_name). 所有因子已做 close/volume
        归一化, 横截面再标准化交给上游.
    """
    c = df["close"].astype(float)
    o = df["open"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    v = df["volume"].astype(float).clip(lower=1e-6)

    out = pd.DataFrame(index=df.index)

    # ----- K 线形态 (5) -----
    out["KMID"] = _safe_div(c - o, o)
    out["KLEN"] = _safe_div(h - l, o)
    out["KUP"] = _safe_div(h - np.maximum(o, c), o)
    out["KLOW"] = _safe_div(np.minimum(o, c) - l, o)
    # K 线重心相对: mid vs range
    out["KSFT"] = _safe_div(2 * c - h - l, o)

    # ----- 归一化价 (3) -----
    out["NORM_OPEN"] = _safe_div(o, c)
    out["NORM_HIGH"] = _safe_div(h, c)
    out["NORM_LOW"] = _safe_div(l, c)

    for w in windows:
        # ----- 动量 / 反转 -----
        out[f"ROC{w}"] = c.pct_change(w)
        out[f"MA{w}"] = _safe_div(c.rolling(w).mean(), c)
        out[f"STD{w}"] = _safe_div(c.rolling(w).std(), c)

    # ----- 极值 & 分位 -----
    for w in (10, 20, 60):
        out[f"MAX{w}"] = _safe_div(h.rolling(w).max(), c)
        out[f"MIN{w}"] = _safe_div(l.rolling(w).min(), c)
        out[f"QTLU{w}"] = _safe_div(c.rolling(w).quantile(0.8), c)
        out[f"QTLD{w}"] = _safe_div(c.rolling(w).quantile(0.2), c)
        # RSV Stochastic: (close - min) / (max - min)
        hh = h.rolling(w).max()
        ll = l.rolling(w).min()
        out[f"RSV{w}"] = _safe_div(c - ll, hh - ll)

    # ----- 线性回归 -----
    out["BETA20"] = _rolling_beta(c, 20) / c
    out["RSQR20"] = _rolling_rsqr(c, 20)

    # ----- 成交量 -----
    for w in (5, 20):
        out[f"VMA{w}"] = _safe_div(v.rolling(w).mean(), v)
    out["VSTD20"] = _safe_div(v.rolling(20).std(), v.rolling(20).mean())

    # 涨日成交量占比 (positive volume sum ratio)
    pct = c.pct_change()
    pos_v = v.where(pct > 0, 0.0)
    neg_v = v.where(pct < 0, 0.0)
    out["VSUMP20"] = _safe_div(
        pos_v.rolling(20).sum(), v.rolling(20).sum(),
    )
    out["VSUMN20"] = _safe_div(
        neg_v.rolling(20).sum(), v.rolling(20).sum(),
    )

    # 量价加权均线
    wv = (c * v).rolling(5).sum() / v.rolling(5).sum()
    out["WVMA5"] = _safe_div(wv, c)

    # ----- 量价相关性 -----
    log_v = np.log(v + 1)
    for w in (10, 20):
        out[f"CORR{w}"] = c.rolling(w).corr(log_v)
    # 变化率相关
    dc = c.pct_change()
    dv = log_v.diff()
    out["CORD10"] = dc.rolling(10).corr(dv)

    # 清理 inf
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def compute_alpha158_panel(
    daily_df: pd.DataFrame,
    windows: tuple = (5, 10, 20, 60),
) -> pd.DataFrame:
    """对全部股票计算 Alpha158-lite, 产出 MultiIndex 面板.

    Args:
        daily_df: [code, date, open, high, low, close, volume] 的长表

    Returns:
        DataFrame, index=date, columns=MultiIndex([factor, code]).
        与 ResearchPipeline._stage_features 的输出结构一致.
    """
    per_stock = {}
    for code, g in daily_df.groupby("code"):
        g = g.sort_values("date").set_index("date")
        per_stock[code] = compute_alpha158_lite(g, windows=windows)

    # 合并: {code: DataFrame[date × factor]} → MultiIndex columns (factor, code)
    combined = pd.concat(per_stock, axis=1, names=["code", "factor"])
    # 交换层级使列为 (factor, code), 方便 feature_df[factor_name] 取截面
    combined.columns = combined.columns.swaplevel(0, 1)
    combined = combined.sort_index(axis=1)
    return combined


def combine_factors_equal_weight(
    feature_df: pd.DataFrame,
) -> pd.DataFrame:
    """等权合成所有因子得到一个 composite signal.

    ⚠️ 因子方向不一致时等权会互相抵消, 建议用 combine_factors_ic_weighted.

    步骤: 每因子横截面 z-score → 等权平均.
    """
    if not isinstance(feature_df.columns, pd.MultiIndex):
        return feature_df

    factor_names = feature_df.columns.get_level_values(0).unique()
    z_list = []
    for f in factor_names:
        mat = feature_df[f]
        mu = mat.mean(axis=1)
        sd = mat.std(axis=1)
        z = mat.sub(mu, axis=0).div(sd.replace(0, np.nan), axis=0)
        z = z.clip(-3, 3)
        z_list.append(z)
    stacked = pd.concat(z_list).groupby(level=0).mean()
    return stacked


def combine_factors_rolling_ic(
    feature_df: pd.DataFrame,
    label_df: pd.DataFrame,
    window: int = 60,
    top_k: int = 8,
    min_abs_ic: float = 0.02,
    label_horizon: int = 10,
) -> pd.DataFrame:
    """Rolling window IC 加权合成 (严格无前视偏差).

    ⚠️ Leak 修复 (2026-04): label_horizon 参数强制把训练窗口后推 H 天.
    原先 win = [i-window, i-1] 的最后 H 天, label.loc[i-1] 是 [i-1, i-1+H]
    的 forward return, 要求知道今天及之后 H-1 天的价格 → **未来函数**.
    修后: win = [i-window-H, i-1-H], 保证窗口内每个标签的 forward-window
    结束 <= i-1, 即"昨天及之前". 无任何未来信息.

    每日 t 用 shift 后的 IC 决定因子权重, 应用到 t 日. 这样因子方向会随
    市场 regime 切换而自动调整.

    Args:
        feature_df: MultiIndex (factor, code)
        label_df: DataFrame (date × code) forward return
        window: IC 回溯天数 (太短噪声大, 太长跟不上 regime 切换)
        top_k: 每日保留 |IC| 最大的 K 个因子
        min_abs_ic: |IC| 阈值
        label_horizon: label 的 forward horizon (天), 用于防泄露 shift.
                       默认 10 = 与 multi_horizon_label 最大 horizon 匹配.

    Returns:
        composite signal DataFrame (date × code).
    """
    if not isinstance(feature_df.columns, pd.MultiIndex):
        return feature_df

    common_idx = feature_df.index.intersection(label_df.index)
    H = max(1, int(label_horizon))
    # 至少要 window + H + 30 个样本, 否则回退等权
    if len(common_idx) < window + H + 30:
        return combine_factors_equal_weight(feature_df)

    factor_names = list(feature_df.columns.get_level_values(0).unique())
    # 预先算每个因子 × 每日的横截面 z-score
    factor_z = {}
    for f in factor_names:
        m = feature_df[f].loc[common_idx]
        mu = m.mean(axis=1)
        sd = m.std(axis=1).replace(0, np.nan)
        factor_z[f] = m.sub(mu, axis=0).div(sd, axis=0).clip(-3, 3)

    lbl = label_df.loc[common_idx]
    composite = pd.DataFrame(index=common_idx, columns=lbl.columns, dtype=float)

    factor_selection_log = []
    for i, dt in enumerate(common_idx):
        # 需要 i >= window + H 才有合法的无泄露窗口
        if i < window + H:
            composite.loc[dt] = np.nan
            continue
        # ⚠️ 关键修复: 窗口后推 H 天, 保证窗口内最新 label 也落在 <= dt-1
        win_dates = common_idx[i - window - H : i - H]
        # 各因子在窗口内的 IC
        ic_map = {}
        for f in factor_names:
            fz = factor_z[f].loc[win_dates].stack()
            lb = lbl.loc[win_dates].stack().reindex(fz.index)
            merged = pd.concat([fz, lb], axis=1).dropna()
            if len(merged) < 20 or merged.iloc[:, 0].std() < 1e-9:
                continue
            ic = merged.iloc[:, 0].corr(merged.iloc[:, 1], method="spearman")
            if pd.notna(ic) and abs(ic) >= min_abs_ic:
                ic_map[f] = float(ic)
        if not ic_map:
            composite.loc[dt] = 0.0
            continue
        sorted_f = sorted(ic_map.items(), key=lambda x: -abs(x[1]))[:top_k]
        total_abs = sum(abs(v) for _, v in sorted_f)
        if total_abs < 1e-9:
            composite.loc[dt] = 0.0
            continue
        # 当日合成
        row = pd.Series(0.0, index=lbl.columns)
        for f, ic in sorted_f:
            row = row.add(factor_z[f].loc[dt] * (ic / total_abs),
                           fill_value=0.0)
        composite.loc[dt] = row
        if i % 50 == 0:
            factor_selection_log.append({
                "date": dt, "top_5": [(f, round(ic, 3))
                                         for f, ic in sorted_f[:5]],
            })

    composite.attrs["window"] = window
    composite.attrs["selection_log"] = factor_selection_log
    return composite


def combine_factors_ic_weighted(
    feature_df: pd.DataFrame,
    label_df: pd.DataFrame,
    train_ratio: float = 0.5,
    min_abs_ic: float = 0.02,
    top_k: int = 8,
) -> pd.DataFrame:
    """IC 加权合成: 用样本内数据估每个因子的 IC 作权重与符号.

    流程:
        1. 按 train_ratio 切分训练 / 全期
        2. 对训练期每因子算全期 Spearman IC (flatten)
        3. 丢弃 |IC| < min_abs_ic 的因子
        4. 用 sign(IC) × |IC| / sum(|IC|) 作为权重
        5. 对整个时间段应用这个固定权重

    这避免了等权合成的方向抵消问题. 有轻微 look-ahead: 样本内 IC 用的是
    整个训练段的 label, 训练段内部存在泄露但样本外 (train_ratio~1.0) 不泄露.

    Args:
        feature_df: MultiIndex (factor, code)
        label_df: DataFrame (date × code), forward return
        train_ratio: 用前多大比例估 IC 权重
        min_abs_ic: 过滤弱因子

    Returns:
        composite signal DataFrame (date × code).
    """
    if not isinstance(feature_df.columns, pd.MultiIndex):
        return feature_df

    common_idx = feature_df.index.intersection(label_df.index)
    if len(common_idx) < 30:
        return combine_factors_equal_weight(feature_df)

    cutoff = int(len(common_idx) * train_ratio)
    train_idx = common_idx[:cutoff]

    factor_names = feature_df.columns.get_level_values(0).unique()
    ic_map = {}
    for f in factor_names:
        mat = feature_df[f]
        # flatten 成单 Series 对齐 label
        feat_flat = mat.loc[train_idx].stack()
        lbl_flat = label_df.loc[train_idx].stack().reindex(feat_flat.index)
        merged = pd.concat([feat_flat, lbl_flat], axis=1).dropna()
        if len(merged) < 30 or merged.iloc[:, 0].std() < 1e-9:
            continue
        ic = merged.iloc[:, 0].corr(merged.iloc[:, 1], method="spearman")
        if pd.notna(ic) and abs(ic) >= min_abs_ic:
            ic_map[f] = float(ic)

    if not ic_map:
        return combine_factors_equal_weight(feature_df)

    # 只保留 |IC| top-K 因子, 避免噪声稀释
    sorted_fact = sorted(ic_map.items(), key=lambda x: -abs(x[1]))[:top_k]
    ic_map = dict(sorted_fact)
    total_abs = sum(abs(v) for v in ic_map.values())
    weights = {f: ic / total_abs for f, ic in ic_map.items()}

    # 合成 (用 sign × |weight|, 即 weights 自带方向)
    z_list = []
    for f, w in weights.items():
        mat = feature_df[f]
        mu = mat.mean(axis=1)
        sd = mat.std(axis=1)
        z = mat.sub(mu, axis=0).div(sd.replace(0, np.nan), axis=0).clip(-3, 3)
        z_list.append(z * w)
    stacked = pd.concat(z_list).groupby(level=0).sum(min_count=1)
    # 附加诊断 (调用方可访问)
    stacked.attrs["n_factors_used"] = len(weights)
    stacked.attrs["top_factors"] = sorted(
        ic_map.items(), key=lambda x: -abs(x[1])
    )[:10]
    return stacked
