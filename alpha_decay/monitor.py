"""因子衰减监控: rolling IC, 半衰期, 健康度."""
from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_ic_decay(
    factor: pd.DataFrame, forward_returns: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """多窗口 IC, 看不同时间尺度的 alpha 衰减.

    Args:
        factor, forward_returns: 形状 [dates, stocks]
        windows: 默认 [20, 60, 120, 252] (月/季/半年/年)

    Returns:
        DataFrame 列: 不同窗口的 rolling Spearman IC
    """
    windows = windows or [20, 60, 120, 252]
    ic_daily = factor.corrwith(forward_returns, axis=1, method="spearman")
    out = {}
    for w in windows:
        out[f"ic_{w}d"] = ic_daily.rolling(w).mean()
    out["ic_daily"] = ic_daily
    return pd.DataFrame(out)


def half_life_estimate(ic_daily: pd.Series) -> float:
    """信号半衰期估计 (指数衰减拟合).

    公式: IC_t = IC_0 × exp(-λ × t)
         半衰期 T_half = ln(2) / λ

    返回 NaN 说明 IC 不是单调衰减 (好事, 说明还没衰).
    """
    series = ic_daily.dropna().abs()
    if len(series) < 30:
        return float("nan")
    x = np.arange(len(series))
    y = np.log(series + 1e-8)
    # 线性拟合 log(IC) ~ -λ × t
    slope, _ = np.polyfit(x, y, 1)
    if slope >= 0:
        return float("inf")    # 没有衰减
    return float(np.log(2) / (-slope))


def alpha_health_score(
    recent_ic: float, longterm_ic: float,
    recent_turnover: float, longterm_turnover: float,
    recent_sharpe: float, longterm_sharpe: float,
) -> dict[str, float]:
    """综合健康度 0-100.

    维度:
        - IC 保持率 40%
        - Turnover 稳定性 30%
        - 夏普保持率 30%

    下线阈值:
        < 30: 立刻下线
        30-60: 降权 50%
        > 60: 正常使用
    """
    ic_ratio = (recent_ic + 1e-8) / (longterm_ic + 1e-8)
    turnover_ratio = (longterm_turnover + 1e-8) / (recent_turnover + 1e-8)
    sharpe_ratio = (recent_sharpe + 1e-8) / (longterm_sharpe + 1e-8)

    # clip 到 [0, 1.2], 超过 1.2 按 1.2 算
    ic_score = np.clip(ic_ratio, 0, 1.2) / 1.2 * 40
    turnover_score = np.clip(turnover_ratio, 0, 1.2) / 1.2 * 30
    sharpe_score = np.clip(sharpe_ratio, 0, 1.2) / 1.2 * 30

    total = float(ic_score + turnover_score + sharpe_score)
    return {
        "total_score": total,
        "ic_score": float(ic_score),
        "turnover_score": float(turnover_score),
        "sharpe_score": float(sharpe_score),
        "action": ("KEEP" if total > 60 else
                   "DOWNWEIGHT" if total > 30 else "RETIRE"),
    }


def ic_ir(ic_series: pd.Series) -> dict[str, float]:
    """IC 信息比率 + p-value.

    IR = mean(IC) / std(IC) × sqrt(252)
    """
    s = ic_series.dropna()
    if len(s) < 20:
        return {"ic_mean": 0, "ic_std": 0, "icir": 0, "ic_t_stat": 0}
    mean, std = s.mean(), s.std()
    if std == 0:
        return {"ic_mean": mean, "ic_std": 0, "icir": 0, "ic_t_stat": 0}
    icir = mean / std * np.sqrt(252)
    t_stat = mean / (std / np.sqrt(len(s)))
    return {"ic_mean": float(mean), "ic_std": float(std),
            "icir": float(icir), "ic_t_stat": float(t_stat)}
