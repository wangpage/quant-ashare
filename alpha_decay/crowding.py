"""因子拥挤度 (Crowding) - 为什么你的 alpha 突然不赚钱了?

行业真相: A股量化规模 2019 年 ~2000 亿, 2024 年 ~2 万亿. 同样的 alpha
被 10 倍资金追逐, 收益被拥挤压缩. 头部私募用这些指标监控并主动"下线".

拥挤度维度:
    1) Turnover 上升 + IC 下降 → 内部信号被淹没
    2) 多空组合的换手率突然飙升 → 外部资金涌入同一方向
    3) 空头融券余额 → 同业在反向
    4) Wind/聚源 "公开策略复现": 论文/研报公开后 6 个月内因子作废

Kakushadze (2019): "When a trading signal is public, its half-life is 18 months."
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def factor_crowding_index(
    factor_returns: pd.Series,
    turnover: pd.Series,
    short_window: int = 20, long_window: int = 120,
) -> pd.Series:
    """综合拥挤度: 短期/长期 对比.

    公式:
        short_turnover = rolling 20d turnover mean
        long_turnover  = rolling 120d turnover mean
        short_return   = rolling 20d factor return
        long_return    = rolling 120d factor return

        crowding = (short_turnover / long_turnover) /
                   (short_return / long_return)

        > 1.5: 资金涌入但收益未同步 → 拥挤
        > 2.0: 危险, 建议降仓或下线
    """
    t_ratio = turnover.rolling(short_window).mean() / \
              turnover.rolling(long_window).mean()
    r_ratio = (factor_returns.rolling(short_window).mean() /
               factor_returns.rolling(long_window).mean()).abs()

    # 避免除 0
    idx = t_ratio / (r_ratio + 1e-6)
    return idx


def turnover_signal(
    positions: pd.DataFrame,
    baseline_turnover: float = 0.3,
) -> dict[str, float]:
    """组合换手率监控.

    Args:
        positions: 每日持仓权重矩阵 [dates, stocks]
        baseline_turnover: 基准换手率 (年化 3-6x 即 0.03-0.06 日)

    Returns:
        实际 turnover vs 基准, 以及异常天数
    """
    turnover = (positions - positions.shift(1)).abs().sum(axis=1) / 2
    mean_t = turnover.mean()
    anomaly_days = (turnover > baseline_turnover * 1.5).sum()
    return {
        "mean_turnover": float(mean_t),
        "baseline": baseline_turnover,
        "ratio": float(mean_t / baseline_turnover),
        "anomaly_days": int(anomaly_days),
    }


def public_strategy_overlap(
    my_factor_values: pd.Series,
    public_factor_values: pd.Series,
) -> float:
    """检查我方因子与公开因子 (如 Fama-French, Alpha158) 的相关性.

    如果 > 0.7, 说明我的 alpha 其实是被行业广泛知晓的, 生命周期风险大.

    建议:
        - 相关性 < 0.3: 独立 alpha, 安全
        - 0.3-0.6: 部分暴露, 监控
        - > 0.6: 大概率被压缩, 寻找正交成分
    """
    mine = my_factor_values.dropna()
    public = public_factor_values.dropna()
    common = mine.index.intersection(public.index)
    if len(common) < 20:
        return 0.0
    return float(mine.loc[common].corr(public.loc[common], method="spearman"))


def alpha_portfolio_correlation(factors: pd.DataFrame) -> pd.DataFrame:
    """多 alpha 组合的相关矩阵. 用于因子池构建.

    目标: 找出低相关 (< 0.3) 因子组合, 分散化.
    内部规则:
        - 一个 alpha pod 里 >= 5 个正交因子
        - 因子两两相关 < 0.3
        - 最后用 Mean-Variance 分配权重
    """
    return factors.corr(method="spearman")


def factor_capacity_estimate(
    factor_ic: float, factor_volatility: float,
    universe_daily_volume: float,
    participation_rate: float = 0.05,
) -> float:
    """容量估算: 因子能管多少钱而不被自己吃掉 alpha.

    简化公式 (头部私募版):
        capacity = universe_volume × participation_rate × (IC / vol) × 252

    作用: 超过 capacity 的资金会把因子打死.
    """
    if factor_volatility <= 0:
        return 0.0
    sharpe = factor_ic / factor_volatility * np.sqrt(252)
    capacity = universe_daily_volume * participation_rate * sharpe / 3.0
    return float(capacity)
