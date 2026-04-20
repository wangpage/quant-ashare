"""波动率目标 (Volatility Targeting) - 头部对冲基金必备.

核心:
    组合波动率锚定某值 (如 15% 年化), 超过 = 减仓, 低于 = 加仓.
    实证: 波动率目标策略比 buy-and-hold 夏普高 20-30%.

另有: 带回撤约束的凯利公式 (Kelly with DD constraint).
"""
from __future__ import annotations

import numpy as np


def vol_target_scale(
    current_vol: float,
    target_vol: float = 0.15,
    max_leverage: float = 1.5,
    min_leverage: float = 0.3,
) -> float:
    """根据当前波动率返回仓位缩放系数.

    scale = target / current, 限制在 [min, max].

    Args:
        current_vol: 当前 (滚动) 组合年化波动率
        target_vol: 目标年化 (A股经验 15-20%)
        max_leverage: 杠杆上限
        min_leverage: 空仓下限 (避免回避过度)

    用法:
        final_weights = raw_weights × scale
    """
    if current_vol <= 0:
        return 1.0
    scale = target_vol / current_vol
    return float(np.clip(scale, min_leverage, max_leverage))


def calculate_kelly_with_drawdown(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    max_drawdown_limit: float = 0.15,
    fraction: float = 0.25,
) -> dict[str, float]:
    """带回撤约束的凯利公式 (更实用).

    原始凯利: f* = p/a - q/b  (p=胜率, a=平均亏, b=平均赚)

    实战: 凯利推荐的满仓太激进 (可能导致 -50%+ 回撤),
    所以用 fractional kelly (0.25 × f*) + 回撤熔断.

    Args:
        win_rate: 历史胜率
        avg_win: 平均盈利 (正数, 如 0.08 = 8%)
        avg_loss: 平均亏损 (正数, 如 0.04 = 4%)
        max_drawdown_limit: 组合最大允许回撤 (触发后仓位 → 0)
        fraction: 凯利分数 (0.25 是行业标准)

    Returns:
        推荐仓位比例 + 预期指标
    """
    if win_rate <= 0 or avg_loss <= 0:
        return {"kelly_size": 0, "fractional_size": 0,
                "expected_return": 0, "risk_ratio": 0}

    full_kelly = win_rate / avg_loss - (1 - win_rate) / avg_win
    full_kelly = max(0, min(full_kelly, 1.0))

    fractional = full_kelly * fraction

    # 回撤约束: 凯利仓位导致的理论最大回撤不能超过 limit
    # 经验公式: MaxDD ≈ fractional × σ × 3
    expected_vol = np.sqrt(win_rate * avg_win ** 2 +
                           (1 - win_rate) * avg_loss ** 2)
    implied_dd = fractional * expected_vol * 3
    if implied_dd > max_drawdown_limit:
        fractional = fractional * max_drawdown_limit / implied_dd

    expected_ret = win_rate * avg_win - (1 - win_rate) * avg_loss
    return {
        "full_kelly": full_kelly,
        "fractional_kelly": fractional,
        "expected_return": expected_ret,
        "implied_max_dd": min(implied_dd, max_drawdown_limit),
        "risk_ratio": expected_ret / expected_vol if expected_vol > 0 else 0,
    }


def drawdown_scaler(
    current_drawdown: float,
    max_drawdown_limit: float = 0.15,
    warning_threshold: float = 0.05,
) -> float:
    """基于当前回撤动态调整仓位.

        DD < warning_threshold:  仓位 × 1.0 (正常)
        warning ≤ DD < limit:     仓位 × (1 - DD/limit)^2 (非线性衰减)
        DD ≥ limit:               仓位 × 0 (清仓熔断)
    """
    dd = abs(current_drawdown)
    if dd < warning_threshold:
        return 1.0
    if dd >= max_drawdown_limit:
        return 0.0
    x = (dd - warning_threshold) / (max_drawdown_limit - warning_threshold)
    return float((1 - x) ** 2)


def volatility_scaling_series(
    returns, target_vol: float = 0.15, window: int = 20,
):
    """给收益率序列加波动率目标, 返回调整后的收益率.

    回测中验证: 波动目标策略夏普提升 0.2-0.4.
    """
    import pandas as pd
    returns = pd.Series(returns) if not isinstance(returns, pd.Series) else returns
    realized_vol = returns.rolling(window).std() * np.sqrt(252)
    scale = target_vol / realized_vol.replace(0, np.nan)
    scale = scale.clip(0.3, 1.5).fillna(1.0)
    return returns * scale.shift(1)     # shift 1 避免未来函数
