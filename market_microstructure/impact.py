"""Almgren-Chriss 冲击成本模型 + Square-root law.

为什么大众回测会骗自己:
    大众: slippage = 2bps (固定)
    实际:
        - 买入 1 亿元 茅台 (日成交 50 亿): 冲击 ~5 bps
        - 买入 1 亿元 某小盘 (日成交 3 亿): 冲击 ~50 bps
    固定滑点低估小盘 10 倍以上, 策略在实盘暴雷.

Almgren-Chriss (2000) 的经典结论:
    permanent_impact = γ × σ × Q / V          (线性)
    temporary_impact = η × σ × sqrt(Q / V)    (sqrt 律)
    total_cost = (permanent + temporary) × trade_size

其中:
    σ = 股票日波动率
    Q = 目标成交金额
    V = 股票日成交额
    γ, η = 市场冲击系数, A股实证: γ ≈ 0.1, η ≈ 0.3

Square-root impact (Torre 2018, BARRA):
    更简洁的经验公式, 单独用 sqrt 项足以解释 80% 的冲击
"""
from __future__ import annotations

import numpy as np


def almgren_chriss_impact(
    trade_amount: float,
    daily_volume: float,
    volatility: float,
    gamma: float = 0.10,
    eta: float = 0.30,
    is_buy: bool = True,
) -> dict[str, float]:
    """完整 AC 冲击模型.

    Args:
        trade_amount: 本次拟成交金额 (元)
        daily_volume: 该股票日成交额 (元)
        volatility: 日波动率 (如 0.02 表示 2%)
        gamma, eta: 市场冲击系数 (A股实证参考)
        is_buy: 方向, 买入冲击会推高成本, 卖出压低

    Returns:
        dict 含 'permanent_bps', 'temporary_bps', 'total_bps'
    """
    participation = trade_amount / max(daily_volume, 1.0)

    # 永久冲击 (影响后续所有成交价)
    permanent = gamma * volatility * participation
    # 临时冲击 (只影响本次, sqrt 律)
    temporary = eta * volatility * np.sqrt(participation)

    total = permanent + temporary
    direction = 1 if is_buy else -1
    return {
        "permanent_bps": permanent * 1e4 * direction,
        "temporary_bps": temporary * 1e4 * direction,
        "total_bps":     total * 1e4 * direction,
        "participation_rate": participation,
    }


def square_root_impact(
    trade_amount: float,
    daily_volume: float,
    volatility: float,
    k: float = 0.40,
) -> float:
    """简化版: 实战中用这个就够.

    cost_bps = k × σ × sqrt(Q/V) × 10000

    A股经验系数 k ≈ 0.4 (大盘) ~ 0.8 (小盘)

    Returns:
        单边冲击 bps (基点)
    """
    participation = trade_amount / max(daily_volume, 1.0)
    cost = k * volatility * np.sqrt(participation)
    return cost * 1e4


def estimate_participation_rate(
    trade_amount: float, daily_volume: float,
) -> dict[str, any]:
    """估算当前订单的"参与率". >10% 算危险, >30% 极不建议.

    头部私募内部规则:
        - 单笔订单参与率上限 20%, 分单执行
        - 日内累计某只票 ≤ 30% 日成交额
        - 小盘 (<50 亿市值) 再 halve
    """
    rate = trade_amount / max(daily_volume, 1.0)
    if rate > 0.30:
        risk = "EXTREME"
    elif rate > 0.15:
        risk = "HIGH"
    elif rate > 0.05:
        risk = "MEDIUM"
    else:
        risk = "LOW"
    return {
        "participation_rate": rate,
        "risk_level": risk,
        "recommended_slicing": max(1, int(rate * 20)),  # 推荐切分次数
    }


def kyle_lambda(
    price_changes: "np.ndarray",
    signed_volume: "np.ndarray",
) -> float:
    """Kyle's Lambda: 单位订单流的价格冲击.

    计算:
        ΔP = λ × signed_volume + ε
        λ 越大, 市场越浅, 冲击越大.

    用途:
        - 按股票估算自己的 λ, 作为流动性代理
        - 盘前 λ 预测, 动态调整参与率

    Args:
        price_changes: 价格变动序列 (bps)
        signed_volume: 有向成交量 (+买 -卖), 股数
    """
    from numpy.linalg import lstsq
    x = np.asarray(signed_volume).reshape(-1, 1)
    y = np.asarray(price_changes).reshape(-1, 1)
    coef, _, _, _ = lstsq(x, y, rcond=None)
    return float(coef[0, 0])
