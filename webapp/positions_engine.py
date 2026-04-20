"""持仓诊断引擎 - 规则化的"何时卖出", 不是预测概率.

核心原则:
    1. 绝不告诉你"涨 73%" 这种骗人数字
    2. 给你 **可操作** 的: 买入价 / 止损位 / 止盈位 / 持仓动作
    3. 给你 **可验证** 的: 信号触发规则 + 历史胜率
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from webapp.realtime_quote import (
    TechnicalSummary, analyze_technicals, realtime_quote,
)


ActionType = Literal["strong_hold", "hold", "reduce", "sell", "stop_loss", "add"]


@dataclass
class PositionDiagnosis:
    code: str
    name: str
    cost_price: float
    shares: int
    current_price: float
    pnl_abs: float
    pnl_pct: float
    # 风险点
    stop_loss_price: float
    stop_profit_price: float
    max_hold_days: int
    # 系统化动作
    action: ActionType
    action_reason: list[str]
    # 参考指标 (不是预测!)
    tech: TechnicalSummary | None = None
    directional_bias: str = "neutral"
    bias_confidence: float = 0.0
    # 规则触发
    triggered_rules: list[str] = field(default_factory=list)


@dataclass
class Recommendation:
    """Top 10 推荐股票."""
    rank: int
    code: str
    name: str
    current_price: float
    model_score: float
    theme: str
    directional_bias: str
    bias_confidence: float
    suggested_entry: float
    stop_loss: float
    take_profit: float
    max_hold_days: int
    reasoning: list[str]


# ==================== 止盈止损规则 ====================
def calculate_stops(
    cost_price: float,
    atr: float,
    current_price: float | None = None,
    atr_multiplier_sl: float = 2.0,
    atr_multiplier_tp: float = 4.0,
    min_stop_loss_pct: float = 0.05,
    min_take_profit_pct: float = 0.08,
) -> tuple[float, float]:
    """基于 ATR 的动态止损止盈.

    规则:
        止损 = cost - ATR × 2  (或成本 -5%, 取更严的)
        止盈 = cost + ATR × 4  (或成本 +8%, 取更宽的)

    为什么不用固定 %:
        高波动股用 2% 止损会被随机波动打掉
        低波动股用 5% 止损又太宽松
        ATR 动态适应每只股的脾气
    """
    ref = cost_price  # 以成本价锚定, 不追高
    sl_atr = ref - atr * atr_multiplier_sl
    sl_pct = ref * (1 - min_stop_loss_pct)
    stop_loss = max(sl_atr, sl_pct)   # 取更近的 (更严)

    tp_atr = ref + atr * atr_multiplier_tp
    tp_pct = ref * (1 + min_take_profit_pct)
    take_profit = max(tp_atr, tp_pct)  # 取更远的 (赚得多)

    return round(stop_loss, 2), round(take_profit, 2)


# ==================== 卖出信号引擎 ====================
def evaluate_position(
    code: str, name: str,
    cost_price: float, shares: int,
    kline_df: pd.DataFrame,
    theme: str = "",
    crowding_score: float = 0.0,   # 0-10, 主题拥挤度
) -> PositionDiagnosis:
    """单一持仓诊断. 返回规则化的动作."""
    if kline_df.empty:
        return PositionDiagnosis(
            code=code, name=name, cost_price=cost_price, shares=shares,
            current_price=cost_price, pnl_abs=0, pnl_pct=0,
            stop_loss_price=cost_price * 0.95,
            stop_profit_price=cost_price * 1.10,
            max_hold_days=10,
            action="hold", action_reason=["无K线数据"],
        )

    tech = analyze_technicals(kline_df)
    current = tech.current_price
    pnl_pct = (current / cost_price - 1) if cost_price else 0
    pnl_abs = (current - cost_price) * shares

    sl, tp = calculate_stops(cost_price, tech.atr14, current)

    # 规则引擎
    rules: list[str] = []
    action: ActionType = "hold"
    reasons: list[str] = []

    # R1: 触发止损 (最高优先级)
    if current <= sl:
        rules.append(f"跌破止损位 {sl:.2f}")
        action = "stop_loss"
        reasons.append(f"强制止损 (触及 {sl:.2f})")
    # R2: 触发止盈
    elif current >= tp:
        rules.append(f"达到止盈位 {tp:.2f}")
        action = "sell"
        reasons.append(f"止盈了结 (触及 {tp:.2f})")
    else:
        # R3: 超买风险
        if tech.rsi14 >= 82:
            rules.append(f"RSI 超买 ({tech.rsi14:.0f})")
            action = "reduce"
            reasons.append("RSI > 80, 短期过热, 减仓 30-50%")
        # R4: MACD 死叉 + 跌破 MA20
        elif tech.macd_signal == "bear" and not tech.ma_info.get("above_ma20"):
            rules.append("MACD 死叉 + 跌破 MA20")
            action = "reduce"
            reasons.append("技术面转弱, 减仓一半")
        # R5: 主题极度拥挤 + 已有浮盈
        elif crowding_score > 8 and pnl_pct > 0.15:
            rules.append(f"主题拥挤 {crowding_score}/10 + 浮盈 {pnl_pct:.1%}")
            action = "reduce"
            reasons.append("主题拥挤度过高, 获利了结部分")
        # R6: 量价背离
        elif tech.bearish_signals >= 3:
            rules.append(f"{tech.bearish_signals} 个空头信号共振")
            action = "reduce"
            reasons.append("多指标转空, 谨慎减仓")
        # R7: 强势持有
        elif tech.bullish_signals >= 4 and pnl_pct > 0:
            action = "strong_hold"
            reasons.append(f"{tech.bullish_signals} 个多头信号, 持仓跟随")
        # R8: 回调加仓机会
        elif (tech.bullish_signals >= 3 and -0.05 < pnl_pct < 0
              and tech.boll_pos < -0.5):
            action = "add"
            reasons.append("技术面强势但短期回调至布林下轨, 可补仓")
        # 默认持有
        else:
            action = "hold"
            reasons.append(f"技术面 {tech.bullish_signals}多 {tech.bearish_signals}空, 继续持有")

    # 持仓时间限制
    holding_limit = 20  # 默认 20 天
    if action == "strong_hold":
        holding_limit = 45
    elif action == "reduce":
        holding_limit = 10

    return PositionDiagnosis(
        code=code, name=name,
        cost_price=cost_price, shares=shares,
        current_price=current,
        pnl_abs=round(pnl_abs, 2),
        pnl_pct=round(pnl_pct, 4),
        stop_loss_price=sl,
        stop_profit_price=tp,
        max_hold_days=holding_limit,
        action=action,
        action_reason=reasons,
        tech=tech,
        directional_bias=tech.directional_bias,
        bias_confidence=tech.confidence_pct,
        triggered_rules=rules,
    )


# ==================== 推荐引擎 ====================
def build_recommendations(
    candidates: list[dict],           # 从 signal 池来
    kline_data: dict[str, pd.DataFrame],
    realtime_quotes: dict[str, dict],
    top_n: int = 10,
) -> list[Recommendation]:
    """从候选池中生成 Top N 推荐, 含实时价/止损/止盈."""
    results = []
    for c in candidates:
        code = c["code"]
        df = kline_data.get(code)
        quote = realtime_quotes.get(code)
        if df is None or df.empty:
            continue

        tech = analyze_technicals(df)
        current_price = (quote.get("price") if quote else None) or tech.current_price

        # 用当前价做锚算止损/止盈 (假设 "此时此刻买入")
        sl, tp = calculate_stops(current_price, tech.atr14, current_price)

        reasoning = [
            f"模型分 {c.get('model_score', 0):.2f}",
            f"主题: {c.get('theme', '-')}",
        ] + tech.signal_detail[:3]

        results.append(Recommendation(
            rank=len(results) + 1,
            code=code,
            name=c.get("name", code),
            current_price=round(current_price, 2),
            model_score=c.get("model_score", 0.0),
            theme=c.get("theme", "-"),
            directional_bias=tech.directional_bias,
            bias_confidence=tech.confidence_pct,
            suggested_entry=round(current_price * 0.995, 2),  # 低于现价 0.5% 限价
            stop_loss=sl,
            take_profit=tp,
            max_hold_days=20,
            reasoning=reasoning,
        ))
        if len(results) >= top_n:
            break
    return results
