"""市场状态检测器 - 输出 regime 标签 + 仓位乘数.

基于 AI-Trader 论文: "Risk control capability determines cross-market robustness"
                    "AI trading strategies achieve excess returns more readily in highly liquid markets"

regime 标签决定:
  - 仓位乘数 (position multiplier): 在 bear/crash 阶段压降仓位
  - 策略偏好: bull_trending 可以追动量, choppy 切反转
  - 是否允许新开仓: crash/euphoria 禁止
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any

import pandas as pd

from utils.logger import logger

from .indicators import (
    compute_trend, compute_volatility, compute_breadth,
    detect_crash, detect_euphoria,
    TrendResult, VolResult, BreadthResult,
)


class MarketRegime(str, Enum):
    BULL_TRENDING = "bull_trending"      # 上升趋势, 适合动量
    BULL_QUIET = "bull_quiet"            # 上升但低波动
    BEAR_TRENDING = "bear_trending"      # 下降趋势, 避免做多
    BEAR_QUIET = "bear_quiet"            # 下降但低波动
    CHOPPY = "choppy"                    # 震荡, 适合反转
    CRASH = "crash"                      # 崩盘, 禁止开仓
    EUPHORIA = "euphoria"                # 狂热, 警惕顶部
    UNKNOWN = "unknown"


# 每种 regime 推荐的仓位乘数 [0, 1] 和允许动作
REGIME_RULES = {
    MarketRegime.BULL_TRENDING: {
        "position_mult": 1.0,
        "allow_new_long": True, "allow_new_short": False,
        "preferred_strategy": "momentum",
        "stop_loss_mult": 1.0,
    },
    MarketRegime.BULL_QUIET: {
        "position_mult": 0.8,
        "allow_new_long": True, "allow_new_short": False,
        "preferred_strategy": "momentum",
        "stop_loss_mult": 1.0,
    },
    MarketRegime.BEAR_TRENDING: {
        "position_mult": 0.3,
        "allow_new_long": False, "allow_new_short": False,
        "preferred_strategy": "defensive",
        "stop_loss_mult": 0.7,
    },
    MarketRegime.BEAR_QUIET: {
        "position_mult": 0.5,
        "allow_new_long": True, "allow_new_short": False,
        "preferred_strategy": "mean_reversion",
        "stop_loss_mult": 0.8,
    },
    MarketRegime.CHOPPY: {
        "position_mult": 0.6,
        "allow_new_long": True, "allow_new_short": False,
        "preferred_strategy": "mean_reversion",
        "stop_loss_mult": 0.7,
    },
    MarketRegime.CRASH: {
        "position_mult": 0.0,
        "allow_new_long": False, "allow_new_short": False,
        "preferred_strategy": "cash",
        "stop_loss_mult": 0.5,
    },
    MarketRegime.EUPHORIA: {
        "position_mult": 0.4,             # 压降, 警惕顶部
        "allow_new_long": True, "allow_new_short": False,
        "preferred_strategy": "trim_profits",
        "stop_loss_mult": 0.6,
    },
    MarketRegime.UNKNOWN: {
        "position_mult": 0.5,
        "allow_new_long": True, "allow_new_short": False,
        "preferred_strategy": "balanced",
        "stop_loss_mult": 1.0,
    },
}


@dataclass
class RegimeSignal:
    regime: MarketRegime
    position_mult: float
    allow_new_long: bool
    allow_new_short: bool
    preferred_strategy: str
    stop_loss_mult: float
    trend: TrendResult
    vol: VolResult
    breadth: BreadthResult
    reasons: list[str]
    confidence: float              # 0-1

    def to_agent_context(self) -> str:
        """给 LLM agent 的 regime 上下文字符串."""
        return (
            f"regime={self.regime.value} (信心{self.confidence:.1%}) | "
            f"趋势={self.trend.direction}({self.trend.strength:.1f}) | "
            f"波动={self.vol.level}({self.vol.vol_20d:.1%}) | "
            f"赚钱效应={self.breadth.money_effect} | "
            f"流动性={self.breadth.liquidity_level} | "
            f"建议={self.preferred_strategy} | "
            f"仓位乘数={self.position_mult:.1f}"
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["regime"] = self.regime.value
        return d


class RegimeDetector:
    def __init__(self):
        pass

    def detect(
        self,
        index_df: pd.DataFrame,                # 需含 close, 最好含 high/low, 长度>=200
        stocks_daily: pd.DataFrame | None = None,  # 当日全市场个股快照, 含 pct_chg
        total_turnover_yi: float | None = None,    # 两市成交额亿元
    ) -> RegimeSignal:
        """综合各类指标输出 regime."""
        if len(index_df) < 60:
            logger.warning("指数数据太少, regime=UNKNOWN")
            return self._unknown_signal()

        # 1. 计算基础指标
        trend = compute_trend(index_df["close"])
        vol = compute_volatility(index_df)
        breadth = (compute_breadth(stocks_daily, total_turnover_yi)
                   if stocks_daily is not None else
                   BreadthResult(0.5, 0.01, 0.01, "温", "normal"))

        # 2. 极端情况优先
        reasons = []
        if detect_crash(index_df["close"]):
            reasons.append("单日/5日跌幅越线")
            return self._build(MarketRegime.CRASH, trend, vol, breadth,
                               reasons, confidence=0.9)

        if detect_euphoria(breadth.pct_up, breadth.pct_limit_up, vol.vol_20d):
            reasons.append(f"赚钱效应{breadth.money_effect}+涨停率"
                           f"{breadth.pct_limit_up:.1%}+高波动")
            return self._build(MarketRegime.EUPHORIA, trend, vol, breadth,
                               reasons, confidence=0.75)

        # 3. 主 regime 分类
        if trend.direction == "up":
            reasons.append(f"趋势上行 (MA spread={trend.ma_spread:+.1%})")
            if vol.level in ("high", "extreme"):
                reasons.append(f"波动{vol.level}, 降级为 bull_quiet 保守")
                regime = MarketRegime.BULL_QUIET
            else:
                regime = MarketRegime.BULL_TRENDING
        elif trend.direction == "down":
            reasons.append(f"趋势下行 (MA spread={trend.ma_spread:+.1%})")
            if vol.level == "low":
                regime = MarketRegime.BEAR_QUIET
            else:
                regime = MarketRegime.BEAR_TRENDING
        else:
            reasons.append("均线纠缠, 震荡")
            regime = MarketRegime.CHOPPY

        # 4. 流动性修正: 低流动性降级仓位
        if breadth.liquidity_level == "low":
            reasons.append("流动性不足, 建议压降仓位")

        confidence = self._estimate_confidence(trend, vol, breadth)
        return self._build(regime, trend, vol, breadth, reasons, confidence)

    def _build(self, regime: MarketRegime,
               trend: TrendResult, vol: VolResult, breadth: BreadthResult,
               reasons: list[str], confidence: float) -> RegimeSignal:
        rule = REGIME_RULES[regime]
        # 低流动性乘数再乘 0.8
        pos_mult = rule["position_mult"]
        if breadth.liquidity_level == "low":
            pos_mult *= 0.8
        return RegimeSignal(
            regime=regime,
            position_mult=pos_mult,
            allow_new_long=rule["allow_new_long"],
            allow_new_short=rule["allow_new_short"],
            preferred_strategy=rule["preferred_strategy"],
            stop_loss_mult=rule["stop_loss_mult"],
            trend=trend, vol=vol, breadth=breadth,
            reasons=reasons, confidence=confidence,
        )

    def _unknown_signal(self) -> RegimeSignal:
        return self._build(
            MarketRegime.UNKNOWN,
            TrendResult("flat", 0, 0, 0),
            VolResult(0, 0, 1, 0, "normal"),
            BreadthResult(0.5, 0.01, 0.01, "温", "normal"),
            reasons=["数据不足"], confidence=0.0,
        )

    def _estimate_confidence(self, t: TrendResult, v: VolResult,
                             b: BreadthResult) -> float:
        """多指标一致性越高, 信心越高."""
        votes = 0
        if t.direction == "up" and b.pct_up > 0.5:
            votes += 1
        if t.direction == "down" and b.pct_up < 0.4:
            votes += 1
        if v.level in ("normal", "low"):
            votes += 0.5
        if b.liquidity_level == "high":
            votes += 0.5
        if t.strength > 0.3:
            votes += 0.5
        return min(1.0, votes / 3.0)
