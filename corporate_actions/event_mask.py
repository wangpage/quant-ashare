"""事件窗口屏蔽器."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass
class EventWindow:
    code: str
    event_date: pd.Timestamp
    pre_days: int
    post_days: int
    event_type: str


class _BaseMask:
    """所有屏蔽器的接口."""
    def __init__(self, events: list[EventWindow]):
        self.events = events

    def apply(self, df: pd.DataFrame) -> pd.Series:
        """返回 bool Series, True=样本可用, False=屏蔽."""
        if df.empty or not self.events:
            return pd.Series(True, index=df.index)
        assert "code" in df.columns and "date" in df.columns
        mask = pd.Series(True, index=df.index)
        dates = pd.to_datetime(df["date"])
        for ev in self.events:
            stock_idx = df["code"] == ev.code
            win_start = ev.event_date - pd.Timedelta(days=ev.pre_days)
            win_end = ev.event_date + pd.Timedelta(days=ev.post_days)
            mask &= ~(stock_idx & (dates >= win_start) & (dates <= win_end))
        return mask


class EarningsMask(_BaseMask):
    """财报披露日 [-2, +1] 屏蔽.

    理由:
        - 披露前 2 天常有信息泄露 / 业绩预告
        - 披露日 ±1 天波动极大, 不适合做 systematic 训练
    """
    def __init__(self, earnings: dict[str, list[str]],
                 pre: int = 2, post: int = 1):
        events = []
        for code, dates in earnings.items():
            for d in dates:
                events.append(EventWindow(
                    code, pd.to_datetime(d), pre, post, "earnings",
                ))
        super().__init__(events)


class UnlockMask(_BaseMask):
    """解禁日 [-10, +3] 屏蔽.

    理由:
        - 大股东解禁前 10 天价格会提前承压 (内部人通知亲属出货)
        - 解禁后 3 天集中抛压
        - 这段时间因子几乎全部失效, 强制屏蔽避免模型学到错误信号
    """
    def __init__(self, unlocks: dict[str, list[tuple[str, float]]],
                 pre: int = 10, post: int = 3,
                 min_unlock_ratio: float = 0.03):
        events = []
        for code, items in unlocks.items():
            for d, ratio in items:
                if ratio < min_unlock_ratio:
                    continue      # 小解禁不屏蔽
                events.append(EventWindow(
                    code, pd.to_datetime(d), pre, post, "unlock",
                ))
        super().__init__(events)


class BlockTradeMask(_BaseMask):
    """大宗交易 [0, +5] 屏蔽 (折价 > 3% 才屏蔽).

    为什么重要:
        大宗折价 > 3% = 大股东/机构急于出货.
        之后 5 个交易日内该股 alpha 噪声极大, 模型别碰.
    """
    def __init__(self, block_trades: dict[str, list[tuple[str, float]]],
                 discount_threshold: float = 0.03,
                 pre: int = 0, post: int = 5):
        events = []
        for code, items in block_trades.items():
            for d, discount in items:
                if abs(discount) < discount_threshold:
                    continue
                events.append(EventWindow(
                    code, pd.to_datetime(d), pre, post, "block_trade",
                ))
        super().__init__(events)


def combined_event_mask(
    df: pd.DataFrame,
    *masks: _BaseMask,
) -> pd.Series:
    """组合多个屏蔽器, 所有条件都满足才可用."""
    if not masks:
        return pd.Series(True, index=df.index)
    result = masks[0].apply(df)
    for m in masks[1:]:
        result &= m.apply(df)
    return result
