"""标签工程 (Label Engineering) - 大众量化师最容易漏掉但影响巨大的一环.

大众做法:
    y = close[t+5] / close[t+1] - 1       # 5 日 forward return

真相:
    1) close/close 包含了 open 跳空的噪音, 应该用 close[t+5]/open[t+1]
       (因为信号是今日收盘后决策, 下一日开盘才能买入)
    2) 原始收益 label 被高波动股票主导 -> 用 ATR 归一化
    3) Top 1% / Bottom 1% 的标签是噪音, 不是 alpha -> winsorize
    4) 单 horizon 训练过拟合噪音 -> 多 horizon 加权融合
    5) 涨跌停 / 停牌 / 公告窗口的 label 应屏蔽 (买不到 / 未来函数)
"""
from .horizons import (
    multi_horizon_label, vol_adjusted_label,
    overnight_label, intraday_label,
)
from .masks import (
    tradeable_mask, event_window_mask,
    disclosure_vs_report_check, timestamp_integrity_check,
    leaky_label_detector,
)

__all__ = [
    "multi_horizon_label", "vol_adjusted_label",
    "overnight_label", "intraday_label",
    "tradeable_mask", "event_window_mask",
    "disclosure_vs_report_check", "timestamp_integrity_check",
    "leaky_label_detector",
]
