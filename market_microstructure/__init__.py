"""市场微结构因子 - Level2 真正的 alpha 来源, 非日线数据可及.

大众以为 Level2 = "看得更快", 实际 Level2 = "看得见真相":
  - 委托队列不平衡 (Order Imbalance Ratio, OIR)
  - 知情交易概率 (VPIN) - 诺贝尔奖得主 Easley-O'Hara 提出
  - 撤单率 (Cancel Ratio) - 幌骗交易 (spoofing) 代理
  - 买卖价差 (Bid-Ask Spread) 与实际成交价偏差
  - 大单切割 (large order slicing) 探测

Almgren-Chriss 冲击成本模型告诉你:
  真实成本 = 常数 × sqrt(成交金额 / 日成交额)
  而大众回测只用固定 bps 滑点, 对小盘票严重低估.
"""
from .impact import (
    almgren_chriss_impact, square_root_impact,
    estimate_participation_rate, kyle_lambda,
)
from .order_flow import (
    order_imbalance_ratio, weighted_oir, vpin, cancel_ratio,
    trade_direction_classify, lee_ready_classify,
)
from .spread_factors import (
    effective_spread, realized_spread,
    depth_weighted_midprice, roll_implicit_spread,
    amihud_illiquidity,
)

__all__ = [
    # 冲击成本
    "almgren_chriss_impact", "square_root_impact",
    "estimate_participation_rate", "kyle_lambda",
    # 订单流
    "order_imbalance_ratio", "weighted_oir",
    "vpin", "cancel_ratio",
    "trade_direction_classify", "lee_ready_classify",
    # 价差
    "effective_spread", "realized_spread",
    "depth_weighted_midprice", "roll_implicit_spread",
    "amihud_illiquidity",
]
