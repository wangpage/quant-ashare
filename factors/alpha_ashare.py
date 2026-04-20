"""A股特化 Alpha 因子库 - qlib Expression 风格.

qlib Expression 速查:
  $close, $open, $high, $low, $volume, $amount
  Ref($close, N)       N日前的收盘
  Mean($close, 20)     20日均值
  Std($close, 20)      20日标准差
  Rank($close, 20)     截面/时序排序
  Corr(x, y, 20)       滚动相关系数
  Sum(x, 20)           滚动求和

这里定义的因子都是在 qlib.contrib.data.handler.Alpha158 基础上
针对 A股短线 追加的因子。与 Alpha158 合并后得到 ~180 维特征。
"""
from __future__ import annotations

from typing import Tuple

# ============ A股短线特化 Alpha ============
# (expression, name) 列表, qlib 会自动向量化计算

ASHARE_ALPHA: list[Tuple[str, str]] = [
    # ---- 1. 反转因子 (A股散户最稳 alpha) ----
    ("-1 * ($close/Ref($close, 5) - 1)", "REV5"),
    ("-1 * ($close/Ref($close, 10) - 1)", "REV10"),
    ("-1 * ($close/Ref($close, 20) - 1)", "REV20"),

    # ---- 2. 换手率异动 (A股散户情绪代理) ----
    ("$volume / Mean($volume, 20)", "TURN_RATIO20"),
    ("$volume / Mean($volume, 5)",  "TURN_RATIO5"),
    ("Std($volume, 20) / Mean($volume, 20)", "TURN_VOL20"),
    ("Rank($volume/Mean($volume, 20), 60)", "TURN_RANK"),

    # ---- 3. 量价关系 ----
    ("Corr($close, Log($volume+1), 10)", "CORR_PV10"),
    ("Corr($close, Log($volume+1), 20)", "CORR_PV20"),
    ("Corr(Rank($close, 10), Rank($volume, 10), 10)", "RANK_CORR10"),

    # ---- 4. 波动率因子 (高波动高收益预警) ----
    ("Std($close/Ref($close, 1) - 1, 20)", "VOL20"),
    ("Std($close/Ref($close, 1) - 1, 5)",  "VOL5"),
    ("(Std($close/Ref($close, 1)-1, 5) + 1e-12) / (Std($close/Ref($close, 1)-1, 20) + 1e-12)", "VOL_RATIO"),
    ("(Mean($high - $low, 10) + 1e-12) / ($close + 1e-12)", "ATR10"),
    ("(Mean($high - $low, 20) + 1e-12) / ($close + 1e-12)", "ATR20"),

    # ---- 5. 涨停板因子 (A股独有) ----
    # 近20日涨停次数 (涨停 ≈ pct_chg > 0.095)
    ("Sum(If($close/Ref($close,1) - 1 > 0.095, 1, 0), 20)", "LIMIT_UP20"),
    # 近20日跌停次数
    ("Sum(If($close/Ref($close,1) - 1 < -0.095, 1, 0), 20)", "LIMIT_DN20"),
    # 最近一次涨停距今天数
    ("Sum(If($close/Ref($close,1) - 1 > 0.095, 1, 0), 5)", "LIMIT_UP5"),

    # ---- 6. 缺口 (gap) ----
    ("$open / Ref($close, 1) - 1", "GAP1"),
    ("Mean($open/Ref($close, 1) - 1, 5)", "GAP5_MA"),

    # ---- 7. 均线偏离 ----
    ("$close / Mean($close, 5) - 1",  "MA_DIFF5"),
    ("$close / Mean($close, 10) - 1", "MA_DIFF10"),
    ("$close / Mean($close, 20) - 1", "MA_DIFF20"),
    ("$close / Mean($close, 60) - 1", "MA_DIFF60"),

    # ---- 8. K线形态 ----
    # 实体占比 (收-开)/(高-低)
    ("($close - $open) / ($high - $low + 1e-12)", "BODY_RATIO"),
    # 上影线占比
    ("($high - Greater($close, $open)) / ($high - $low + 1e-12)", "UP_SHADOW"),
    # 下影线占比
    ("(Less($close, $open) - $low) / ($high - $low + 1e-12)", "DN_SHADOW"),

    # ---- 9. Bollinger 通道 ----
    ("($close - Mean($close, 20)) / (Std($close, 20) * 2 + 1e-12)", "BOLL_POS"),

    # ---- 10. 动量/反转混合 ----
    ("$close / Ref($close, 20) - 1 - ($close / Ref($close, 5) - 1)", "MOM_REV_MIX"),

    # ---- 11. 成交额动量 ----
    ("Mean($amount, 5) / Mean($amount, 20)", "AMT_MOM"),
    ("Std($amount, 20) / (Mean($amount, 20) + 1e-12)", "AMT_VOL"),

    # ---- 12. 高低价突破 ----
    ("($close - Min($low, 20)) / (Max($high, 20) - Min($low, 20) + 1e-12)", "POS_IN_RANGE20"),
    ("($close - Min($low, 60)) / (Max($high, 60) - Min($low, 60) + 1e-12)", "POS_IN_RANGE60"),

    # ---- 13. KDJ-style ----
    ("(Max($high, 9) - $close) / (Max($high, 9) - Min($low, 9) + 1e-12)", "KDJ_K"),

    # ---- 14. 成交量突变 ----
    ("If($volume > 2*Mean($volume, 20), 1, 0)", "VOL_SPIKE"),

    # ---- 15. 振幅 ----
    ("($high - $low) / (Ref($close, 1) + 1e-12)", "AMP"),
    ("Mean(($high - $low) / (Ref($close, 1) + 1e-12), 20)", "AMP_MA20"),

    # ---- 16. 趋势一致性 ----
    ("Sum(If($close > Ref($close, 1), 1, -1), 10)", "TREND10"),
]

ASHARE_ALPHA_FIELDS: list[str] = [expr for expr, _ in ASHARE_ALPHA]
ASHARE_ALPHA_NAMES:  list[str] = [name for _, name in ASHARE_ALPHA]


# ============ qlib DataHandlerLP 子类 ============
try:
    from qlib.contrib.data.handler import Alpha158
    from qlib.data.dataset.processor import (
        CSRankNorm, CSZScoreNorm, DropnaProcessor, Fillna, RobustZScoreNorm
    )

    class AshareAlphaHandler(Alpha158):
        """Alpha158 + A股特化因子."""

        def get_feature_config(self):
            fields, names = super().get_feature_config()
            fields = list(fields) + ASHARE_ALPHA_FIELDS
            names  = list(names)  + ASHARE_ALPHA_NAMES
            return fields, names

except ImportError:
    # qlib 未安装时占位
    class AshareAlphaHandler:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("pip install pyqlib 后再使用 AshareAlphaHandler")
