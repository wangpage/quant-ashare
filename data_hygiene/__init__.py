"""数据清洗暗门 - 回测 vs 实盘差距 80% 来自这里.

头部机构每周跑数据诊断, 发现这些陷阱就重新回测:
  1. 幸存偏差: 数据源只包含现存公司, 退市的被悄悄删除
  2. 前视偏差: feature 用到了 t+k 信息 (如财报报告期 vs 披露日)
  3. 复权错误: 跨除权日计算 return 不做调整
  4. 时钟错位: 数据源毫秒 vs 交易所毫秒 差 50-200ms
  5. 停牌填充: 直接用前值会让"停牌恢复日"看起来像正常交易
  6. 代码变更: 000022 改为 001872, 断档
  7. 行业变更: 股票一级行业可能每 2-3 年调整, 历史行业不准

使用方法:
    from data_hygiene import DataHealthChecker
    checker = DataHealthChecker()
    report = checker.audit_full(df, start='2020-01-01')
    print(report)
"""
from .survivorship import (
    detect_survivorship_bias, delisted_stock_checker,
)
from .lookahead import (
    scan_lookahead_bias, time_index_integrity_check,
)
from .adjustment import (
    detect_price_jumps, verify_adjustment_factor,
)
from .gaps import (
    find_suspension_days, gap_aware_fill,
)
from .timezone_sync import (
    clock_skew_detector, align_to_exchange_time,
)
from .audit import DataHealthChecker, AuditReport

__all__ = [
    "detect_survivorship_bias", "delisted_stock_checker",
    "scan_lookahead_bias", "time_index_integrity_check",
    "detect_price_jumps", "verify_adjustment_factor",
    "find_suspension_days", "gap_aware_fill",
    "clock_skew_detector", "align_to_exchange_time",
    "DataHealthChecker", "AuditReport",
]
