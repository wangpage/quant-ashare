"""统一数据门面 - Mock/Real 自动切换.

DATA_MODE 环境变量:
    "mock" (默认): 使用 mock_data, 无需真实 API
    "real":       尝试接真实 akshare / qlib / LLM, 失败回退 mock
"""
from __future__ import annotations

import os
from functools import lru_cache

from . import mock_data as _mock


DATA_MODE = os.getenv("QUANT_WEB_MODE", "mock").lower()


def is_mock() -> bool:
    return DATA_MODE == "mock"


# ==================== 对外 API ====================
def get_market_regime():
    """Real 模式: 拉沪深300指数 + 当日个股快照, 调 RegimeDetector.
    失败 fallback 到 mock, 并在返回 dict 里标记 data_source.
    """
    if DATA_MODE == "real":
        try:
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            from webapp.live_regime import fetch_live_regime
            return fetch_live_regime()
        except Exception as e:
            m = _mock.get_market_regime()
            m["data_source"] = f"mock (live 失败: {e.__class__.__name__})"
            return m
    result = _mock.get_market_regime()
    result["data_source"] = "mock"
    return result


def get_theme_scores():
    return _mock.get_theme_scores()


def get_today_signals(top_k: int = 20):
    return _mock.get_today_signals(top_k)


def get_portfolio_allocation():
    return _mock.get_portfolio_allocation()


def get_stock_kline(code: str, days: int = 120):
    if DATA_MODE == "real":
        try:
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            from data_adapter.em_direct import fetch_daily
            df = fetch_daily(code, start="20250101", end="20261231")
            if not df.empty:
                df = df.tail(days).copy()
                df["date"] = pd.to_datetime(df["date"])
                return df[["date", "open", "high", "low", "close", "volume"]]
        except Exception:
            pass
    return _mock.get_stock_kline(code, days)


def get_stock_factors(code: str):
    return _mock.get_stock_factors(code)


def get_stock_themes(code: str):
    return _mock.get_stock_themes(code)


def get_stock_info(code: str):
    return _mock.get_stock_info(code)


def get_recent_debates():
    return _mock.get_recent_debates()


def get_debate_detail(decision_id: str):
    return _mock.get_debate_detail(decision_id)


def get_backtest_report():
    return _mock.get_backtest_report()
