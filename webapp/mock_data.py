"""Mock 数据提供者 - 无真实 API / LLM / 数据库时可直接展示页面."""
from __future__ import annotations

from datetime import date, datetime, timedelta
import random

import numpy as np
import pandas as pd


# ==================== 市场状态 ====================
def get_market_regime() -> dict:
    return {
        "regime": "bull_trending",
        "position_mult": 1.0,
        "confidence": 0.78,
        "trend_direction": "up",
        "trend_strength": 0.42,
        "vol_level": "normal",
        "money_effect": "温",
        "liquidity_level": "high",
        "breadth_pct_up": 0.58,
        "limit_up_count": 48,
        "limit_down_count": 12,
        "reasons": [
            "MA20 方向上行 (ma_spread=+3.2%)",
            "赚钱效应温, 58% 个股上涨",
            "两市成交额 11200 亿, 流动性高",
        ],
    }


# ==================== 主题评分 ====================
def get_theme_scores() -> pd.DataFrame:
    data = [
        ("AI 算力国产化",     6.04, 9.27, 5.32, 1.95, ["688256", "688041", "300474"]),
        ("半导体+HBM",         4.73, 6.87, 3.91, 1.99, ["600584", "002156", "688012"]),
        ("小金属战略反制",     3.96, 3.05, 0.00, 8.37, ["600549", "002155", "600111"]),
        ("军工+商业航天",      3.57, 6.93, 0.96, 0.62, ["600760", "000768", "600118"]),
        ("人形机器人",         3.51, 3.63, 5.54, 0.00, ["002031", "601100", "603728"]),
        ("可控核聚变",         3.40, 7.55, 0.01, 0.00, ["603011", "688135", "002130"]),
        ("红利+中特估",        2.83, 2.91, 0.00, 4.26, ["601088", "600028", "600900"]),
        ("新能源出海",         2.29, 1.25, 0.09, 4.34, ["300750", "002594", "601012"]),
    ]
    df = pd.DataFrame(data, columns=[
        "主题", "总分", "政策", "巨头", "地缘", "龙头 top3",
    ])
    df["龙头 top3"] = df["龙头 top3"].apply(lambda x: ", ".join(x))
    return df


# ==================== 今日信号 ====================
def get_today_signals(top_k: int = 20) -> pd.DataFrame:
    rows = [
        ("300750", "宁德时代", 0.87, "buy",  0.12, ["新能源出海"],      168),
        ("002156", "通富微电", 0.82, "buy",  0.10, ["半导体+HBM"],       85),
        ("600584", "长电科技", 0.79, "buy",  0.10, ["半导体+HBM"],       75),
        ("688256", "寒武纪",   0.77, "buy",  0.08, ["AI 算力"],         650),
        ("600760", "中航沈飞", 0.74, "buy",  0.08, ["军工"],             88),
        ("002155", "湖南黄金", 0.72, "buy",  0.07, ["小金属"],            35),
        ("600549", "厦门钨业", 0.70, "buy",  0.07, ["小金属"],            26),
        ("601100", "恒立液压", 0.68, "hold", 0.00, ["人形机器人"],        85),
        ("603728", "鸣志电器", 0.65, "hold", 0.00, ["人形机器人"],       100),
        ("000977", "浪潮信息", 0.64, "buy",  0.06, ["AI 算力"],          75),
        ("300308", "中际旭创", 0.62, "buy",  0.06, ["AI 算力"],         215),
        ("603011", "合锻智能", 0.60, "buy",  0.05, ["核聚变"],            22),
        ("601088", "中国神华", 0.55, "hold", 0.00, ["红利+中特估"],       41),
        ("600519", "贵州茅台", 0.52, "hold", 0.00, [],                 1476),
        ("688012", "中微公司", 0.50, "buy",  0.04, ["半导体+HBM"],      195),
    ]
    df = pd.DataFrame(rows, columns=[
        "代码", "名称", "模型分", "动作", "仓位", "主题", "参考价",
    ])
    df["主题"] = df["主题"].apply(lambda x: ", ".join(x) if x else "-")
    df.insert(0, "排名", range(1, len(df) + 1))
    return df.head(top_k)


# ==================== 资金分配 ====================
def get_portfolio_allocation() -> pd.DataFrame:
    return pd.DataFrame([
        {"主题": "AI 算力",       "分配(元)": 264000, "占比": 0.264},
        {"主题": "半导体+HBM",    "分配(元)": 207000, "占比": 0.207},
        {"主题": "小金属战略反制",  "分配(元)": 173000, "占比": 0.173},
        {"主题": "军工+商业航天",   "分配(元)": 156000, "占比": 0.156},
        {"主题": "现金保留",       "分配(元)": 200000, "占比": 0.200},
    ])


# ==================== 个股详情 ====================
def get_stock_kline(code: str, days: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(hash(code) % 2**32)
    n = days
    base = {
        "300750": 450, "600519": 1476, "002156": 85, "600584": 75,
        "688256": 650, "600760": 88, "002155": 35, "600549": 26,
        "601100": 85, "603728": 100,
    }.get(code, 50)
    rets = rng.normal(0.0008, 0.022, n)
    close = base * np.cumprod(1 + rets)
    open_ = close * (1 + rng.normal(0, 0.005, n))
    high = np.maximum(close, open_) * (1 + rng.uniform(0, 0.015, n))
    low = np.minimum(close, open_) * (1 - rng.uniform(0, 0.015, n))
    volume = rng.integers(1_000_000, 20_000_000, n).astype(float)
    dates = pd.date_range(end=date.today(), periods=n, freq="B")
    return pd.DataFrame({
        "date": dates, "open": open_, "high": high, "low": low,
        "close": close, "volume": volume,
    })


def get_stock_factors(code: str) -> pd.DataFrame:
    rng = np.random.default_rng(hash(code) % 2**32)
    factors = [
        ("REV5",        rng.uniform(-0.5, 0.5),   "反转 5日"),
        ("TURN_RATIO20", rng.uniform(0.5, 2.5),   "换手比 20日"),
        ("VOL20",       rng.uniform(0.15, 0.4),   "波动率 20日"),
        ("BOLL_POS",    rng.uniform(-1, 1),       "布林位置"),
        ("MA_DIFF20",   rng.uniform(-0.1, 0.15),  "MA20 偏离"),
        ("AMP_MA20",    rng.uniform(0.02, 0.08),  "平均振幅"),
        ("GAP1",        rng.uniform(-0.03, 0.03), "昨缺口"),
        ("LIMIT_UP20",  rng.integers(0, 4),       "近20日涨停"),
        ("TREND10",     rng.integers(-10, 10),    "趋势一致性"),
        ("AMT_MOM",     rng.uniform(0.5, 2.0),    "成交额动量"),
    ]
    df = pd.DataFrame(factors, columns=["因子", "数值", "说明"])
    df["数值"] = df["数值"].round(3)
    return df


def get_stock_themes(code: str) -> list[str]:
    mapping = {
        "300750": ["新能源出海", "人形机器人"],
        "600519": [],
        "002156": ["半导体+HBM"],
        "600584": ["半导体+HBM"],
        "688256": ["AI 算力"],
        "600760": ["军工+商业航天"],
        "002155": ["小金属战略反制"],
        "600549": ["小金属战略反制"],
        "601100": ["人形机器人"],
        "603728": ["人形机器人"],
        "603011": ["核聚变"],
    }
    return mapping.get(code, [])


def get_stock_info(code: str) -> dict:
    mapping = {
        "300750": {"name": "宁德时代", "industry": "电池",       "market_cap": "1.4万亿"},
        "600519": {"name": "贵州茅台", "industry": "白酒",       "market_cap": "1.8万亿"},
        "002156": {"name": "通富微电", "industry": "半导体",     "market_cap": "1200亿"},
        "600584": {"name": "长电科技", "industry": "半导体",     "market_cap": "1400亿"},
        "688256": {"name": "寒武纪",   "industry": "AI 芯片",    "market_cap": "2600亿"},
        "600760": {"name": "中航沈飞", "industry": "军工",       "market_cap": "2300亿"},
        "002155": {"name": "湖南黄金", "industry": "有色金属",   "market_cap": "370亿"},
        "600549": {"name": "厦门钨业", "industry": "有色金属",   "market_cap": "390亿"},
        "601100": {"name": "恒立液压", "industry": "机械",       "market_cap": "1200亿"},
        "603728": {"name": "鸣志电器", "industry": "电机",       "market_cap": "180亿"},
        "603011": {"name": "合锻智能", "industry": "专用设备",   "market_cap": "85亿"},
    }
    base = mapping.get(code, {
        "name": code, "industry": "未知", "market_cap": "N/A",
    })
    base["code"] = code
    return base


# ==================== Agent 辩论 ====================
def get_recent_debates() -> list[dict]:
    """最近 N 次多智能体决策的辩论记录."""
    return [
        {
            "id": "dec_20260420_300750",
            "datetime": "2026-04-20 15:30",
            "code": "300750",
            "name": "宁德时代",
            "final_action": "buy",
            "final_conviction": 0.72,
            "final_score": 0.42,
            "risk_decision": "approve",
        },
        {
            "id": "dec_20260420_688256",
            "datetime": "2026-04-20 15:28",
            "code": "688256",
            "name": "寒武纪",
            "final_action": "buy",
            "final_conviction": 0.68,
            "final_score": 0.51,
            "risk_decision": "modify",
        },
        {
            "id": "dec_20260420_600519",
            "datetime": "2026-04-20 15:20",
            "code": "600519",
            "name": "贵州茅台",
            "final_action": "hold",
            "final_conviction": 0.45,
            "final_score": 0.12,
            "risk_decision": "approve",
        },
    ]


def get_debate_detail(decision_id: str) -> dict:
    """单次决策的完整辩论日志."""
    return {
        "id": decision_id,
        "code": "300750",
        "name": "宁德时代",
        "analysts": {
            "fundamental": {
                "view": "bullish",
                "score": 0.45,
                "thinking": "宁德时代估值已充分反应国内新能源放缓,但海外扩张(匈牙利/墨西哥/泰国)是第二增长曲线.",
                "reasoning": "1) 海外收入占比从 25% 向 40% 切换;  2) 储能业务 2027 年有望超车动力电池;  3) 麒麟电池 + 凝聚态电池技术壁垒仍在.",
                "risk": "欧盟反补贴关税 + 国内集采压力",
            },
            "technical": {
                "view": "neutral",
                "score": 0.15,
                "thinking": "突破 440 颈线但量能不足,需要 500 亿成交额确认.",
                "reasoning": "MA20 上行但距离前高 15%, 布林中轨震荡, MACD 即将金叉.",
                "risk": "若跌破 430 则形态失效",
            },
            "sentiment": {
                "view": "bullish",
                "score": 0.55,
                "thinking": "北向近 5 日净买入 32 亿, 融资余额上升 8%, 机构调研 8 次激增.",
                "reasoning": "外资回流新能源板块, 机构关注度回归.",
                "risk": "若北向单日转出则要警惕",
            },
        },
        "debate_rounds": [
            {
                "round": 0,
                "bull": "三面综合看多, 海外+储能+外资共振, 建议目标位 520",
                "bear": "短期量能不足, 估值仍高, 450-480 是短期顶",
            },
            {
                "round": 1,
                "bull": "回应空头: 海外订单已锁定 2026-2027, 450 估值对应 2026 业绩只有 22 倍 PE, 不高",
                "bear": "反驳: 欧盟反补贴关税 12 月生效, 海外订单可能重谈, 220 亿利润下修风险",
            },
        ],
        "judge": {
            "solution": "bullish",
            "score": 0.42,
            "conviction": 0.72,
            "reasoning": "海外+储能双线兑现节奏强于风险因素, 空头反补贴担忧已 price-in",
            "explanation": "12 月反补贴落地之前, 仓位适度配置; 反补贴结果公布后根据条款再决定是否加仓",
        },
        "trader": {
            "action": "buy",
            "size_pct": 0.12,
            "entry_price": 451.0,
            "stop_loss_price": 425.0,
            "take_profit_price": 520.0,
            "holding_days": 20,
            "reasoning": "按凯利建议仓位 12%, 止损 5.8%, 目标收益 15%",
        },
        "risk_review": {
            "action": "approve",
            "reasoning": "当日仓位总计 38%, 行业集中度 18% 未超 30% 红线; 通过",
        },
    }


# ==================== 回测 ====================
def get_backtest_report() -> dict:
    n = 250
    dates = pd.date_range(end=date.today(), periods=n, freq="B")
    np.random.seed(42)
    strat_ret = np.random.normal(0.0008, 0.012, n)
    bench_ret = np.random.normal(0.0003, 0.011, n)
    strat_nav = np.cumprod(1 + strat_ret)
    bench_nav = np.cumprod(1 + bench_ret)

    ic_daily = np.random.normal(0.035, 0.10, n)
    ic_20d = pd.Series(ic_daily).rolling(20).mean()

    curve = pd.DataFrame({
        "date": dates,
        "strategy": strat_nav,
        "benchmark": bench_nav,
        "excess": strat_nav - bench_nav,
    })

    ic_df = pd.DataFrame({
        "date": dates,
        "ic_daily": ic_daily,
        "ic_20d": ic_20d,
    })

    stats = {
        "period":           "2025-04-20 → 2026-04-20",
        "annual_return":    0.2603,
        "annual_vol":       0.1890,
        "sharpe":           1.38,
        "max_drawdown":     -0.1245,
        "win_rate":         0.585,
        "avg_turnover":     0.048,
        "ic_mean":          0.041,
        "icir":             0.95,
        "bench_annual":     0.0821,
        "excess_annual":    0.1782,
        "trade_count":      186,
    }

    return {"curve": curve, "ic": ic_df, "stats": stats}
