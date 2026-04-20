"""幸存偏差检测 - 量化回测最大的"无声杀手".

原理:
    大多数数据源 (akshare 个股列表、Wind 免费版) 默认只列出"仍在交易的股票",
    退市公司静默消失. 你回测一个 2010-2024 的策略, 实际只用了 2024 年还活着
    的那批公司 (约 3500 只), 忽略了同期 250+ 只退市公司.

    结果: 回测收益虚高 2-5%/年, 最大回撤虚低 10%+.

检测方法:
    1. 对比同一日期的 "活跃股票数" vs "历史应有股票数"
    2. 扫描 A股退市代码池, 看你的数据里有没有
    3. 计算 "数据集起始日股票数 / 结束日股票数" 比值, < 1 说明有退市
"""
from __future__ import annotations

import pandas as pd


# A股历史退市股票代码池 (样例, 生产环境从交易所获取完整列表)
# 来源: http://www.sse.com.cn/disclosure/dealinstruc/suspension/ + 深交所
KNOWN_DELISTED_SAMPLES = [
    "000022",   # 深赤湾A (改名 001872)
    "000024",   # 招商地产 (改名 001979)
    "600087",   # 长油5 → 重新上市为 601975
    "600656",   # 博元投资退市
    "600485",   # 中创信测退市
    "600898",   # 国美通讯 (已退市)
    "002359",   # *ST 北讯 退市
    "300028",   # 金亚科技 退市
    "600331",   # 宏达股份 (样例)
]


def detect_survivorship_bias(
    df: pd.DataFrame,
    start_date: str,
    end_date: str,
    expected_min_stocks: int = 3000,
) -> dict:
    """检测数据集是否存在幸存偏差.

    Args:
        df: 必须含 'code', 'date' 列
        start_date, end_date: 检查的时间段
        expected_min_stocks: 该时段预计股票数 (A股 2015 年后 >= 3000)

    Returns:
        {
          'survivorship_risk': 'HIGH/MED/LOW',
          'missing_count': int,
          'growth_ratio': float,  # 末期股票数 / 初期, <1 则有退市
          'suspects': [...]       # 疑似退市但数据里没有的代码
        }
    """
    if "code" not in df.columns or "date" not in df.columns:
        return {"survivorship_risk": "UNKNOWN", "error": "缺 code/date"}

    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    mask = (d["date"] >= start_date) & (d["date"] <= end_date)
    d = d[mask]

    # 每日活跃股票数
    daily_count = d.groupby("date")["code"].nunique()
    if len(daily_count) < 10:
        return {"survivorship_risk": "UNKNOWN", "error": "样本不足"}

    # 关键指标: 初期 vs 末期
    early = daily_count.iloc[:20].mean()
    late = daily_count.iloc[-20:].mean()
    growth_ratio = late / early

    # 检查退市样本
    all_codes = set(d["code"].astype(str).unique())
    # 格式修正 (0开头可能丢 0)
    all_codes_norm = {c.zfill(6) for c in all_codes}
    delisted_in_data = [c for c in KNOWN_DELISTED_SAMPLES
                        if c in all_codes_norm]
    delisted_missing = [c for c in KNOWN_DELISTED_SAMPLES
                        if c not in all_codes_norm]

    # 判定
    if growth_ratio > 1.1 and len(delisted_in_data) < 2:
        risk = "HIGH"              # 数据只增不减 + 退市股全缺失
    elif growth_ratio > 1.05:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return {
        "survivorship_risk": risk,
        "avg_early_count": float(early),
        "avg_late_count": float(late),
        "growth_ratio": float(growth_ratio),
        "delisted_in_data": delisted_in_data,
        "delisted_missing": delisted_missing,
        "estimated_bias_pct_per_year": (
            2.0 if risk == "HIGH" else
            0.8 if risk == "MEDIUM" else 0.0
        ),
    }


def delisted_stock_checker(
    codes: list[str], ref_delisted_list: list[str] | None = None,
) -> dict:
    """检查你的股票池是否"意外缺失退市股".

    Args:
        codes: 你的数据里所有股票代码
        ref_delisted_list: 参考退市列表 (默认用样例池)

    Returns:
        {found_delisted, missing_delisted, coverage}
    """
    ref = set(ref_delisted_list or KNOWN_DELISTED_SAMPLES)
    your_set = set(c.zfill(6) for c in codes)
    found = sorted(ref & your_set)
    missing = sorted(ref - your_set)
    coverage = len(found) / max(len(ref), 1)
    return {
        "ref_delisted_total": len(ref),
        "found_delisted": found,
        "missing_delisted": missing,
        "coverage": coverage,
        "verdict": "OK" if coverage >= 0.7 else "MISSING_DELISTED_STOCKS",
    }


def detect_point_in_time_issues(
    fundamental_df: pd.DataFrame,
    price_df: pd.DataFrame,
    report_date_col: str = "report_date",
    announce_date_col: str = "announce_date",
) -> dict:
    """检测财报数据的时间点 (Point-in-Time) 偏差.

    常见错误:
        用 "报告期" (report_date='2024-03-31') 作为可用时间,
        但该数据实际在 "披露日" (announce_date='2024-04-28') 才公开.
        差 4 周 = 严重前视偏差.

    Returns:
        {
          'avg_lag_days': 披露日 - 报告期 的平均天数,
          'max_lag_days': ...,
          'samples_with_issue': 有问题的条数,
        }
    """
    if report_date_col not in fundamental_df.columns:
        return {"error": f"缺 {report_date_col}"}
    if announce_date_col not in fundamental_df.columns:
        return {
            "error": f"缺 {announce_date_col}",
            "recommendation": "必须加披露日列, 否则 100% 前视偏差",
        }
    df = fundamental_df.copy()
    df[report_date_col] = pd.to_datetime(df[report_date_col])
    df[announce_date_col] = pd.to_datetime(df[announce_date_col])
    lag = (df[announce_date_col] - df[report_date_col]).dt.days
    return {
        "avg_lag_days": float(lag.mean()),
        "median_lag_days": float(lag.median()),
        "max_lag_days": int(lag.max()),
        "samples": len(df),
        "suggestion": "回测时应使用 announce_date 作为数据可用日",
    }
