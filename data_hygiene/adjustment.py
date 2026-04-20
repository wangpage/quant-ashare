"""复权因子 / 除权 验证.

陷阱:
    1. akshare 日线默认前复权, 但跨除权日前的数据会改变,
       导致某天回测的"历史 close" 今天变了另一个值.
    2. 未复权数据跨除权日, 10股送10股后股价 "突然腰斩",
       错误触发止损 / 动量失效.
    3. 配股 / 定增 / 分红 的复权逻辑各不同.

检测方法:
    a. 扫描相邻两日收益率 > 30% 的可疑跳点
    b. 交叉验证: 用不同源数据的除权点对齐
    c. 检查 factor 列单调性 (后复权应单调增)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def detect_price_jumps(
    df: pd.DataFrame,
    pct_threshold: float = 0.20,
    exclude_limit_up_down: bool = True,
) -> pd.DataFrame:
    """扫描可疑跳点 (可能是未正确处理的除权).

    Args:
        df: 含 ['code', 'date', 'close']
        pct_threshold: 单日收益率绝对值阈值, 超过视为可疑
        exclude_limit_up_down: 剔除 ±10% 涨跌停 (正常)

    Returns:
        可疑跳点 DataFrame, 按幅度倒序.
    """
    need = {"code", "date", "close"}
    if not need.issubset(df.columns):
        return pd.DataFrame()

    d = df.sort_values(["code", "date"]).copy()
    d["prev_close"] = d.groupby("code")["close"].shift(1)
    d["pct_change"] = d["close"] / d["prev_close"] - 1

    mask = d["pct_change"].abs() > pct_threshold
    if exclude_limit_up_down:
        mask &= d["pct_change"].abs() > 0.105      # 排除涨跌停
    suspects = d[mask].copy()
    suspects = suspects.sort_values("pct_change", key=lambda s: s.abs(),
                                     ascending=False)

    return suspects[["code", "date", "close", "prev_close", "pct_change"]]


def verify_adjustment_factor(df: pd.DataFrame) -> dict:
    """验证复权因子列的合理性.

    Args:
        df: 含 ['code', 'date', 'factor']

    后复权因子 (hfq): 应单调不减, 起始值为 1
    前复权因子 (qfq): 应单调不增, 末值为 1
    """
    if "factor" not in df.columns:
        return {"has_factor_col": False,
                "suggestion": "强烈建议增加 factor 列做复权校验"}

    issues = []
    for code, g in df.groupby("code"):
        g = g.sort_values("date")
        factors = g["factor"].dropna()
        if len(factors) < 2:
            continue

        # 是否所有值都相等 (未做复权)
        if factors.nunique() == 1:
            continue          # 不报警, 可能是新股没除权

        # 后复权单调递增 / 前复权单调递减
        monotonic_up = (factors.diff().dropna() >= 0).all()
        monotonic_down = (factors.diff().dropna() <= 0).all()
        if not (monotonic_up or monotonic_down):
            issues.append({
                "code": code,
                "issue": "复权因子非单调",
                "min": float(factors.min()),
                "max": float(factors.max()),
            })

    return {
        "has_factor_col": True,
        "non_monotonic_count": len(issues),
        "samples": issues[:5],
        "verdict": "PASS" if not issues else "FAIL",
    }


def cross_validate_with_raw(
    adjusted_df: pd.DataFrame, raw_df: pd.DataFrame,
    tolerance: float = 0.005,
) -> dict:
    """用未复权数据交叉验证复权数据质量.

    逻辑:
        raw_close × factor_ratio = adjusted_close ± 容差

    如果大量不匹配 = 复权处理有 bug.
    """
    need = {"code", "date", "close"}
    if not need.issubset(adjusted_df.columns) or not need.issubset(raw_df.columns):
        return {"error": "列不全"}

    merged = pd.merge(
        adjusted_df[["code", "date", "close"]].rename(columns={"close": "adj_close"}),
        raw_df[["code", "date", "close"]].rename(columns={"close": "raw_close"}),
        on=["code", "date"], how="inner",
    )
    if merged.empty:
        return {"error": "无重合样本"}

    merged["ratio"] = merged["adj_close"] / merged["raw_close"]
    # 同一股票的比值应该在某些日期不变, 除权日才变化
    merged["ratio_change"] = merged.groupby("code")["ratio"].diff().abs()
    abnormal = merged[merged["ratio_change"] > tolerance]

    return {
        "total_samples": len(merged),
        "abnormal_samples": len(abnormal),
        "abnormal_pct": len(abnormal) / len(merged),
        "suspicious_codes": abnormal["code"].value_counts().head(10).to_dict(),
    }


def fix_missing_adjustment(
    df: pd.DataFrame,
    dividend_events: list[tuple[str, str, float]],
) -> pd.DataFrame:
    """对未复权数据应用复权.

    Args:
        df: 含 ['code', 'date', 'close']
        dividend_events: [(code, ex_date, factor_multiplier)]
                          如送 10 股 10 股: factor = 0.5 之前价格 × 0.5

    Returns:
        新增 'factor' 和 'close_adj' 列.
    """
    out = df.sort_values(["code", "date"]).copy()
    out["factor"] = 1.0

    for code, ex_date, mult in dividend_events:
        mask = (out["code"] == code) & (out["date"] < ex_date)
        out.loc[mask, "factor"] = out.loc[mask, "factor"] * mult

    out["close_adj"] = out["close"] * out["factor"]
    return out
