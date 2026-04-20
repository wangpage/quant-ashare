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


def detect_small_dividend_gaps(
    df: pd.DataFrame,
    dividend_events: pd.DataFrame | None = None,
    small_gap_low: float = 0.015,
    small_gap_high: float = 0.08,
    volume_drop_threshold: float = 0.3,
) -> pd.DataFrame:
    """扫描小幅除权缺口 (分红 / 配股 / 转赠常见 1.5%-8%).

    >20% 跳点 detect_price_jumps 已能捕获, 但分红送股常在 2-5% 区间,
    与正常波动混杂, 单看涨跌幅无法分辨. 本函数联合两个信号判定:
        1) 跌幅落在 [small_gap_low, small_gap_high] (与 ±10% 涨跌停错开)
        2) 成交量同比大幅萎缩 (除权日次成交通常放量, 但价跳是刚性的)

    如提供 dividend_events, 则只标记未在事件表中的跳点 (= 漏复权).

    Args:
        df: 含 ['code', 'date', 'close', 'volume']
        dividend_events: 已知除权事件 [code, ex_date], 可选
        small_gap_low/high: 小幅缺口判定区间 (负收益绝对值)
        volume_drop_threshold: 成交量跌幅 (相对 20 日均量)

    Returns:
        可疑小幅未复权跳点 DataFrame.
    """
    need = {"code", "date", "close", "volume"}
    if not need.issubset(df.columns):
        return pd.DataFrame()

    d = df.sort_values(["code", "date"]).copy()
    d["date"] = pd.to_datetime(d["date"])
    d["prev_close"] = d.groupby("code")["close"].shift(1)
    d["pct_change"] = d["close"] / d["prev_close"] - 1
    d["vol_ma20"] = (
        d.groupby("code")["volume"]
        .transform(lambda s: s.rolling(20, min_periods=5).mean())
    )
    d["vol_ratio"] = d["volume"] / d["vol_ma20"]

    # 条件: 小幅下跳 + 成交未异常放量 (除权不等于利空, 常量能维持或萎缩)
    mask = (
        (d["pct_change"] <= -small_gap_low)
        & (d["pct_change"] >= -small_gap_high)
        & (d["vol_ratio"].fillna(1.0) < 1 + volume_drop_threshold)
    )

    # 剔除已在事件表中的日期
    if dividend_events is not None and not dividend_events.empty:
        ev = dividend_events.copy()
        ev["date"] = pd.to_datetime(ev["date"] if "date" in ev.columns else ev["ex_date"])
        known = set(zip(ev["code"].astype(str), ev["date"]))
        mask &= ~d.apply(
            lambda r: (str(r["code"]), r["date"]) in known, axis=1
        )

    suspects = d.loc[mask, ["code", "date", "close", "prev_close",
                            "pct_change", "vol_ratio"]].copy()
    return suspects.sort_values("pct_change")


def validate_factor_ratio_steps(
    adjusted_df: pd.DataFrame, raw_df: pd.DataFrame,
    expected_ratios: list[float] | None = None,
    tolerance: float = 0.002,
) -> dict:
    """验证累计复权比值的阶梯是否与已知送转比例匹配.

    分拆 / 送股的真实复权因子变化应是离散阶梯 (如 10送10 → ratio × 2,
    10送5 → ratio × 1.5), 且相邻阶梯比例应为简单分数.
    若出现 1.073 / 0.987 等非标准值 = 复权来源本身有误差,
    长期累计可产生 0.5-1% 的年化回测漂移.

    Args:
        adjusted_df / raw_df: 均含 ['code', 'date', 'close']
        expected_ratios: 允许的标准阶梯 (比如 [1.1, 1.2, 1.5, 2.0, 3.0]).
                         默认覆盖 A 股常见送转配股比例.
        tolerance: 匹配容差
    """
    if expected_ratios is None:
        expected_ratios = [
            1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.40, 1.50,
            1.60, 1.75, 2.00, 2.50, 3.00, 4.00, 5.00,
        ]

    merged = pd.merge(
        adjusted_df[["code", "date", "close"]].rename(columns={"close": "adj"}),
        raw_df[["code", "date", "close"]].rename(columns={"close": "raw"}),
        on=["code", "date"], how="inner",
    )
    if merged.empty:
        return {"error": "无重合样本"}

    merged["ratio"] = merged["adj"] / merged["raw"]
    merged = merged.sort_values(["code", "date"])
    merged["ratio_step"] = (
        merged.groupby("code")["ratio"].transform(lambda s: s / s.shift(1))
    )

    # 只看发生变化的步 (> 0.5%)
    steps = merged[(merged["ratio_step"] - 1).abs() > 0.005].copy()
    if steps.empty:
        return {"verdict": "PASS", "total_steps": 0, "note": "无除权事件"}

    def is_standard(r: float) -> bool:
        return any(abs(r - ref) < tolerance or abs(r - 1 / ref) < tolerance
                   for ref in expected_ratios)

    steps["is_standard"] = steps["ratio_step"].apply(is_standard)
    abnormal = steps[~steps["is_standard"]]

    return {
        "total_steps": len(steps),
        "abnormal_steps": len(abnormal),
        "abnormal_pct": float(len(abnormal) / max(len(steps), 1)),
        "samples": abnormal.head(10)[["code", "date", "ratio_step"]].to_dict("records"),
        "verdict": "PASS" if abnormal.empty else "WARN",
    }


def cumulative_adjustment_drift(
    adjusted_df: pd.DataFrame, raw_df: pd.DataFrame,
    dividend_events: pd.DataFrame | None = None,
) -> dict:
    """累计复权漂移检测: 从样本起点到终点,
    `理论累计复权比 = Π(1 + dividend_pct)` 应与 `adj/raw` 比值一致.

    这是多次分拆合并后的保险校验. 单步误差 0.1% × 10 次 ≈ 1% 累计漂移,
    够让 5 年回测的夏普从 1.5 变成 1.2.

    Args:
        adjusted_df / raw_df: 均含 ['code', 'date', 'close']
        dividend_events: 含 ['code', 'date', 'ratio'] (比如 10送10 → ratio=2.0)
                         若 None, 仅返回实际 ratio 终值供人工核对.
    """
    merged = pd.merge(
        adjusted_df[["code", "date", "close"]].rename(columns={"close": "adj"}),
        raw_df[["code", "date", "close"]].rename(columns={"close": "raw"}),
        on=["code", "date"], how="inner",
    )
    if merged.empty:
        return {"error": "无重合样本"}

    merged = merged.sort_values(["code", "date"])
    actual = (
        merged.groupby("code")
        .agg(start_ratio=("adj", lambda s: s.iloc[0] / merged.loc[s.index[0], "raw"]),
             end_ratio=("adj", lambda s: s.iloc[-1] / merged.loc[s.index[-1], "raw"]))
    )
    actual["actual_total"] = actual["end_ratio"] / actual["start_ratio"]

    if dividend_events is None or dividend_events.empty:
        return {
            "codes_checked": len(actual),
            "actual_total_ratios": actual["actual_total"].to_dict(),
            "verdict": "MANUAL",
            "note": "提供 dividend_events 可自动对照",
        }

    ev = dividend_events.copy()
    ev["date"] = pd.to_datetime(ev["date"])
    expected = ev.groupby("code")["ratio"].prod()

    cmp = pd.concat([actual["actual_total"], expected.rename("expected")], axis=1).dropna()
    cmp["drift_pct"] = (cmp["actual_total"] - cmp["expected"]) / cmp["expected"]
    bad = cmp[cmp["drift_pct"].abs() > 0.01]

    return {
        "codes_checked": len(cmp),
        "max_drift_pct": float(cmp["drift_pct"].abs().max()) if len(cmp) else 0.0,
        "drifted_codes": bad.head(10).to_dict("index"),
        "verdict": "PASS" if bad.empty else "FAIL",
    }


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
