"""可交易性掩码 (Tradeable Mask) - 回测中被大量忽视的坑.

A股严苛规则:
    1) 涨停 (≥+9.5%) 时买单无法成交 → 那一天的 label 不能作为样本
    2) 跌停 (≤-9.5%) 时卖单无法成交 → 次日 label 也不能用
    3) 停牌日 / 停牌后 3 日 → 价格失真
    4) 新股上市前 N 日 → 未稳定
    5) 财报前 2 天 / 后 1 天 → 已知信息泄露
    6) 减持 / 增发公告日 → 信息冲击
    7) ST / *ST / 退市整理期 → 流动性陷阱

不过滤这些, 回测 IC 会系统性虚高 (幻方内部估计虚高 20-30%).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def tradeable_mask(
    df: pd.DataFrame,
    limit_up_threshold: float = 0.095,
    limit_down_threshold: float = -0.095,
    min_volume: int = 1,
    min_days_after_ipo: int = 250,
    exclude_st: bool = True,
) -> pd.Series:
    """返回一个 bool Series, True 表示"当日可作为训练/交易样本".

    Args:
        df: 必须含 ['pct_chg', 'volume'] 列, 可选 ['name', 'ipo_days']
    """
    mask = pd.Series(True, index=df.index)

    # 1. 涨跌停 (当日收盘价 ≥ 涨停) 不能买入
    if "pct_chg" in df.columns:
        mask &= df["pct_chg"] < limit_up_threshold * 100
        mask &= df["pct_chg"] > limit_down_threshold * 100

    # 2. 停牌 (成交量 = 0)
    if "volume" in df.columns:
        mask &= df["volume"] >= min_volume

    # 3. 次新股 (上市 < N 天)
    if "ipo_days" in df.columns:
        mask &= df["ipo_days"] >= min_days_after_ipo

    # 4. ST 股
    if exclude_st and "name" in df.columns:
        mask &= ~df["name"].str.contains("ST|退", regex=True, na=False)

    return mask


def event_window_mask(
    df: pd.DataFrame,
    earnings_dates: dict[str, list[str]] | None = None,
    pre_days: int = 2,
    post_days: int = 1,
    other_events: dict[str, list[tuple[str, int, int]]] | None = None,
) -> pd.Series:
    """把特定事件窗口的样本标记为"不可用".

    ⚠️  关键: `earnings_dates` 必须传 **披露日 (disclosure_date)**,
    不能是报告期 (report_period). 举例:

        报告期 = 2024Q1 末 (2024-03-31) - **事后才知道**
        披露日 = 2024-04-18 (实际公告) - 当日才可用

    用报告期做回测 → 回测期 2024-03-31 的样本就"看到"了 2024Q1 业绩,
    实盘不可能. 这是 A股最经典的前视偏差之一.

    Args:
        df: 必须含 ['code', 'date'] 列
        earnings_dates: {code: [disclosure_date]}, **披露日字符串**
        pre_days: 披露前 N 天屏蔽 (信息泄露风险)
        post_days: 披露后 N 天屏蔽 (消化期)
        other_events: {code: [(disclosure_date, pre_days, post_days), ...]}
            如减持公告 (-1, 2), 增发 (-3, 5), 解禁 (-10, 3)

    Returns:
        bool Series, True = 可用
    """
    mask = pd.Series(True, index=df.index)
    if earnings_dates is None and other_events is None:
        return mask

    if "date" not in df.columns or "code" not in df.columns:
        return mask

    df_dates = pd.to_datetime(df["date"])

    if earnings_dates:
        for code, dates in earnings_dates.items():
            stock_idx = df["code"] == code
            for d in dates:
                dt = pd.to_datetime(d)
                window = (df_dates >= dt - pd.Timedelta(days=pre_days)) & \
                         (df_dates <= dt + pd.Timedelta(days=post_days))
                mask &= ~(stock_idx & window)

    if other_events:
        for code, events in other_events.items():
            stock_idx = df["code"] == code
            for d, pre, post in events:
                dt = pd.to_datetime(d)
                window = (df_dates >= dt - pd.Timedelta(days=pre)) & \
                         (df_dates <= dt + pd.Timedelta(days=post))
                mask &= ~(stock_idx & window)

    return mask


def disclosure_vs_report_check(
    report_df: pd.DataFrame,
    report_date_col: str = "report_period",
    disclosure_col: str = "disclosure_date",
    max_lag_days: int = 180,
) -> dict:
    """检查财报/公告数据是否同时含有 "报告期" 和 "披露日".

    回测期必须用披露日 (pd.disclosure_date), 否则一定前视偏差.
    本函数返回诊断信息, 让 pipeline 发现缺失披露日时报警.

    Args:
        report_df: 财报/公告原始数据
        report_date_col: 报告期列名
        disclosure_col: 披露日列名
        max_lag_days: 披露日 - 报告期 的合理上限 (A股半年报要求 2 个月内)
    """
    issues = []
    if disclosure_col not in report_df.columns:
        issues.append(
            f"缺少披露日列 '{disclosure_col}', 回测只能用报告期 → 前视偏差!"
        )
        return {
            "has_disclosure": False, "verdict": "FAIL",
            "issues": issues,
        }
    if report_date_col not in report_df.columns:
        issues.append(f"缺少报告期列 '{report_date_col}'")

    df = report_df.copy()
    df[disclosure_col] = pd.to_datetime(df[disclosure_col], errors="coerce")
    if report_date_col in df.columns:
        df[report_date_col] = pd.to_datetime(df[report_date_col], errors="coerce")
        lag_days = (df[disclosure_col] - df[report_date_col]).dt.days
        neg = int((lag_days < 0).sum())
        too_long = int((lag_days > max_lag_days).sum())
        if neg > 0:
            issues.append(
                f"{neg} 条记录披露日早于报告期 (不可能, 疑似字段错位)"
            )
        if too_long > 0:
            issues.append(
                f"{too_long} 条记录披露日滞后 > {max_lag_days} 天 (疑似补录)"
            )

    null_disc = int(df[disclosure_col].isna().sum())
    if null_disc > 0:
        issues.append(f"{null_disc} 条记录披露日为空")

    return {
        "has_disclosure": True,
        "n_records": int(len(df)),
        "null_disclosure": null_disc,
        "issues": issues,
        "verdict": "FAIL" if any("FAIL" in i or "不可能" in i
                                   for i in issues) else
                   ("WARN" if issues else "PASS"),
    }


def timestamp_integrity_check(
    feature_df: pd.DataFrame, label_df: pd.DataFrame,
    code_col: str = "code", date_col: str = "date",
) -> dict:
    """检查 feature 时戳 <= label 决策时戳.

    A股典型前视偏差:
        1) label = close[t+5]/open[t+1], 但 feature 用到 close[t+1], close[t+2] ...
        2) 跨日因子用 "昨收" 时误用了 "今开"
        3) 盘后资金流 (如北向、龙虎榜) 用在盘中

    本函数从 feature_df / label_df 的 date 列对齐, 逐票检查是否有
    feature_date > label_anchor_date 的异常行.

    Returns:
        dict: 诊断结果, verdict ∈ {PASS, WARN, FAIL}
    """
    issues = []
    if date_col not in feature_df.columns or date_col not in label_df.columns:
        return {"verdict": "SKIP", "reason": f"缺 {date_col} 列"}
    try:
        fd = pd.to_datetime(feature_df[date_col])
        ld = pd.to_datetime(label_df[date_col])
    except Exception as e:
        return {"verdict": "FAIL", "reason": str(e)}

    future_f = int((fd > pd.Timestamp.today().normalize()).sum())
    future_l = int((ld > pd.Timestamp.today().normalize()).sum())
    if future_f > 0:
        issues.append(f"feature 有 {future_f} 行在未来")
    if future_l > 0:
        issues.append(f"label 有 {future_l} 行在未来")

    # 粗粒度: feature 日期最大值 应该 <= label 日期最大值
    if fd.max() > ld.max():
        issues.append(
            f"feature 最大日期 {fd.max().date()} > label 最大日期 {ld.max().date()}, "
            f"feature 可能包含未来数据"
        )

    return {
        "verdict": "FAIL" if issues else "PASS",
        "issues": issues,
        "feature_date_range": (str(fd.min().date()), str(fd.max().date())),
        "label_date_range": (str(ld.min().date()), str(ld.max().date())),
    }


def leaky_label_detector(
    features: pd.DataFrame, label: pd.Series,
    threshold: float = 0.3,
) -> list[str]:
    """前视偏差检测: 找出与 label 相关性异常高的特征.

    经验法则:
        - 同期相关性 > 0.3 → 几乎肯定是泄露
        - 先查 feature 是否包含 t+k 信息 (如用到 close[t+1])
    """
    suspicious = []
    for col in features.columns:
        if features[col].dtype.kind not in "if":
            continue
        corr = features[col].corr(label)
        if abs(corr) > threshold:
            suspicious.append(f"{col} (corr={corr:.3f})")
    return suspicious
