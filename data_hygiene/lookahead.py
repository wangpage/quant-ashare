"""前视偏差 (Lookahead Bias) 扫描.

前视偏差 = feature 用到了"当时不可能知道"的信息.

常见触发方式:
    1. shift 方向错: df['ma5'] = df['close'].rolling(5).mean()
       但要求是 T 日用, 结果 T 日 MA 包含了 T 日 close. 看似合理, 实际要 shift(1)
    2. 财报用报告期而非披露日
    3. 次新股用上市后 N 天数据, 但其中某些天是退市风险股回档
    4. 股价用前复权, 但复权因子在除权日当天更新, 导致"提前反应"
    5. 数据源是事后重新发布的 (如 Wind 修正版), 实盘当时没有

头部机构扫描流程:
    a. 算每个 feature 与 label 的 Spearman 相关性
    b. 异常高 (> 0.3) 重点查
    c. 用 "shift(1) 后相关性" 对比: 相差 > 50% 必有问题
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def scan_lookahead_bias(
    features: pd.DataFrame, labels: pd.Series,
    threshold_corr: float = 0.25,
    shift_drop_threshold: float = 0.5,
) -> dict:
    """多维度扫描前视偏差.

    Args:
        features: 输入特征矩阵
        labels: forward return 标签
        threshold_corr: 相关性阈值, > 视为可疑
        shift_drop_threshold: shift(1) 后相关性下降比例,
                              > 此值 = 当前 feature 用了未来数据

    Returns:
        按可疑程度排序的可疑 feature 报告
    """
    suspicious = []
    clean_features = features.select_dtypes(include=[np.number])
    common_idx = clean_features.dropna().index.intersection(
        labels.dropna().index
    )
    if len(common_idx) < 30:
        return {"error": "有效样本不足 30"}

    f = clean_features.loc[common_idx]
    l = labels.loc[common_idx]

    for col in f.columns:
        corr_now = f[col].corr(l, method="spearman")
        corr_shifted = f[col].shift(1).corr(l, method="spearman")

        if pd.isna(corr_now) or pd.isna(corr_shifted):
            continue

        # 判定
        if abs(corr_now) < threshold_corr:
            continue

        if abs(corr_now) > 0:
            drop_ratio = 1 - abs(corr_shifted) / abs(corr_now)
        else:
            drop_ratio = 0

        severity = "LOW"
        reasons = []
        if abs(corr_now) > 0.5:
            severity = "CRITICAL"
            reasons.append(f"相关性 {corr_now:.3f} 过高, 几乎肯定泄露")
        elif abs(corr_now) > 0.35:
            severity = "HIGH"
            reasons.append(f"相关性 {corr_now:.3f} 可疑")
        elif drop_ratio > shift_drop_threshold:
            severity = "MEDIUM"
            reasons.append(f"shift 后相关性骤降 {drop_ratio:.1%}")
        else:
            severity = "LOW"

        suspicious.append({
            "feature": col,
            "corr_now": float(corr_now),
            "corr_shifted": float(corr_shifted),
            "drop_ratio": float(drop_ratio),
            "severity": severity,
            "reasons": reasons,
        })

    suspicious.sort(key=lambda x: -abs(x["corr_now"]))
    critical = [s for s in suspicious if s["severity"] == "CRITICAL"]
    high = [s for s in suspicious if s["severity"] == "HIGH"]

    return {
        "total_features": len(clean_features.columns),
        "suspicious_count": len(suspicious),
        "critical_count": len(critical),
        "high_count": len(high),
        "verdict": ("FAIL" if critical else
                    "WARN" if high else "PASS"),
        "suspicious_features": suspicious,
    }


def time_index_integrity_check(df: pd.DataFrame, date_col: str = "date") -> dict:
    """时间索引完整性检查.

    检测:
        1. 是否有重复日期 (同 code + 同 date 多条)
        2. 是否排序
        3. 是否有未来日期
    """
    if date_col not in df.columns:
        return {"error": f"缺 {date_col}"}
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    today = pd.Timestamp.today().normalize()

    issues = []
    if "code" in d.columns:
        dup = d.groupby(["code", date_col]).size()
        dup_count = (dup > 1).sum()
        if dup_count > 0:
            issues.append(f"有 {dup_count} 组 (code+date) 重复")
    else:
        dup_count = d[date_col].duplicated().sum()
        if dup_count > 0:
            issues.append(f"有 {dup_count} 个日期重复")

    # 按 code 分组检查排序 (面板数据)
    if "code" in d.columns:
        sort_ok = d.groupby("code")[date_col].apply(
            lambda s: s.is_monotonic_increasing
        ).all()
    else:
        sort_ok = d[date_col].is_monotonic_increasing
    if not sort_ok:
        issues.append("date 未排序 (建议 df.sort_values(['code', 'date']))")

    future = (d[date_col] > today).sum()
    if future > 0:
        issues.append(f"有 {future} 条日期在未来 (>{today.date()})")

    # A股节假日: 春节 9d / 国庆 7d / 五一 5d. 超过 15 天才算异常.
    unique_dates = d[date_col].drop_duplicates().sort_values()
    gaps = unique_dates.diff().dt.days
    big_gaps = (gaps > 15).sum()
    if big_gaps > 0:
        issues.append(f"有 {big_gaps} 个 >15 天的数据跳档 (排除春节/国庆)")

    return {
        "total_rows": len(d),
        "unique_dates": d[date_col].nunique(),
        "date_range": (
            str(d[date_col].min().date()),
            str(d[date_col].max().date()),
        ),
        "issues": issues,
        "verdict": "PASS" if not issues else "FAIL",
    }


def label_leakage_test(
    features: pd.DataFrame, label: pd.Series,
    model_cls=None,
) -> dict:
    """严格的泄露测试: 用 shuffle label 训练模型.

    原理:
        如果模型在 "标签完全打乱" 的数据上仍然 out-of-sample 有 IC,
        说明 features 已经泄露了 label 信息.

    Returns:
        {'shuffled_ic': 打乱标签后的 IC, 应接近 0}
    """
    try:
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import train_test_split
    except ImportError:
        return {"error": "缺 sklearn"}

    common = features.dropna().index.intersection(label.dropna().index)
    if len(common) < 100:
        return {"error": "样本不足"}

    X = features.loc[common].values
    y = label.loc[common].values

    # 打乱 label
    rng = np.random.default_rng(42)
    y_shuffled = y.copy()
    rng.shuffle(y_shuffled)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_shuffled, test_size=0.3, random_state=42,
    )
    m = (model_cls or Ridge)()
    m.fit(X_tr, y_tr)
    pred = m.predict(X_te)
    ic = np.corrcoef(pred, y_te)[0, 1]

    return {
        "shuffled_label_ic": float(ic),
        "pass_threshold": 0.05,
        "verdict": "PASS" if abs(ic) < 0.05 else "FAIL (feature 可能泄露)",
    }
