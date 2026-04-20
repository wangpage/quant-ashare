"""一键数据体检报告.

用法:
    checker = DataHealthChecker()
    report = checker.audit_full(df)
    print(report.summary())
    report.fail_checklist()  # 抛异常如有 critical 问题
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .survivorship import (
    detect_survivorship_bias, detect_point_in_time_issues,
)
from .lookahead import (
    scan_lookahead_bias, time_index_integrity_check,
)
from .adjustment import (
    detect_price_jumps, verify_adjustment_factor,
)
from .gaps import find_suspension_days


@dataclass
class AuditReport:
    checks: dict = field(default_factory=dict)
    critical_issues: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    passed: list = field(default_factory=list)

    def summary(self) -> str:
        lines = ["=" * 60]
        lines.append("  数据体检报告")
        lines.append("=" * 60)
        lines.append(f"CRITICAL: {len(self.critical_issues)}")
        lines.append(f"WARNING:  {len(self.warnings)}")
        lines.append(f"PASSED:   {len(self.passed)}")
        lines.append("")
        for i in self.critical_issues:
            lines.append(f"❌ {i}")
        for w in self.warnings:
            lines.append(f"⚠️  {w}")
        for p in self.passed:
            lines.append(f"✅ {p}")
        return "\n".join(lines)

    def has_critical(self) -> bool:
        return len(self.critical_issues) > 0

    def fail_checklist(self):
        """如有 critical 问题抛异常 (CI/CD 用)."""
        if self.has_critical():
            raise RuntimeError(
                "数据体检失败, 问题:\n" + "\n".join(self.critical_issues)
            )


class DataHealthChecker:
    """数据健康度一键检查器."""

    def audit_full(
        self,
        df: pd.DataFrame,
        features: pd.DataFrame | None = None,
        label: pd.Series | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> AuditReport:
        """跑所有暗门检查, 汇总报告.

        Args:
            df: 主数据表, 应含 [code, date, close, volume, ...]
            features: 可选, 特征矩阵 (前视偏差扫描)
            label: 可选, 标签 (前视偏差扫描)
            start_date, end_date: 可选, 时段
        """
        report = AuditReport()

        # 1. 时间索引完整性
        t_check = time_index_integrity_check(df)
        report.checks["time_integrity"] = t_check
        if t_check.get("verdict") == "FAIL":
            report.critical_issues.append(
                f"时间索引问题: {'; '.join(t_check.get('issues', []))}"
            )
        else:
            report.passed.append("时间索引完整")

        # 2. 幸存偏差
        if "date" in df.columns:
            if not start_date:
                start_date = str(pd.to_datetime(df["date"]).min().date())
            if not end_date:
                end_date = str(pd.to_datetime(df["date"]).max().date())
            sb = detect_survivorship_bias(df, start_date, end_date)
            report.checks["survivorship"] = sb
            if sb.get("survivorship_risk") == "HIGH":
                report.critical_issues.append(
                    f"幸存偏差: 数据缺少退市股, 预计每年虚高 "
                    f"{sb.get('estimated_bias_pct_per_year', 0)}%"
                )
            elif sb.get("survivorship_risk") == "MEDIUM":
                report.warnings.append("疑似幸存偏差")
            else:
                report.passed.append("幸存偏差: LOW")

        # 3. 价格跳点
        if {"code", "close"}.issubset(df.columns):
            jumps = detect_price_jumps(df)
            report.checks["price_jumps"] = {
                "count": len(jumps),
                "samples": jumps.head(5).to_dict("records") if not jumps.empty else [],
            }
            if len(jumps) > 20:
                report.warnings.append(
                    f"检测到 {len(jumps)} 个价格跳点, 可能未做除权"
                )
            else:
                report.passed.append(f"价格跳点 OK ({len(jumps)} 个, 可接受)")

        # 4. 复权因子验证
        adj_check = verify_adjustment_factor(df)
        report.checks["adjustment"] = adj_check
        if not adj_check.get("has_factor_col"):
            report.warnings.append("缺 factor 列, 无法校验复权")
        elif adj_check.get("verdict") == "FAIL":
            report.critical_issues.append("复权因子非单调, 处理有 bug")
        else:
            report.passed.append("复权因子一致")

        # 5. 停牌识别
        if {"code", "volume"}.issubset(df.columns):
            susp = find_suspension_days(df)
            report.checks["suspensions"] = {
                "total_periods": len(susp),
                "total_days": int(susp["days"].sum()) if not susp.empty else 0,
            }
            if len(susp) > 0:
                report.passed.append(f"停牌统计: {len(susp)} 区间")

        # 6. 前视偏差
        if features is not None and label is not None:
            la = scan_lookahead_bias(features, label)
            report.checks["lookahead"] = la
            v = la.get("verdict")
            if v == "FAIL":
                report.critical_issues.append(
                    f"前视偏差: {la.get('critical_count')} 个 feature 严重泄露"
                )
            elif v == "WARN":
                report.warnings.append(
                    f"前视偏差: {la.get('high_count')} 个 feature 可疑"
                )
            elif v == "PASS":
                report.passed.append("前视偏差检查通过")

        return report
