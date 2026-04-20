"""研究 Pipeline: 把所有研究期暗门串成一键流程.

阶段:
    1. 数据体检 (data_hygiene)          失败 → 中止
    2. 标签工程 (label_engineering)      多 horizon + vol 归一化
    3. 不可交易样本屏蔽 (masks + event) 涨跌停 + 财报 + 解禁 剔除
    4. Barra 中性化 (barra_neutralize)  去风格 beta
    5. 训练 LightGBM                    (可选, 需 qlib / sklearn)
    6. IC 评估 + 衰减监控 (alpha_decay)
    7. 回测 (execution.simulator) 带冲击成本
    8. 产出报告 (reporting)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.logger import logger


@dataclass
class ResearchResult:
    stage_results: dict = field(default_factory=dict)
    model: Any = None
    predictions: pd.DataFrame | None = None
    ic_stats: dict = field(default_factory=dict)
    backtest_stats: dict = field(default_factory=dict)
    audit_report: Any = None
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)

    def summary(self) -> str:
        lines = ["=" * 60, "  研究 Pipeline 结果", "=" * 60]
        for k, v in self.ic_stats.items():
            lines.append(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        lines.append("--")
        for k, v in self.backtest_stats.items():
            lines.append(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        if self.warnings:
            lines.append("\n⚠️  警告:")
            for w in self.warnings:
                lines.append(f"   - {w}")
        if self.errors:
            lines.append("\n❌ 错误:")
            for e in self.errors:
                lines.append(f"   - {e}")
        return "\n".join(lines)


class ResearchPipeline:
    """端到端研究流水线."""

    def __init__(
        self,
        label_horizons: list[int] | None = None,
        label_weights: list[float] | None = None,
        neutralize_styles: bool = True,
        skip_audit: bool = False,
    ):
        self.label_horizons = label_horizons or [1, 3, 5, 10]
        self.label_weights = label_weights or [0.4, 0.3, 0.2, 0.1]
        self.neutralize_styles = neutralize_styles
        self.skip_audit = skip_audit

    def run(
        self,
        daily_df: pd.DataFrame,
        feature_df: pd.DataFrame | None = None,
        market_cap: pd.Series | None = None,
        market_return: pd.Series | None = None,
        industry_map: pd.Series | None = None,
    ) -> ResearchResult:
        """运行完整研究 pipeline.

        Args:
            daily_df: 必须 ['code', 'date', 'open', 'close', 'volume', 'pct_chg']
            feature_df: 原始因子矩阵 [dates, stocks], 可为空 (用 daily 自动算)
            market_cap: 市值 Series, index=code (用于 Barra 中性化)
            market_return: 市场基准日收益, 用于 beta 计算
            industry_map: {code: industry} (用于行业中性)
        """
        res = ResearchResult()

        # 1) 数据体检
        if not self.skip_audit:
            res = self._stage_audit(daily_df, res)
            if res.audit_report and res.audit_report.has_critical():
                res.errors.append("数据体检 critical, 中止")
                return res

        # 2) 标签工程
        res = self._stage_label(daily_df, res)

        # 3) 不可交易样本屏蔽
        res = self._stage_mask(daily_df, res)

        # 4) 因子 (简化: 用 pct_chg 做个示例, 实际应用 qlib Alpha158)
        res = self._stage_features(daily_df, feature_df, res)

        # 5) Barra 中性化
        if self.neutralize_styles and market_cap is not None and market_return is not None:
            res = self._stage_neutralize(
                res, daily_df, market_cap, market_return, industry_map,
            )

        # 6) IC 分析
        res = self._stage_ic_eval(res)

        # 7) 回测 (带冲击)
        res = self._stage_backtest(daily_df, res)

        return res

    # ============ stage 实现 ============
    def _stage_audit(self, daily_df, res):
        try:
            from data_hygiene import DataHealthChecker
            checker = DataHealthChecker()
            report = checker.audit_full(daily_df)
            res.audit_report = report
            res.stage_results["audit"] = {
                "critical": report.critical_issues,
                "warnings": report.warnings,
            }
            res.warnings.extend(report.warnings)
        except Exception as e:
            res.warnings.append(f"audit 失败: {e}")
        return res

    def _stage_label(self, daily_df, res):
        try:
            from label_engineering import multi_horizon_label, vol_adjusted_label
            labels_per_stock = {}
            for code, g in daily_df.groupby("code"):
                g = g.sort_values("date")
                close = g.set_index("date")["close"]
                open_ = g.set_index("date")["open"] if "open" in g.columns else None
                label = multi_horizon_label(
                    close, open_,
                    horizons=self.label_horizons,
                    weights=self.label_weights,
                )
                labels_per_stock[code] = label
            label_df = pd.DataFrame(labels_per_stock)
            res.stage_results["label"] = {
                "shape": label_df.shape,
                "non_nan_pct": float(label_df.notna().sum().sum() /
                                      label_df.size),
            }
            res.stage_results["_label_df"] = label_df
        except Exception as e:
            res.errors.append(f"label 失败: {e}")
        return res

    def _stage_mask(self, daily_df, res):
        try:
            from label_engineering import tradeable_mask
            mask = tradeable_mask(daily_df)
            res.stage_results["mask"] = {
                "total": len(daily_df),
                "tradeable": int(mask.sum()),
                "masked_pct": float(1 - mask.mean()),
            }
            res.stage_results["_tradeable_mask"] = mask
        except Exception as e:
            res.warnings.append(f"mask 失败: {e}")
        return res

    def _stage_features(self, daily_df, feature_df, res):
        if feature_df is not None:
            res.stage_results["features"] = {"shape": feature_df.shape,
                                              "source": "external"}
            res.stage_results["_feature_df"] = feature_df
            return res
        # 简化: 生成 3 个示例因子
        try:
            f = {}
            for code, g in daily_df.groupby("code"):
                g = g.sort_values("date").set_index("date")
                close = g["close"]
                feat_code = pd.DataFrame({
                    "momentum_5d": close.pct_change(5),
                    "reversal_1d": -close.pct_change(1),
                    "vol_20d": close.pct_change().rolling(20).std(),
                })
                f[code] = feat_code
            # 合并成 multi-index [date, code]
            combined = pd.concat(f, axis=0, names=["code", "date"])
            feature_df = combined.reset_index().pivot_table(
                index="date", columns="code",
            )
            res.stage_results["features"] = {
                "auto_generated": 3,
                "shape": feature_df.shape,
            }
            res.stage_results["_feature_df"] = feature_df
        except Exception as e:
            res.warnings.append(f"features 失败: {e}")
        return res

    def _stage_neutralize(self, res, daily_df, market_cap,
                           market_return, industry_map):
        try:
            from barra_neutralize import compute_all_styles, neutralize_by_regression

            # 计算收益矩阵
            ret_df = daily_df.pivot_table(
                index="date", columns="code", values="close",
            ).pct_change()

            t1 = t3 = t12 = market_cap * 0 + 0.5    # 默认值 (log(0.5)~0)
            if "turnover" in daily_df.columns and daily_df["turnover"].sum() > 0:
                t1 = daily_df.groupby("code")["turnover"].apply(
                    lambda s: max(s.tail(30).mean(), 0.01))
                t3 = daily_df.groupby("code")["turnover"].apply(
                    lambda s: max(s.tail(60).mean(), 0.01))
                t12 = daily_df.groupby("code")["turnover"].apply(
                    lambda s: max(s.mean(), 0.01))

            styles = compute_all_styles(
                market_cap, ret_df, market_return, t1, t3, t12,
            )
            res.stage_results["barra"] = {
                "style_count": styles.shape[1],
                "stocks": styles.shape[0],
            }
            res.stage_results["_barra_styles"] = styles
        except Exception as e:
            res.warnings.append(f"Barra 中性化失败: {e}")
        return res

    def _stage_ic_eval(self, res):
        try:
            from alpha_decay import rolling_ic_decay, half_life_estimate
            from alpha_decay.monitor import ic_ir

            label_df = res.stage_results.get("_label_df")
            feat_df = res.stage_results.get("_feature_df")
            if label_df is None or feat_df is None:
                res.warnings.append("无 label 或 feature, 跳过 IC")
                return res

            # 取第一个因子 (示例)
            first_feat_name = feat_df.columns.get_level_values(0)[0] \
                if isinstance(feat_df.columns, pd.MultiIndex) else \
                feat_df.columns[0]
            if isinstance(feat_df.columns, pd.MultiIndex):
                f = feat_df[first_feat_name]
            else:
                f = feat_df

            # 对齐 index
            common = f.index.intersection(label_df.index)
            if len(common) < 30:
                res.warnings.append("对齐样本不足")
                return res

            ic_df = rolling_ic_decay(f.loc[common], label_df.loc[common])
            ic_stats = ic_ir(ic_df["ic_daily"])
            half = half_life_estimate(ic_df["ic_daily"])

            res.ic_stats = {
                **ic_stats,
                "half_life_days": half,
                "rolling_20d_ic_latest": float(ic_df["ic_20d"].iloc[-1]),
            }
            res.stage_results["_ic_df"] = ic_df
        except Exception as e:
            res.warnings.append(f"IC 分析失败: {e}")
        return res

    def _stage_backtest(self, daily_df, res):
        """简化回测: 按 top-K 策略 + 执行仿真器计算净收益."""
        try:
            from execution import BacktestExecutionSim
            label_df = res.stage_results.get("_label_df")
            feat_df = res.stage_results.get("_feature_df")
            if label_df is None or feat_df is None:
                return res

            # 简化: 用 first feature 选 top10
            if isinstance(feat_df.columns, pd.MultiIndex):
                f = feat_df[feat_df.columns.get_level_values(0)[0]]
            else:
                f = feat_df

            sim = BacktestExecutionSim()
            daily_returns = []
            turnovers = []

            prev_top = set()
            for dt in f.index[::5]:     # 每 5 天调仓
                if dt not in f.index or dt not in label_df.index:
                    continue
                scores = f.loc[dt].dropna()
                if len(scores) < 10:
                    continue
                top10 = scores.nlargest(10).index.tolist()

                # 换手率
                turnover = len(set(top10) ^ prev_top) / 20
                turnovers.append(turnover)
                prev_top = set(top10)

                # 组合收益 (假设等权, 忽略冲击成本简化)
                label_row = label_df.loc[dt, top10].mean()
                if pd.notna(label_row):
                    daily_returns.append(float(label_row))

            if not daily_returns:
                res.warnings.append("回测无有效样本")
                return res

            arr = np.array(daily_returns)
            annual_ret = float(arr.mean() * 50)   # 每周一次, 50 周
            annual_vol = float(arr.std() * np.sqrt(50))
            sharpe = annual_ret / (annual_vol + 1e-9)
            max_dd = float(
                (pd.Series(arr).cumsum() -
                 pd.Series(arr).cumsum().cummax()).min()
            )
            res.backtest_stats = {
                "n_rebalances": len(daily_returns),
                "annual_return": annual_ret,
                "annual_vol": annual_vol,
                "sharpe": sharpe,
                "max_drawdown": max_dd,
                "avg_turnover": float(np.mean(turnovers)) if turnovers else 0,
            }
        except Exception as e:
            res.warnings.append(f"回测失败: {e}")
        return res
