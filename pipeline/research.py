"""研究 Pipeline: 把所有研究期暗门串成一键流程.

阶段:
    1. 数据体检 (data_hygiene)           失败 → 中止
    2. 标签工程 (label_engineering)       多 horizon + vol 归一化
    3. 不可交易样本屏蔽 (masks + event)   涨跌停 + 财报披露日 + 解禁
    4. 特征生成 / 外部输入
    5. 前视偏差扫描 (lookahead)            CRITICAL → 中止; HIGH → warning
    6. Barra 中性化 (分层 + 岭回归)
    7. IC 评估 + 衰减监控 (alpha_decay)
    8. 回测 (execution.simulator + PreTradeGate 强制过闸)
    9. 产出报告 (reporting)

新增的强制环节:
    - 第 5 步: 默认调 scan_lookahead_bias; 若 CRITICAL 且 fail_on_critical
              为 True (默认), 中止 pipeline
    - 第 8 步: 每次调仓过 PreTradeGate, 拒单不进组合; gate.stats 回传
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
    lookahead_report: dict = field(default_factory=dict)
    neutralize_diagnostics: dict = field(default_factory=dict)
    gate_stats: dict = field(default_factory=dict)
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
        neutralize_method: str = "hierarchical",   # "hierarchical" | "single"
        skip_audit: bool = False,
        lookahead_scan: bool = True,
        fail_on_lookahead_critical: bool = True,
        enforce_risk_gate: bool = True,
    ):
        self.label_horizons = label_horizons or [1, 3, 5, 10]
        self.label_weights = label_weights or [0.4, 0.3, 0.2, 0.1]
        self.neutralize_styles = neutralize_styles
        self.neutralize_method = neutralize_method
        self.skip_audit = skip_audit
        self.lookahead_scan = lookahead_scan
        self.fail_on_lookahead_critical = fail_on_lookahead_critical
        self.enforce_risk_gate = enforce_risk_gate

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

        # 4) 因子
        res = self._stage_features(daily_df, feature_df, res)

        # 5) 前视偏差扫描 (强制, 可关闭)
        if self.lookahead_scan:
            res = self._stage_lookahead_scan(res)
            if (self.fail_on_lookahead_critical
                    and res.lookahead_report.get("verdict") == "FAIL"):
                res.errors.append(
                    "检测到 CRITICAL 前视偏差, 中止 pipeline. "
                    "详见 lookahead_report.suspicious_features"
                )
                return res

        # 6) Barra 中性化
        if (self.neutralize_styles and market_cap is not None
                and market_return is not None):
            res = self._stage_neutralize(
                res, daily_df, market_cap, market_return, industry_map,
            )

        # 7) IC 分析
        res = self._stage_ic_eval(res)

        # 8) 回测 (带冲击 + 风控过闸)
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

    def _stage_lookahead_scan(self, res):
        """前视偏差扫描: 横截面 feature vs 未来 return 的相关性."""
        try:
            from data_hygiene.lookahead import scan_lookahead_bias
            feat_df = res.stage_results.get("_feature_df")
            label_df = res.stage_results.get("_label_df")
            if feat_df is None or label_df is None:
                res.warnings.append("无 feature/label, 跳过 lookahead 扫描")
                return res
            # 取第一只票 / 第一因子简化扫描
            if isinstance(feat_df.columns, pd.MultiIndex):
                feat0 = feat_df.xs(feat_df.columns.get_level_values(0)[0],
                                    axis=1, level=0)
            else:
                feat0 = feat_df
            # 对齐成单一 Series (打平)
            common = feat0.index.intersection(label_df.index)
            if len(common) < 30:
                res.warnings.append("lookahead 扫描: 样本不足")
                return res
            label_flat = label_df.loc[common].stack()
            feat_flat = feat0.loc[common].stack().reindex(label_flat.index)
            # scan 要求 DataFrame
            scan_df = pd.DataFrame({"feat0": feat_flat}).dropna()
            label_aligned = label_flat.reindex(scan_df.index)
            report = scan_lookahead_bias(scan_df, label_aligned)
            res.lookahead_report = report
            if report.get("verdict") == "FAIL":
                logger.error(f"前视偏差扫描 FAIL: {report.get('critical_count')} 个 CRITICAL")
            elif report.get("verdict") == "WARN":
                res.warnings.append(
                    f"lookahead: {report.get('high_count')} 个 HIGH 可疑特征"
                )
        except Exception as e:
            res.warnings.append(f"lookahead 扫描失败: {e}")
        return res

    def _stage_neutralize(self, res, daily_df, market_cap,
                           market_return, industry_map):
        try:
            from barra_neutralize import (
                compute_all_styles, neutralize_by_regression,
                neutralize_hierarchical,
            )

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
                "method": self.neutralize_method,
            }
            res.stage_results["_barra_styles"] = styles

            # 若有 feature_df 的"最新一期"因子, 应用中性化并记录诊断
            feat_df = res.stage_results.get("_feature_df")
            if feat_df is not None and not feat_df.empty:
                latest_date = feat_df.index.max()
                try:
                    if isinstance(feat_df.columns, pd.MultiIndex):
                        alpha_raw = (feat_df.xs(
                            feat_df.columns.get_level_values(0)[0],
                            axis=1, level=0,
                        ).loc[latest_date])
                    else:
                        alpha_raw = feat_df.loc[latest_date]
                    alpha_raw = alpha_raw.dropna()
                    if self.neutralize_method == "hierarchical":
                        _, diag = neutralize_hierarchical(
                            alpha_raw, styles.reindex(alpha_raw.index),
                            industries=industry_map, weights=market_cap.pow(0.5),
                            return_diagnostics=True,
                        )
                    else:
                        _, diag = neutralize_by_regression(
                            alpha_raw, styles.reindex(alpha_raw.index),
                            industries=industry_map, weights=market_cap.pow(0.5),
                            return_diagnostics=True,
                        )
                    res.neutralize_diagnostics = {
                        "n_samples": diag.n_samples,
                        "n_style_factors": diag.n_style_factors,
                        "n_industry_factors": diag.n_industry_factors,
                        "condition_number": diag.condition_number,
                        "r_squared": diag.r_squared,
                        "ridge_alpha": diag.ridge_alpha,
                        "warnings": diag.warnings,
                    }
                    if diag.warnings:
                        res.warnings.extend(diag.warnings)
                except Exception as e:
                    res.warnings.append(f"中性化诊断失败: {e}")
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
        """简化回测: 按 top-K 策略 + 执行仿真器 + PreTradeGate 强制过闸."""
        try:
            from execution import BacktestExecutionSim
            from risk import (
                PreTradeGate, OrderIntent, Portfolio, build_default_gate,
            )

            label_df = res.stage_results.get("_label_df")
            feat_df = res.stage_results.get("_feature_df")
            if label_df is None or feat_df is None:
                return res

            if isinstance(feat_df.columns, pd.MultiIndex):
                f = feat_df[feat_df.columns.get_level_values(0)[0]]
            else:
                f = feat_df

            sim = BacktestExecutionSim()
            gate = build_default_gate() if self.enforce_risk_gate else None
            # gate_stats 必须在任一退出路径都被记录, 方便上游审计
            if gate is not None:
                res.gate_stats = gate.stats.to_dict()
            # 简化 portfolio: 回测中的上下文, 仅用来让 gate 通过基础校验
            portfolio = Portfolio(
                cash=1_000_000.0, initial_capital=1_000_000.0,
                high_water_mark=1_000_000.0, daily_start_value=1_000_000.0,
            )

            # 预取 pct_chg 便于过闸涨跌停判定
            daily_by_code = {
                c: g.sort_values("date").set_index("date")
                for c, g in daily_df.groupby("code")
            } if "code" in daily_df.columns else {}

            daily_returns = []
            turnovers = []
            n_hard_rejects = 0

            prev_top: set = set()
            # 小样本股票池也要能跑通 (如测试场景 3 只); 阈值取 min(10, universe)
            universe_size = f.shape[1] if hasattr(f, "shape") else 0
            min_scores = max(1, min(10, universe_size))
            top_n = max(1, min(10, universe_size))

            for dt in f.index[::5]:
                if dt not in f.index or dt not in label_df.index:
                    continue
                scores = f.loc[dt].dropna()
                if len(scores) < min_scores:
                    continue
                top_candidates = scores.nlargest(min(20, universe_size)).index.tolist()

                # 过闸: 逐个候选跑 PreTradeGate, 拒单不进组合
                if gate is not None:
                    admitted: list = []
                    for code in top_candidates:
                        row = daily_by_code.get(code)
                        if row is None or dt not in row.index:
                            admitted.append(code)  # 缺数据默认放行
                            continue
                        snap = row.loc[dt]
                        price = float(snap.get("close", 0))
                        # prev_close 估算
                        if "pct_chg" in snap and price:
                            pct = float(snap.get("pct_chg", 0)) / 100
                            prev_close = price / (1 + pct) if (1 + pct) else price
                        else:
                            prev_close = price
                        intent = OrderIntent(
                            code=str(code), side="buy", shares=100,
                            price=price, prev_close=prev_close,
                        )
                        decision = gate.check(intent, portfolio, today=dt.date() if hasattr(dt, "date") else None)
                        if decision.allow:
                            admitted.append(code)
                        else:
                            n_hard_rejects += 1
                        if len(admitted) >= top_n:
                            break
                    top = admitted
                else:
                    top = top_candidates[:top_n]

                if len(top) == 0:
                    continue

                turnover = len(set(top) ^ prev_top) / max(len(top) + len(prev_top), 1)
                turnovers.append(turnover)
                prev_top = set(top)

                label_row = label_df.loc[dt, top].mean()
                if pd.notna(label_row):
                    daily_returns.append(float(label_row))

            if not daily_returns:
                res.warnings.append("回测无有效样本")
                if gate is not None:
                    res.gate_stats = gate.stats.to_dict()
                return res

            arr = np.array(daily_returns)
            annual_ret = float(arr.mean() * 50)
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
                "gate_hard_rejects": int(n_hard_rejects),
            }
            if gate is not None:
                res.gate_stats = gate.stats.to_dict()
        except Exception as e:
            res.warnings.append(f"回测失败: {e}")
        return res
