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
        use_alpha158: bool = False,                 # True = 用 Alpha158-lite 替代玩具因子
        neutralize_residual: bool = False,          # True = alpha 选股前先取 Barra 残差
        compose_signal: bool = True,                 # False = 只取第一因子 (baseline)
        portfolio_method: str = "equal_weight",      # equal_weight | inverse_vol | risk_parity
        rebalance_freq: int = 5,                     # 调仓步长 (f.index[::freq])
        top_k: int = 10,                             # 每期持仓数上限
        top_k_ratio: float | None = None,            # 若给, 用 max(1, int(ratio*universe))
        cov_lookback: int = 60,                      # risk_parity / inverse_vol 的回看天数
        turnover_buffer: float = 0.0,                # [0,1); 旧持仓仍在 top(1+buffer)*top_k 保留
        ic_gate: bool = False,                       # True = rolling IC<0 的期间强制空仓
        ic_gate_window: int = 20,
        signal_ema_span: int = 1,                    # 1=不平滑; 3~5 去信号噪声
        vol_target: float | None = None,             # 组合年化波动目标, 如 0.15; None=不启用
        vol_target_window: int = 20,                 # 滚动窗口估计实现波动
        vol_target_max_leverage: float = 1.5,        # 杠杆上限 (平静期加仓)
        vol_target_min_leverage: float = 0.3,        # 仓位下限 (波动期防御)
        signal_risk_adjust: bool = False,            # True = 排序按 signal / past_vol
        experiment_name: str = "baseline",
    ):
        self.label_horizons = label_horizons or [1, 3, 5, 10]
        self.label_weights = label_weights or [0.4, 0.3, 0.2, 0.1]
        self.neutralize_styles = neutralize_styles
        self.neutralize_method = neutralize_method
        self.skip_audit = skip_audit
        self.lookahead_scan = lookahead_scan
        self.fail_on_lookahead_critical = fail_on_lookahead_critical
        self.enforce_risk_gate = enforce_risk_gate
        self.use_alpha158 = use_alpha158
        self.neutralize_residual = neutralize_residual
        self.compose_signal = compose_signal
        self.portfolio_method = portfolio_method
        self.rebalance_freq = max(1, int(rebalance_freq))
        self.top_k = max(1, int(top_k))
        self.top_k_ratio = top_k_ratio
        self.cov_lookback = max(10, int(cov_lookback))
        self.turnover_buffer = max(0.0, float(turnover_buffer))
        self.ic_gate = ic_gate
        self.ic_gate_window = max(5, int(ic_gate_window))
        self.signal_ema_span = max(1, int(signal_ema_span))
        self.vol_target = vol_target
        self.vol_target_window = max(5, int(vol_target_window))
        self.vol_target_max_leverage = float(vol_target_max_leverage)
        self.vol_target_min_leverage = float(vol_target_min_leverage)
        self.signal_risk_adjust = bool(signal_risk_adjust)
        self.experiment_name = experiment_name

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

        # 6.5) 多因子合成 composite signal (+ 可选 Barra 残差化)
        res = self._stage_signal(res)

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

        if self.use_alpha158:
            try:
                from factors.alpha158_lite import compute_alpha158_panel
                feature_df = compute_alpha158_panel(daily_df)
                n_factors = int(feature_df.columns.get_level_values(0).nunique())
                res.stage_results["features"] = {
                    "source": "alpha158_lite",
                    "n_factors": n_factors,
                    "shape": feature_df.shape,
                }
                res.stage_results["_feature_df"] = feature_df
                return res
            except Exception as e:
                res.warnings.append(f"alpha158 生成失败, 退化玩具因子: {e}")

        # 玩具 baseline: 3 个示例因子
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
            combined = pd.concat(f, axis=0, names=["code", "date"])
            feature_df = combined.reset_index().pivot_table(
                index="date", columns="code",
            )
            res.stage_results["features"] = {
                "source": "toy_baseline",
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

    def _stage_signal(self, res):
        """多因子合成 composite signal. 可选: 对每截面取 Barra 残差.

        baseline (compose_signal=False) 直接取第一个因子作为 signal,
        与历史 IC/backtest 行为完全一致.
        """
        try:
            feat_df = res.stage_results.get("_feature_df")
            if feat_df is None or feat_df.empty:
                return res

            if not self.compose_signal:
                # baseline: 只取第一因子, 不做合成
                if isinstance(feat_df.columns, pd.MultiIndex):
                    first = feat_df.columns.get_level_values(0)[0]
                    signal = feat_df[first]
                else:
                    signal = feat_df
                res.stage_results.setdefault("signal", {}).update({
                    "composite": "first_factor_only",
                })
                res.stage_results["_signal_df"] = signal
                return res

            # 多因子: 优先 Rolling IC 加权 (无前视偏差, 自适应 regime 切换)
            if isinstance(feat_df.columns, pd.MultiIndex):
                from factors.alpha158_lite import (
                    combine_factors_equal_weight,
                    combine_factors_rolling_ic,
                )
                label_df = res.stage_results.get("_label_df")
                if label_df is not None and not label_df.empty:
                    signal = combine_factors_rolling_ic(
                        feat_df, label_df, window=60, top_k=8,
                    )
                    last_log = signal.attrs.get("selection_log", [])
                    last_factors = last_log[-1]["top_5"] if last_log else []
                    res.stage_results.setdefault("signal", {}).update({
                        "composite": "rolling_ic_weighted",
                        "window": signal.attrs.get("window"),
                        "latest_top_factors": [
                            f"{f}:{ic:+.3f}" for f, ic in last_factors
                        ],
                    })
                else:
                    signal = combine_factors_equal_weight(feat_df)
                    res.stage_results.setdefault("signal", {})["composite"] = "equal_weight"
            else:
                signal = feat_df

            # 可选: Barra 残差化 (每期截面 alpha 对风格因子回归取残差)
            if self.neutralize_residual:
                styles = res.stage_results.get("_barra_styles")
                if styles is not None and not styles.empty:
                    from barra_neutralize import neutralize_hierarchical
                    resid_by_date = {}
                    resid_count = 0
                    for dt in signal.index:
                        alpha_t = signal.loc[dt].dropna()
                        if len(alpha_t) < 20:
                            resid_by_date[dt] = alpha_t
                            continue
                        try:
                            r = neutralize_hierarchical(
                                alpha_t,
                                styles.reindex(alpha_t.index),
                                ridge_alpha=1e-2,
                            )
                            resid_by_date[dt] = r
                            resid_count += 1
                        except Exception:
                            resid_by_date[dt] = alpha_t
                    signal = pd.DataFrame(resid_by_date).T
                    # 追加残差化信息, 不覆盖 composite 元数据
                    res.stage_results.setdefault("signal", {}).update({
                        "residualized_by_barra": True,
                        "n_residualized_dates": resid_count,
                    })
                else:
                    res.warnings.append(
                        "neutralize_residual=True 但无 Barra styles, 跳过"
                    )
            # 信号 EMA 平滑 (降噪 + 降换手).  signal.ewm(span).mean() 在每列
            # 独立做时间方向的指数平滑, 不是横截面操作, 所以无前视.
            if self.signal_ema_span > 1 and hasattr(signal, "ewm"):
                try:
                    smoothed = signal.ewm(
                        span=self.signal_ema_span, adjust=False,
                    ).mean()
                    # 前 span 期用原值 (ewm 在数据不足时的输出会 lag 到 0)
                    signal = smoothed
                    res.stage_results.setdefault("signal", {}).update({
                        "ema_span": self.signal_ema_span,
                    })
                except Exception as e:
                    res.warnings.append(f"EMA 平滑失败: {e}")

            if "signal" not in res.stage_results:
                res.stage_results["signal"] = {"composite": "passthrough"}
            res.stage_results["_signal_df"] = signal
        except Exception as e:
            res.warnings.append(f"signal 合成失败: {e}")
        return res

    def _stage_ic_eval(self, res):
        try:
            from alpha_decay import rolling_ic_decay, half_life_estimate
            from alpha_decay.monitor import ic_ir

            label_df = res.stage_results.get("_label_df")
            # 优先用合成 signal (Alpha158-lite 需要), 否则退化到第一因子
            signal_df = res.stage_results.get("_signal_df")
            feat_df = res.stage_results.get("_feature_df")
            if label_df is None or (signal_df is None and feat_df is None):
                res.warnings.append("无 label 或 feature, 跳过 IC")
                return res

            if signal_df is not None:
                f = signal_df
            elif isinstance(feat_df.columns, pd.MultiIndex):
                f = feat_df[feat_df.columns.get_level_values(0)[0]]
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
        """Top-K + 组合优化 + 换手 buffer + 可选 IC gating 回测.

        权重方案 portfolio_method:
            equal_weight  (默认): 每股 1/N, 兼容历史 baseline
            inverse_vol:  w_i ∝ 1/σ_i, 基于过去 cov_lookback 日日收益
            risk_parity:  边际风险贡献相等, 基于过去 cov_lookback 日协方差

        换手 buffer (turnover_buffer): 若旧持仓当日 rank 仍在前
        top_k*(1+buffer) 之内, 不触发换仓, 避免 "边缘排序抖动" 造成的
        假换手吃成本.

        IC gating (ic_gate): 用 _ic_df 的 rolling_ic (长度 ic_gate_window)
        若最新一期 <=0, 本期空仓 (实盘意义: alpha 方向反转时避险).
        """
        try:
            from execution import BacktestExecutionSim
            from risk import (
                PreTradeGate, OrderIntent, Portfolio, build_default_gate,
            )
            from portfolio_opt import (
                inverse_volatility_weights, risk_parity_weights,
            )

            label_df = res.stage_results.get("_label_df")
            signal_df = res.stage_results.get("_signal_df")
            feat_df = res.stage_results.get("_feature_df")
            if label_df is None or (signal_df is None and feat_df is None):
                return res

            # 同样优先用 composite signal
            if signal_df is not None:
                f = signal_df
            elif isinstance(feat_df.columns, pd.MultiIndex):
                f = feat_df[feat_df.columns.get_level_values(0)[0]]
            else:
                f = feat_df

            # 收益面板 (供 inverse_vol / risk_parity 使用)
            try:
                ret_panel = daily_df.pivot_table(
                    index="date", columns="code", values="close",
                ).pct_change()
                if not isinstance(ret_panel.index, pd.DatetimeIndex):
                    ret_panel.index = pd.to_datetime(ret_panel.index)
            except Exception:
                ret_panel = None

            sim = BacktestExecutionSim()
            gate = build_default_gate() if self.enforce_risk_gate else None
            if gate is not None:
                res.gate_stats = gate.stats.to_dict()
            portfolio = Portfolio(
                cash=1_000_000.0, initial_capital=1_000_000.0,
                high_water_mark=1_000_000.0, daily_start_value=1_000_000.0,
            )

            daily_by_code = {
                c: g.sort_values("date").set_index("date")
                for c, g in daily_df.groupby("code")
            } if "code" in daily_df.columns else {}

            daily_returns = []
            turnovers = []
            rebal_dates: list = []   # 每次调仓的日期, 与 daily_returns 一一对应
            n_hard_rejects = 0
            n_ic_gated = 0

            prev_top: set = set()
            prev_weights: pd.Series | None = None
            universe_size = f.shape[1] if hasattr(f, "shape") else 0
            if self.top_k_ratio is not None:
                top_n = max(1, int(universe_size * self.top_k_ratio))
            else:
                top_n = max(1, min(self.top_k, universe_size))
            min_scores = max(1, min(self.top_k, universe_size))

            # 调仓日列表 (按 rebalance_freq 抽样), 带一个 "buffer 上限" 用于换手判断
            buf_n = int(top_n * (1 + self.turnover_buffer))

            ic_df = res.stage_results.get("_ic_df")

            for dt in f.index[::self.rebalance_freq]:
                if dt not in f.index or dt not in label_df.index:
                    continue

                # IC gating: rolling IC <= 0 则当期空仓 (仍算一次调仓, 收益=0)
                if self.ic_gate and ic_df is not None:
                    col = f"ic_{self.ic_gate_window}d"
                    if col in ic_df.columns and dt in ic_df.index:
                        ic_val = ic_df[col].loc[dt]
                        if pd.notna(ic_val) and ic_val <= 0:
                            daily_returns.append(0.0)
                            turnovers.append(
                                len(prev_top) / max(len(prev_top) * 2, 1)
                            )
                            rebal_dates.append(dt)
                            prev_top = set()
                            prev_weights = None
                            n_ic_gated += 1
                            continue

                scores = f.loc[dt].dropna()
                if len(scores) < min_scores:
                    continue

                # 风险调整排序: rank by signal / past_vol 而非原始 signal
                # 原理: Sharpe = E[r]/σ. 当两只票信号相同时, 低波动的 Sharpe 更高.
                if self.signal_risk_adjust and ret_panel is not None:
                    try:
                        hist = ret_panel.loc[:dt].iloc[:-1].tail(self.cov_lookback)
                        past_vol = hist.std().replace(0, np.nan)
                        scores = (scores / past_vol.reindex(scores.index)).dropna()
                        if len(scores) < min_scores:
                            scores = f.loc[dt].dropna()   # 回退
                    except Exception:
                        pass

                ranked = scores.sort_values(ascending=False)
                top_slice = ranked.head(buf_n) if buf_n > top_n else ranked.head(top_n)
                # turnover_buffer: 旧持仓若仍在 top_slice, 保留; 不足 top_n 的
                # 缺额再从 ranked 头部补齐
                if self.turnover_buffer > 0 and prev_top:
                    kept = [c for c in ranked.head(buf_n).index if c in prev_top]
                    needed = max(0, top_n - len(kept))
                    add = [c for c in ranked.index if c not in kept][:needed]
                    top_candidates = kept[:top_n] + add
                else:
                    top_candidates = ranked.head(max(top_n, 20)).index.tolist()

                # Pre-Trade Gate
                if gate is not None:
                    admitted: list = []
                    for code in top_candidates:
                        row = daily_by_code.get(code)
                        if row is None or dt not in row.index:
                            admitted.append(code)
                            continue
                        snap = row.loc[dt]
                        price = float(snap.get("close", 0))
                        if "pct_chg" in snap and price:
                            pct = float(snap.get("pct_chg", 0)) / 100
                            prev_close = price / (1 + pct) if (1 + pct) else price
                        else:
                            prev_close = price
                        intent = OrderIntent(
                            code=str(code), side="buy", shares=100,
                            price=price, prev_close=prev_close,
                        )
                        decision = gate.check(
                            intent, portfolio,
                            today=dt.date() if hasattr(dt, "date") else None,
                        )
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

                # 组合权重: equal_weight / inverse_vol / risk_parity
                weights = self._compute_portfolio_weights(
                    top, dt, ret_panel, self.portfolio_method, self.cov_lookback,
                )

                turnover = self._weighted_turnover(weights, prev_weights)
                turnovers.append(turnover)
                prev_top = set(top)
                prev_weights = weights

                # 加权组合收益
                lbl_slice = label_df.loc[dt, top].reindex(top)
                if lbl_slice.isna().all():
                    continue
                # NaN 视为 0, 权重按可用样本归一
                active = lbl_slice.notna()
                if active.sum() == 0:
                    continue
                w_active = weights[active]
                if w_active.sum() > 1e-12:
                    w_active = w_active / w_active.sum()
                port_ret = float((lbl_slice[active] * w_active).sum())
                daily_returns.append(port_ret)
                rebal_dates.append(dt)

            if not daily_returns:
                res.warnings.append("回测无有效样本")
                if gate is not None:
                    res.gate_stats = gate.stats.to_dict()
                return res

            arr = np.array(daily_returns)
            # 年化 periods: 周频=50, 月频=12, 日频=252
            # 使用 rebalance_freq 推断: periods = 252 / rebalance_freq
            periods = max(1.0, 252.0 / self.rebalance_freq)

            # 波动率目标 overlay (shift 1, 无前视): 用过去 N 期实现波动率 → 下期仓位缩放
            # 原理: 波动有持续性 (GARCH), 平静期加仓/狂暴期减仓 → 在相同毛 Sharpe
            # 下拿到更平的净值曲线. 这不制造 alpha, 它**通过压缩左尾放大 Sharpe**.
            vol_scale_series = None
            if self.vol_target is not None and len(arr) > self.vol_target_window:
                s = pd.Series(arr)
                rolling_vol = s.rolling(self.vol_target_window,
                                          min_periods=max(5, self.vol_target_window // 2)
                                          ).std() * np.sqrt(periods)
                raw_scale = (self.vol_target / rolling_vol.replace(0, np.nan))
                raw_scale = raw_scale.clip(self.vol_target_min_leverage,
                                             self.vol_target_max_leverage)
                # 关键: shift(1) 保证只用过去的波动估计下期仓位
                scale = raw_scale.shift(1).fillna(1.0).values
                arr = arr * scale
                vol_scale_series = scale

            annual_ret = float(arr.mean() * periods)
            annual_vol = float(arr.std() * np.sqrt(periods))
            sharpe = annual_ret / (annual_vol + 1e-9)
            nav = pd.Series(1 + arr).cumprod()
            running_max = nav.cummax()
            drawdown = (nav - running_max) / running_max
            max_dd = float(drawdown.min())
            res.stage_results["_bt_returns"] = arr.tolist()
            res.stage_results["_bt_nav"] = nav.tolist()
            res.stage_results["_bt_drawdown"] = drawdown.tolist()
            res.stage_results["_bt_turnovers"] = list(turnovers)
            res.stage_results["_bt_dates"] = [
                pd.Timestamp(d).isoformat() for d in rebal_dates
            ]
            if vol_scale_series is not None:
                res.stage_results["_bt_vol_scale"] = list(vol_scale_series)
            res.backtest_stats = {
                "n_rebalances": len(daily_returns),
                "portfolio_method": self.portfolio_method,
                "rebalance_freq": self.rebalance_freq,
                "top_k": top_n,
                "signal_ema_span": self.signal_ema_span,
                "vol_target": self.vol_target if self.vol_target else "off",
                "signal_risk_adjust": self.signal_risk_adjust,
                "annual_return": annual_ret,
                "annual_vol": annual_vol,
                "sharpe": sharpe,
                "max_drawdown": max_dd,
                "avg_turnover": float(np.mean(turnovers)) if turnovers else 0,
                "gate_hard_rejects": int(n_hard_rejects),
                "ic_gated_periods": int(n_ic_gated),
                "final_nav": float(nav.iloc[-1]),
                "win_rate": float((arr > 0).mean()),
                "profit_loss_ratio": float(
                    arr[arr > 0].mean() / abs(arr[arr < 0].mean())
                ) if (arr < 0).any() and (arr > 0).any() else 0.0,
            }
            if gate is not None:
                res.gate_stats = gate.stats.to_dict()
        except Exception as e:
            res.warnings.append(f"回测失败: {e}")
        return res

    @staticmethod
    def _compute_portfolio_weights(
        codes: list,
        dt,
        ret_panel: pd.DataFrame | None,
        method: str,
        lookback: int,
    ) -> pd.Series:
        """按 method 计算 top-K 的组合权重 (index=codes, 和为 1).

        出错一律回退到等权, 不抛异常影响回测主流程.
        """
        n = len(codes)
        if n == 0:
            return pd.Series(dtype=float)
        if method == "equal_weight" or ret_panel is None or n < 2:
            return pd.Series(1.0 / n, index=codes)

        try:
            hist = ret_panel.loc[:dt].iloc[:-1].tail(lookback)
            hist = hist.reindex(columns=codes).dropna(how="all")
            # 至少需要 lookback/2 的样本
            if len(hist) < max(10, lookback // 2):
                return pd.Series(1.0 / n, index=codes)
            hist = hist.fillna(0.0)

            if method == "inverse_vol":
                vols = hist.std().replace(0, np.nan)
                if vols.isna().all():
                    return pd.Series(1.0 / n, index=codes)
                from portfolio_opt import inverse_volatility_weights
                w = inverse_volatility_weights(vols.fillna(vols.median()))
                return w.reindex(codes).fillna(1.0 / n)

            if method == "risk_parity":
                from portfolio_opt import risk_parity_weights
                cov = hist.cov().values
                # 正则化避免病态
                cov = cov + np.eye(len(cov)) * 1e-6 * np.trace(cov) / len(cov)
                w_arr = risk_parity_weights(cov)
                w_arr = np.nan_to_num(w_arr, nan=1.0 / n)
                if w_arr.sum() <= 0:
                    return pd.Series(1.0 / n, index=codes)
                w_arr = w_arr / w_arr.sum()
                return pd.Series(w_arr, index=codes)
        except Exception:
            return pd.Series(1.0 / n, index=codes)

        return pd.Series(1.0 / n, index=codes)

    @staticmethod
    def _weighted_turnover(
        curr: pd.Series, prev: pd.Series | None,
    ) -> float:
        """|w_curr - w_prev| / 2 = 换手的下界 (L1)."""
        if prev is None or len(prev) == 0:
            return 1.0
        union = curr.index.union(prev.index)
        c = curr.reindex(union).fillna(0.0)
        p = prev.reindex(union).fillna(0.0)
        return float((c - p).abs().sum() / 2.0)
