"""验证 6 个"圈内暗门"模块全部可用."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from tabulate import tabulate


def _a(cond, msg): return bool(cond), msg


# ==================== 1. label_engineering ====================
def test_label_engineering():
    from label_engineering import (
        multi_horizon_label, vol_adjusted_label,
        overnight_label, intraday_label,
        tradeable_mask, event_window_mask,
    )
    from label_engineering.horizons import winsorize_label, cs_rank_label
    from label_engineering.masks import leaky_label_detector

    n = 300
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    np.random.seed(0)
    close = pd.Series(100 + np.cumsum(np.random.randn(n)), index=idx)
    open_ = close.shift(1) * (1 + np.random.randn(n) * 0.005)
    open_ = open_.fillna(close.iloc[0])

    multi = multi_horizon_label(close, open_)
    vol_lab = vol_adjusted_label(close, horizon=5, open_=open_)
    ov = overnight_label(open_, close)
    intra = intraday_label(open_, close)

    df = pd.DataFrame({
        "pct_chg": [11, 3, -10.5, 4, 0, 0],      # 涨停/正常/跌停/正常/停牌/停牌
        "volume":  [1000, 500, 800, 300, 0, 0],
        "ipo_days": [500, 500, 500, 500, 500, 100],
        "name":    ["A", "ST B", "C", "D", "E", "F"],
    })
    mask = tradeable_mask(df)

    # 前视偏差检测: leaky 特征 = label 的 noisy copy (相关性应 > 0.5)
    lbl = multi
    ft = pd.DataFrame({
        "leaky": lbl * 0.9 + np.random.randn(n) * 0.01,
        "ok": np.random.randn(n),
    }, index=lbl.index)
    leaks = leaky_label_detector(ft.dropna(), lbl.dropna(), threshold=0.5)

    return [
        _a(not multi.dropna().empty, "multi_horizon_label 输出"),
        _a(not vol_lab.dropna().empty, "vol_adjusted_label"),
        _a(not ov.dropna().empty, "overnight_label"),
        _a(not intra.dropna().empty, "intraday_label"),
        _a(mask.sum() == 1, f"tradeable_mask 仅 1 行通过 实际 {mask.sum()}"),
        _a(any("leaky" in l for l in leaks), "前视偏差检测到"),
    ]


# ==================== 2. market_microstructure ====================
def test_microstructure():
    from market_microstructure import (
        almgren_chriss_impact, square_root_impact,
        estimate_participation_rate,
        order_imbalance_ratio, weighted_oir,
        effective_spread, realized_spread,
        depth_weighted_midprice,
        lee_ready_classify,
    )
    from market_microstructure.impact import kyle_lambda
    from market_microstructure.order_flow import vpin
    from market_microstructure.spread_factors import (
        roll_implicit_spread, amihud_illiquidity,
    )

    # 冲击模型: 1亿在50亿成交额上
    ac = almgren_chriss_impact(1e8, 5e10, 0.02)
    # 1亿在3亿成交额上 (小盘)
    ac_small = almgren_chriss_impact(1e8, 3e9, 0.025)
    sqrt_impact = square_root_impact(1e8, 5e10, 0.02)

    oir = order_imbalance_ratio([1000, 800, 600], [500, 400, 300])
    woir = weighted_oir([99.9, 99.8, 99.7], [100.1, 100.2, 100.3],
                        [1000, 800, 600], [500, 400, 300], 100.0)

    mid = depth_weighted_midprice([99.9, 99.8], [100.1, 100.2],
                                   [1000, 500], [100, 50])

    eff = effective_spread(100.1, 99.9, 100.1, is_buy=True)
    real = realized_spread(100.1, 100.0, 100.0)

    direction = lee_ready_classify(100.1, 500, 99.9, 100.1, 100.0)

    signed = np.array([100, -50, 200, -150, 300, -100] * 50)
    vp = vpin(signed, bucket_size=500, window_n=20)

    trades = pd.Series([100.1, 100.0, 100.1, 99.9, 100.0] * 10)
    roll = roll_implicit_spread(trades)

    returns = pd.Series([0.01, -0.02, 0.005] * 10)
    volumes = pd.Series([1e6] * 30)
    amihud = amihud_illiquidity(returns, volumes)

    part = estimate_participation_rate(1e9, 3e9)  # 1亿 / 3亿 ≈ 33% EXTREME

    # Kyle lambda
    pc = np.array([0.01, -0.02, 0.015, -0.005, 0.008] * 20)
    sv = np.array([1000, -800, 1200, -400, 600] * 20)
    kl = kyle_lambda(pc, sv)

    return [
        _a(ac["total_bps"] > 0, f"AC 大盘冲击 {ac['total_bps']:.1f}"),
        _a(ac_small["total_bps"] > ac["total_bps"], "小盘冲击 > 大盘"),
        _a(sqrt_impact > 0, "sqrt 冲击"),
        _a(-1 <= oir <= 1, f"OIR 范围 {oir}"),
        _a(oir > 0, "OIR 买压主导 (测试数据设置)"),
        _a(99.9 <= mid <= 100.1, f"micro-price 合理 {mid}"),
        _a(eff > 0, "有效价差"),
        _a(real >= 0, "已实现价差"),
        _a(direction == 1, f"Lee-Ready 判买方 {direction}"),
        _a(0 <= vp <= 1, f"VPIN 范围 {vp:.3f}"),
        _a(roll >= 0, "Roll 价差"),
        _a(not amihud.dropna().empty, "Amihud"),
        _a(part["risk_level"] in ("EXTREME", "HIGH"), f"参与率 risk={part['risk_level']}"),
        _a(isinstance(kl, float), "Kyle lambda"),
    ]


# ==================== 3. alpha_decay ====================
def test_alpha_decay():
    from alpha_decay import (
        rolling_ic_decay, half_life_estimate, alpha_health_score,
        factor_crowding_index, turnover_signal, public_strategy_overlap,
    )
    from alpha_decay.monitor import ic_ir
    from alpha_decay.crowding import alpha_portfolio_correlation

    n = 300
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    codes = [f"S{i}" for i in range(20)]
    np.random.seed(1)
    factor = pd.DataFrame(np.random.randn(n, 20), index=idx, columns=codes)
    fw_ret = factor * 0.05 + np.random.randn(n, 20) * 0.1
    fw_ret = pd.DataFrame(fw_ret.values, index=idx, columns=codes)

    ic_df = rolling_ic_decay(factor, fw_ret)
    half = half_life_estimate(ic_df["ic_daily"])

    score = alpha_health_score(0.04, 0.06, 0.5, 0.4, 1.2, 2.0)

    # 拥挤度
    f_ret = pd.Series(np.random.randn(n) * 0.01, index=idx)
    turnover = pd.Series(np.random.uniform(0.2, 0.5, n), index=idx)
    crowd = factor_crowding_index(f_ret, turnover)

    # turnover signal
    positions = pd.DataFrame(np.random.rand(n, 20), index=idx, columns=codes)
    positions = positions.div(positions.sum(axis=1), axis=0)
    t_sig = turnover_signal(positions)

    # public overlap
    mine = pd.Series(np.random.randn(100))
    public = mine * 0.8 + pd.Series(np.random.randn(100)) * 0.2
    overlap = public_strategy_overlap(mine, public)

    # ic ir
    ir = ic_ir(ic_df["ic_daily"])

    # 因子相关性
    corr = alpha_portfolio_correlation(factor.iloc[:, :5])

    return [
        _a("ic_20d" in ic_df.columns, "rolling IC 多窗口"),
        _a("action" in score, f"健康度: {score['action']}"),
        _a(not crowd.dropna().empty, "拥挤度"),
        _a("mean_turnover" in t_sig, "turnover 信号"),
        _a(0 <= abs(overlap) <= 1, "公开策略重合度"),
        _a("icir" in ir, "IC IR"),
        _a(corr.shape == (5, 5), "因子相关矩阵"),
    ]


# ==================== 4. corporate_actions ====================
def test_corporate_actions():
    from corporate_actions import (
        EarningsMask, UnlockMask, BlockTradeMask,
        combined_event_mask,
        unlock_pressure_factor, block_trade_discount_factor,
        insider_net_activity_factor,
    )

    df = pd.DataFrame({
        "code": ["300750"] * 5 + ["600519"] * 5,
        "date": (pd.date_range("2024-01-01", periods=5).tolist() * 2),
        "close": [100] * 10,
    })

    em = EarningsMask({"300750": ["2024-01-03"]})
    mask_em = em.apply(df)

    um = UnlockMask({"300750": [("2024-01-10", 0.05)]})
    mask_um = um.apply(df)

    bm = BlockTradeMask({"300750": [("2024-01-01", -0.05)]})
    mask_bm = bm.apply(df)

    combined = combined_event_mask(df, em, um, bm)

    # 因子
    pressure = unlock_pressure_factor(
        pd.Timestamp("2024-01-01"),
        [(pd.Timestamp("2024-01-15"), 0.10)],
    )
    bt = block_trade_discount_factor(
        [(pd.Timestamp("2024-01-01"), -0.04, 1e7)],
        pd.Timestamp("2024-01-02"),
    )
    ins = insider_net_activity_factor(
        [(pd.Timestamp("2024-01-01"), "sell", 500),
         (pd.Timestamp("2024-01-05"), "sell", 300),
         (pd.Timestamp("2024-01-08"), "sell", 400)],
        current_date=pd.Timestamp("2024-02-01"),
    )

    return [
        _a(mask_em.sum() < 10, f"财报屏蔽部分样本 剩 {mask_em.sum()}"),
        _a(mask_bm.sum() < 10, f"大宗屏蔽 剩 {mask_bm.sum()}"),
        _a(pressure > 0, f"解禁压力 {pressure:.3f}"),
        _a(bt["avg_discount"] < 0, "大宗折价负"),
        _a(bt["recent_large_discount"] < 0, "最近大幅折价"),
        _a(ins["net_count"] == -3, "减持 3 次"),
        _a(ins["concentration"] >= 3, "集中减持侦测"),
    ]


# ==================== 5. barra_neutralize ====================
def test_barra_neutralize():
    from barra_neutralize import (
        compute_size, compute_beta, compute_momentum,
        compute_residual_volatility, compute_liquidity,
        compute_all_styles,
        neutralize_by_regression, industry_dummies,
    )
    from barra_neutralize.neutralize import explained_variance_by_styles
    from barra_neutralize.style_factors import compute_non_linear_size

    n_stocks = 30
    n_days = 250
    codes = [f"S{i}" for i in range(n_stocks)]
    np.random.seed(2)
    market_cap = pd.Series(np.random.uniform(5e9, 5e11, n_stocks), index=codes)

    rets = pd.DataFrame(
        np.random.randn(n_days, n_stocks) * 0.02,
        index=pd.date_range("2024-01-01", periods=n_days),
        columns=codes,
    )
    mkt_ret = rets.mean(axis=1)
    t1 = pd.Series(np.random.uniform(0.1, 2.0, n_stocks), index=codes)
    t3 = pd.Series(np.random.uniform(0.1, 2.0, n_stocks), index=codes)
    t12 = pd.Series(np.random.uniform(0.1, 2.0, n_stocks), index=codes)

    size = compute_size(market_cap)
    beta = compute_beta(rets, mkt_ret)
    mom = compute_momentum(rets)
    rv = compute_residual_volatility(rets, mkt_ret)
    liq = compute_liquidity(t1, t3, t12)
    nls = compute_non_linear_size(size)

    all_styles = compute_all_styles(market_cap, rets, mkt_ret, t1, t3, t12)

    # 中性化
    alpha = pd.Series(np.random.randn(n_stocks), index=codes)
    industries = pd.Series(np.random.choice(["tech", "fin", "cons"], n_stocks),
                           index=codes)
    clean = neutralize_by_regression(alpha, all_styles, industries)

    # 解释变异
    r2 = explained_variance_by_styles(alpha, all_styles)

    return [
        _a(abs(size.mean()) < 0.1 and abs(size.std() - 1) < 0.1, "Size Z-score"),
        _a(not beta.empty, "Beta"),
        _a(not mom.empty, "Momentum"),
        _a(not rv.empty, "ResidualVol"),
        _a(not liq.empty, "Liquidity"),
        _a(not nls.empty, "NonLinSize"),
        _a(all_styles.shape[1] == 6, "6 个风格因子"),
        _a(not clean.empty, "中性化输出"),
        _a(0 <= r2 <= 1, f"R² 范围 {r2:.3f}"),
    ]


# ==================== 6. portfolio_opt ====================
def test_portfolio_opt():
    from portfolio_opt import (
        risk_parity_weights, inverse_volatility_weights,
        vol_target_scale, calculate_kelly_with_drawdown,
        mean_variance_optimize, black_litterman_posterior,
    )
    from portfolio_opt.risk_parity import portfolio_risk_breakdown
    from portfolio_opt.vol_targeting import drawdown_scaler

    n = 10
    codes = [f"S{i}" for i in range(n)]
    np.random.seed(3)
    vols = pd.Series(np.random.uniform(0.15, 0.4, n), index=codes)
    cov = pd.DataFrame(np.random.randn(n, n), index=codes, columns=codes)
    cov = cov @ cov.T / 100              # PSD
    for c in codes:
        cov.loc[c, c] = vols[c] ** 2

    inv_vol_w = inverse_volatility_weights(vols)
    rp = risk_parity_weights(cov.values)
    rp_w = pd.Series(rp, index=codes)
    breakdown = portfolio_risk_breakdown(rp_w, cov)

    scale = vol_target_scale(current_vol=0.25, target_vol=0.15)

    kelly = calculate_kelly_with_drawdown(win_rate=0.55, avg_win=0.08,
                                          avg_loss=0.04)

    dd_scale_ok = drawdown_scaler(-0.02)
    dd_scale_warn = drawdown_scaler(-0.10)
    dd_scale_halt = drawdown_scaler(-0.20)

    mu = pd.Series(np.random.randn(n) * 0.01, index=codes)
    mvo_w = mean_variance_optimize(mu, cov, max_weight=0.2)

    # Black-Litterman
    mkt_w = pd.Series(np.ones(n) / n, index=codes)
    bl = black_litterman_posterior(
        mkt_w, cov,
        views={"S0": 0.10, "S1": -0.05},
        view_confidence={"S0": 0.7, "S1": 0.3},
    )

    return [
        _a(abs(inv_vol_w.sum() - 1) < 1e-6, "inv_vol 归一化"),
        _a(abs(rp_w.sum() - 1) < 1e-3, "risk_parity 归一化"),
        _a("risk_contribution" in breakdown.columns, "风险分解"),
        _a(0.3 <= scale <= 1.5, f"vol target scale {scale}"),
        _a("fractional_kelly" in kelly, "凯利仓位"),
        _a(kelly["fractional_kelly"] < kelly["full_kelly"], "fraction < full"),
        _a(dd_scale_ok == 1.0, "回撤正常 =1"),
        _a(0 < dd_scale_warn < 1, f"回撤预警 {dd_scale_warn:.2f}"),
        _a(dd_scale_halt == 0, "回撤熔断 =0"),
        _a(abs(mvo_w.sum() - 1) < 1e-6, "MVO 归一化"),
        _a(not bl.empty, "BL posterior"),
        _a(bl["S0"] > 0, "BL S0 买入 view 生效"),
    ]


def main():
    all_rows = []
    total_pass = total = 0
    for name, fn in [
        ("label_engineering", test_label_engineering),
        ("microstructure", test_microstructure),
        ("alpha_decay", test_alpha_decay),
        ("corporate_actions", test_corporate_actions),
        ("barra_neutralize", test_barra_neutralize),
        ("portfolio_opt", test_portfolio_opt),
    ]:
        try:
            results = fn()
        except Exception as e:
            import traceback
            all_rows.append([name, f"EXCEPTION: {e}", "✗"])
            traceback.print_exc()
            total += 1
            continue
        for ok, desc in results:
            all_rows.append([name, desc, "✓" if ok else "✗"])
            total_pass += ok
            total += 1

    print("\n==== 6 大暗门模块测试 ====")
    print(tabulate(all_rows, headers=["模块", "用例", "结果"]))
    print(f"\n通过率: {total_pass}/{total}")
    sys.exit(0 if total_pass == total else 1)


if __name__ == "__main__":
    main()
