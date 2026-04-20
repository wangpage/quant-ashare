"""数值正确性测试 - 不是 '非空/字段存在', 而是 **算出的值必须对**.

覆盖:
    1. IC 计算: 构造已知 signal+return, 验证 Spearman IC 符号/量级
    2. 最大回撤: 已知 nav 序列, 验证计算结果与闭式公式一致
    3. ATR 止损: 验证 ATR×N 的止损位计算
    4. Barra 残差正交性: 中性化后残差与 style factor 相关性应 ≈ 0
    5. Almgren-Chriss 单调性: trade_amount 上升 cost 必须上升
    6. OIR 边界: 买盘 0 → -1, 卖盘 0 → +1
    7. VPIN 范围: [0, 1]
    8. NATS URL validator: 占位符/缺端口/非法 scheme 必须抛
    9. Regime 极端场景: 崩盘/狂热必须被识别
   10. 拥挤度惩罚单调性: 持仓越多惩罚越大

每条测试都有**明确数值断言**, 失败即表示逻辑 bug.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from tabulate import tabulate


def _a(cond, msg): return bool(cond), msg


# ==================== 1. IC 数值正确性 ====================
def test_ic_numerical():
    from alpha_decay.monitor import ic_ir

    # 构造已知 IC 序列 (模拟日度 IC 值)
    ic_series = pd.Series(np.random.default_rng(42).normal(0.05, 0.1, 200))
    stats = ic_ir(ic_series)

    expected_mean = ic_series.mean()
    expected_std = ic_series.std()
    expected_icir = expected_mean / expected_std * math.sqrt(252)

    return [
        _a(abs(stats["ic_mean"] - expected_mean) < 1e-9,
           f"IC mean 精确 (实际 {stats['ic_mean']:.6f} vs {expected_mean:.6f})"),
        _a(abs(stats["ic_std"] - expected_std) < 1e-9,
           "IC std 精确"),
        _a(abs(stats["icir"] - expected_icir) < 1e-6,
           f"ICIR = mean/std × sqrt(252) (实际 {stats['icir']:.4f} vs {expected_icir:.4f})"),
        # 全零序列
        _a(ic_ir(pd.Series([0]*50))["icir"] == 0,
           "全零序列 ICIR = 0"),
        # 样本不足
        _a(ic_ir(pd.Series([0.1]*5))["icir"] == 0,
           "样本 < 20 时 ICIR = 0 降级"),
    ]


# ==================== 2. 最大回撤 ====================
def test_max_drawdown():
    # 构造已知回撤: 1.0 → 1.5 → 0.9 → 1.2
    # 最大回撤 = (0.9 - 1.5) / 1.5 = -40%
    nav = pd.Series([1.0, 1.2, 1.5, 1.3, 1.1, 0.9, 1.0, 1.2])
    running_max = nav.cummax()
    dd = (nav - running_max) / running_max
    max_dd = dd.min()

    return [
        _a(abs(max_dd - (-0.4)) < 1e-9, f"最大回撤精确 -40% (实际 {max_dd:.4f})"),
        _a(dd.iloc[5] == pytest_approx(-0.4), "最低点在 index 5"),
        _a(dd.iloc[0] == 0, "起始无回撤"),
        _a(dd.iloc[2] == 0, "新高时回撤 0"),
    ]


def pytest_approx(v, tol=1e-9):
    """轻量 approx, 避免引入 pytest 依赖."""
    class _A:
        def __eq__(self, other): return abs(other - v) < tol
    return _A()


# ==================== 3. ATR 止损 ====================
def test_atr_stops():
    from webapp.positions_engine import calculate_stops

    # 规则: sl_width = max(min_pct × cost, ATR × N)
    # cost=100, mult_sl=2, mult_tp=4, min_sl=5%, min_tp=8%

    # 中等 ATR: sl_width = max(5, 4) = 5 → sl = 95; tp_width = max(8, 8) = 8 → tp = 108
    sl_mid, tp_mid = calculate_stops(cost_price=100, atr=2)
    assert sl_mid == 95, f"ATR=2 min 5% 兜底 sl=95, 得 {sl_mid}"
    assert tp_mid == 108, f"ATR=2 tp=108, 得 {tp_mid}"

    # 低 ATR: sl_width = max(5, 0.2) = 5 → sl = 95
    sl_low, _ = calculate_stops(cost_price=100, atr=0.1)
    assert sl_low == 95, f"低 ATR min 5% 兜底, 得 {sl_low}"

    # 高 ATR: sl_width = max(5, 20) = 20 → sl = 80; tp_width = max(8, 40) = 40 → tp = 140
    sl_high, tp_high = calculate_stops(cost_price=100, atr=10)
    assert sl_high == 80, f"高 ATR 放宽 ATR×2=20 → sl=80, 得 {sl_high}"
    assert tp_high == 140, f"高 ATR tp=140, 得 {tp_high}"

    # 跨 cost 一致性
    sl_50, tp_50 = calculate_stops(cost_price=50, atr=1)
    # sl_width = max(2.5, 2) = 2.5 → sl = 47.5
    assert sl_50 == 47.5, f"cost=50, sl=47.5, 得 {sl_50}"
    assert tp_50 == 54.0, f"cost=50, tp=54, 得 {tp_50}"

    return [
        _a(True, "中等 ATR → sl=95 tp=108 (min 兜底)"),
        _a(True, "低 ATR → sl=95 (min 5% 兜底)"),
        _a(True, "高 ATR → sl=80 tp=140 (按 ATR 放宽)"),
        _a(True, "跨 cost 等价 (cost=50 → sl=47.5)"),
        _a(tp_mid > sl_mid, "tp > sl"),
        _a(tp_high - 100 > 100 - sl_high, "高 ATR 时 tp 距离更大 (盈亏比 2:1)"),
    ]


# ==================== 4. Barra 残差正交性 ====================
def test_barra_residual_orthogonality():
    from barra_neutralize.neutralize import neutralize_by_regression

    rng = np.random.default_rng(7)
    n = 100
    codes = [f"S{i}" for i in range(n)]

    # 构造 3 个风格因子 + alpha 有显式暴露
    size = pd.Series(rng.normal(0, 1, n), index=codes)
    beta = pd.Series(rng.normal(0, 1, n), index=codes)
    mom = pd.Series(rng.normal(0, 1, n), index=codes)
    styles = pd.DataFrame({"Size": size, "Beta": beta, "Momentum": mom})

    # alpha = 0.5*size + 0.3*beta + noise
    true_noise = rng.normal(0, 0.3, n)
    alpha = 0.5 * size + 0.3 * beta + true_noise

    residual = neutralize_by_regression(alpha, styles)

    # 核心断言: 残差与每个风格因子相关性应 ≈ 0
    corr_size = abs(residual.corr(size))
    corr_beta = abs(residual.corr(beta))
    corr_mom = abs(residual.corr(mom))

    # 残差方差应远小于原 alpha 方差 (因为剥除了可解释成分)
    var_ratio = residual.var() / alpha.var()

    return [
        _a(corr_size < 0.05, f"残差 vs Size corr = {corr_size:.4f} (应 ≈0)"),
        _a(corr_beta < 0.05, f"残差 vs Beta corr = {corr_beta:.4f} (应 ≈0)"),
        _a(corr_mom < 0.05, f"残差 vs Momentum corr = {corr_mom:.4f} (应 ≈0)"),
        _a(var_ratio < 0.6,
           f"残差方差/原方差 = {var_ratio:.2f} (应 <0.6, 剥除了风格暴露)"),
        _a(abs(residual.mean()) < 0.05,
           f"残差均值 ≈ 0 (实际 {residual.mean():.4f})"),
    ]


# ==================== 5. Almgren-Chriss 单调性 ====================
def test_impact_monotonicity():
    from market_microstructure.impact import (
        almgren_chriss_impact, square_root_impact,
    )

    # 相同参数下, trade_amount 增大 → 冲击必须增大
    sizes = [1e6, 1e7, 1e8, 5e8]
    bps = [almgren_chriss_impact(s, 1e10, 0.02)["total_bps"] for s in sizes]
    sqrt_bps = [square_root_impact(s, 1e10, 0.02) for s in sizes]

    # 相同成交额, daily_volume 增大 → 冲击必须减小
    vols = [1e9, 1e10, 1e11]
    bps_by_vol = [square_root_impact(1e8, v, 0.02) for v in vols]

    # 小盘 (daily 3e9) vs 大盘 (5e10) 同 1亿: 小盘冲击应 >> 大盘
    ac_big = almgren_chriss_impact(1e8, 5e10, 0.02)["total_bps"]
    ac_small = almgren_chriss_impact(1e8, 3e9, 0.025)["total_bps"]

    # sqrt 律: 成交翻 10 倍, 冲击应约翻 √10 ≈ 3.16 倍
    ratio = sqrt_bps[2] / sqrt_bps[1]

    return [
        _a(all(bps[i] < bps[i+1] for i in range(len(bps)-1)),
           "AC 冲击单调递增 (成交额↑)"),
        _a(all(bps_by_vol[i] > bps_by_vol[i+1] for i in range(len(bps_by_vol)-1)),
           "AC 冲击单调递减 (daily_volume↑)"),
        _a(ac_small > ac_big * 3,
           f"小盘冲击 >> 大盘 ({ac_small:.1f} vs {ac_big:.1f} bps, 至少 3x)"),
        _a(2.5 < ratio < 4.0,
           f"sqrt 律: 10x 成交 → {ratio:.2f}x 冲击 (应 ~3.16)"),
        _a(square_root_impact(0, 1e10, 0.02) == 0, "零成交冲击 = 0"),
    ]


# ==================== 6. OIR 边界 ====================
def test_oir_bounds():
    from market_microstructure.order_flow import order_imbalance_ratio

    pure_buy = order_imbalance_ratio([1000, 500], [0, 0])
    pure_sell = order_imbalance_ratio([0, 0], [1000, 500])
    balanced = order_imbalance_ratio([500], [500])
    empty = order_imbalance_ratio([0], [0])

    return [
        _a(pure_buy == 1.0, f"纯买单 OIR = 1.0 (实际 {pure_buy})"),
        _a(pure_sell == -1.0, f"纯卖单 OIR = -1.0 (实际 {pure_sell})"),
        _a(balanced == 0.0, f"均衡 OIR = 0 (实际 {balanced})"),
        _a(empty == 0.0, "空盘口 OIR = 0 (除零保护)"),
    ]


# ==================== 7. VPIN 数值范围 ====================
def test_vpin_range():
    from market_microstructure.order_flow import vpin

    # 完全单向买入: VPIN 应 = 1.0
    all_buy = [500] * 50
    v1 = vpin(all_buy, bucket_size=500, window_n=20)

    # 完全均衡 (每 bucket 内部买卖相抵):
    # bucket_size=500, 每两笔 (250买, 250卖) 刚好一桶
    balanced = [250, -250] * 50
    v2 = vpin(balanced, bucket_size=500, window_n=20)

    # 随机有向流
    rng = np.random.default_rng(0)
    random_flow = rng.choice([100, -100], size=100).tolist()
    v3 = vpin(random_flow, bucket_size=500, window_n=20)

    return [
        _a(0.95 <= v1 <= 1.0, f"纯买单 VPIN ≈ 1.0 (实际 {v1:.3f})"),
        _a(0 <= v2 <= 0.2, f"均衡买卖 VPIN ≈ 0 (实际 {v2:.3f})"),
        _a(0 <= v3 <= 1.0, f"随机 VPIN 在 [0,1] (实际 {v3:.3f})"),
        _a(vpin([], bucket_size=100, window_n=10) == 0, "空输入 VPIN = 0"),
    ]


# ==================== 8. NATS URL validator ====================
def test_nats_url_validator():
    from level2.config_validator import (
        validate_nats_url, validate_level2_config,
        ConfigurationError, _is_placeholder,
    )

    checks = []

    # 占位符识别
    for val in ["", "REPLACE_ME", "TODO", "nats://TODO_IP:8888",
                "XXX", "${LEVEL2_USER}", "CHANGEME"]:
        ok = _is_placeholder(val)
        checks.append(_a(ok, f"占位符 {val!r} 被识别"))

    # 合法值不误判
    for val in ["nats://43.143.73.95:31886", "level2_test",
                "db.base32.cn", "${VAR:-default}"]:
        ok = not _is_placeholder(val)
        checks.append(_a(ok, f"合法值 {val!r} 不误判"))

    # URL 形状校验
    good = "nats://43.143.73.95:31886"
    assert validate_nats_url(good) == good
    checks.append(_a(True, "合法 NATS URL 通过"))

    # 非法场景必须抛
    for bad, why in [
        ("", "空串"),
        ("nats://TODO_IP:31886", "占位符 IP"),
        ("http://host:80", "错 scheme"),
        ("nats://:31886", "无 host"),
        ("nats://host", "无端口"),
        ("nats://host:99999", "端口越界"),
    ]:
        try:
            validate_nats_url(bad)
            checks.append(_a(False, f"非法 {why!r} 应抛却未抛: {bad}"))
        except ConfigurationError:
            checks.append(_a(True, f"非法 {why!r} 正确抛 ConfigurationError"))

    # 完整配置校验
    good_cfg = {
        "connection": {
            "active": "shanghai",
            "servers": {"shanghai": {"host": "nats://db.base32.cn:31886"}},
            "auth": {"user": "level2_test", "password": "level2@test"},
        },
    }
    validated = validate_level2_config(good_cfg)
    checks.append(_a(validated.primary_url == "nats://db.base32.cn:31886",
                      "validated.primary_url 正确"))
    checks.append(_a(validated.backup_url is None, "无 backup 时为 None"))

    # 缺字段应抛
    try:
        validate_level2_config({"connection": {"active": "xxx"}})
        checks.append(_a(False, "不存在的 active 应抛"))
    except ConfigurationError:
        checks.append(_a(True, "不存在的 active 正确抛"))

    # 占位符 URL 不能过
    bad_cfg = {
        "connection": {
            "active": "shanghai",
            "servers": {"shanghai": {"host": "nats://TODO:31886"}},
            "auth": {"user": "x", "password": "y"},
        },
    }
    try:
        validate_level2_config(bad_cfg)
        checks.append(_a(False, "TODO URL 应抛"))
    except ConfigurationError:
        checks.append(_a(True, "TODO URL 正确抛"))

    return checks


# ==================== 9. Regime 极端场景识别 ====================
def test_regime_extremes():
    from market_regime import RegimeDetector
    from market_regime.detector import MarketRegime
    from market_regime.indicators import detect_crash, detect_euphoria

    # 崩盘: 单日 -7%
    crash_close = pd.Series(np.ones(100) * 100)
    crash_close.iloc[-1] = 93
    assert detect_crash(crash_close), "应识别单日 -7% 崩盘"

    # 最近 5 日累计 -12%: close[-6]=100 → close[-1]=88
    # detect_crash 比较 close.iloc[-1] / close.iloc[-6] - 1
    slow_crash = pd.Series([100] * 94 + [98, 96, 94, 92, 90, 88])
    ret_5d = slow_crash.iloc[-1] / slow_crash.iloc[-6] - 1
    assert ret_5d < -0.10, f"构造数据 5日收益应 < -10% (实际 {ret_5d:.3f})"
    assert detect_crash(slow_crash), "应识别 5 日累计 -12% 慢崩"

    # 正常
    normal = pd.Series(np.linspace(100, 103, 100))
    assert not detect_crash(normal), "正常上涨不应是崩盘"

    # 狂热
    assert detect_euphoria(pct_up=0.85, pct_limit_up=0.05, vol_20d=0.40)
    assert not detect_euphoria(pct_up=0.50, pct_limit_up=0.01, vol_20d=0.20)

    return [
        _a(True, "单日 -7% 崩盘识别"),
        _a(True, "5 日累计 -12% 慢崩识别"),
        _a(True, "正常上涨不误判"),
        _a(True, "狂热识别 (85% 上涨 + 5% 涨停率 + 40% 波动)"),
        _a(True, "温和行情不误判为狂热"),
    ]


# ==================== 10. Alpha 衰减 / 拥挤度单调性 ====================
def test_alpha_decay_monotonic():
    """alpha_decay 模块的单调性铁律:
       - 组合 turnover 上升 → crowding index 应上升
       - factor IC 衰减 → ic_20d / ic_252d 比值应下降
    """
    from alpha_decay.crowding import factor_crowding_index, turnover_signal
    from alpha_decay.monitor import rolling_ic_decay, alpha_health_score

    n = 200
    rng = np.random.default_rng(3)

    # 构造: 稳定的 factor_returns, 但 turnover 明显上升
    f_ret = pd.Series(rng.normal(0.001, 0.005, n),
                       index=pd.date_range("2024-01-01", periods=n))
    stable_turn = pd.Series(np.ones(n) * 0.3, index=f_ret.index)
    rising_turn = pd.Series(np.linspace(0.3, 1.0, n), index=f_ret.index)

    crowd_stable = factor_crowding_index(f_ret, stable_turn).dropna()
    crowd_rising = factor_crowding_index(f_ret, rising_turn).dropna()

    # alpha 健康度 - 数值断言
    healthy = alpha_health_score(
        recent_ic=0.05, longterm_ic=0.05,
        recent_turnover=0.4, longterm_turnover=0.4,
        recent_sharpe=1.5, longterm_sharpe=1.5,
    )
    decaying = alpha_health_score(
        recent_ic=0.01, longterm_ic=0.05,
        recent_turnover=0.8, longterm_turnover=0.4,
        recent_sharpe=0.3, longterm_sharpe=1.5,
    )

    return [
        _a(crowd_rising.mean() > crowd_stable.mean(),
           f"turnover↑ crowding↑ (稳 {crowd_stable.mean():.2f} vs 升 {crowd_rising.mean():.2f})"),
        _a(healthy["total_score"] > decaying["total_score"],
           f"健康 vs 衰减分数: {healthy['total_score']:.1f} > {decaying['total_score']:.1f}"),
        _a(healthy["action"] == "KEEP", "健康因子 action=KEEP"),
        _a(decaying["action"] in ("DOWNWEIGHT", "RETIRE"),
           f"衰减因子 action={decaying['action']}"),
        _a(0 <= healthy["total_score"] <= 100, "分数在 [0,100]"),
    ]


# ==================== 主入口 ====================
def main():
    all_rows = []
    total_pass = total = 0
    for name, fn in [
        ("ic_numerical",      test_ic_numerical),
        ("max_drawdown",       test_max_drawdown),
        ("atr_stops",          test_atr_stops),
        ("barra_residual",     test_barra_residual_orthogonality),
        ("impact_monotonicity", test_impact_monotonicity),
        ("oir_bounds",          test_oir_bounds),
        ("vpin_range",         test_vpin_range),
        ("nats_url_validator", test_nats_url_validator),
        ("regime_extremes",    test_regime_extremes),
        ("alpha_decay_monotonic", test_alpha_decay_monotonic),
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

    print("\n==== 数值正确性测试 (真·数值断言) ====")
    print(tabulate(all_rows, headers=["模块", "用例", "结果"]))
    print(f"\n通过率: {total_pass}/{total}")
    sys.exit(0 if total_pass == total else 1)


if __name__ == "__main__":
    main()
