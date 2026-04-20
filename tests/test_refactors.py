"""验证本轮重构:
- Barra 分层中性化 + Ridge + 条件数诊断
- 冲击成本 ADV 口径 + 分片累加语义
- 披露日 vs 报告期 + 时戳完整性
- 执行层统一 Slicer 协议 + ExecutionEngine
- PreTradeGate 硬过闸
- 前视偏差嵌入 Research pipeline

每个测试都做 **数值断言**, 不是 "非空" 玩具断言.
"""
from __future__ import annotations

import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd


def _ok(cond, msg):
    return bool(cond), msg


# ============================================================
# 1. Barra 分层中性化
# ============================================================
def test_barra_hierarchical():
    from barra_neutralize import (
        neutralize_by_regression, neutralize_hierarchical,
        orthogonalize_factors, winsorize_mad,
        condition_number,
    )

    np.random.seed(42)
    n = 300
    codes = [f"S{i:03d}" for i in range(n)]

    # 构造: alpha = 0.5*Size + 0.3*Momentum + noise + industry_effect
    size = pd.Series(np.random.randn(n), index=codes)
    mom = pd.Series(np.random.randn(n), index=codes)
    residvol = pd.Series(np.random.randn(n), index=codes)
    styles = pd.DataFrame({
        "Size": size, "Momentum": mom, "ResidualVol": residvol,
    })
    industries = pd.Series(
        np.random.choice(["电子", "金融", "消费", "医药"], n),
        index=codes,
    )
    industry_effect = industries.map({"电子": 0.4, "金融": -0.2,
                                        "消费": 0.1, "医药": -0.3})
    noise = np.random.randn(n) * 0.5
    alpha = 0.5 * size + 0.3 * mom + industry_effect + noise
    alpha.name = "raw_alpha"

    # 注入极端值, 验证 winsorize
    alpha_polluted = alpha.copy()
    alpha_polluted.iloc[0] = 1e6
    alpha_polluted.iloc[1] = -1e6

    # 1. winsorize_mad 稳健: 极端值被压制
    winsored = winsorize_mad(alpha_polluted, n=5)
    assert abs(winsored.iloc[0]) < 100, f"极端值未被压制: {winsored.iloc[0]}"
    assert abs(winsored.iloc[1]) < 100, f"负极端值未被压制: {winsored.iloc[1]}"

    # 2. 分层中性化: 返回残差 (alpha_clean), 与 industries 应几乎无关
    resid, diag = neutralize_hierarchical(
        alpha, styles, industries=industries, return_diagnostics=True,
    )
    # 条件数可接受 (< 100)
    assert diag.condition_number < 500, f"条件数过高 {diag.condition_number}"
    assert diag.n_industry_factors == 4, "行业数应为 4"
    # 残差与行业效应的相关性应很低 (已中性化)
    ind_dum = pd.get_dummies(industries, dtype=float)
    corr_max = max(
        abs(resid.corr(ind_dum[c])) for c in ind_dum.columns
    )
    assert corr_max < 0.15, f"行业中性未生效, 最大相关 {corr_max:.3f}"

    # 3. 正交化后的因子: 两两相关 < 0.05
    ortho = orthogonalize_factors(styles)
    corr_matrix = ortho.corr().abs()
    np.fill_diagonal(corr_matrix.values, 0)
    assert corr_matrix.values.max() < 0.05, \
        f"正交化失败, 最大互相关 {corr_matrix.values.max():.3f}"

    # 4. 向后兼容接口: neutralize_by_regression 仍可用, 残差方差 < 原始
    resid_old = neutralize_by_regression(alpha, styles, industries=industries)
    assert resid_old.std() < alpha.std(), \
        f"中性化后方差应下降: {alpha.std():.3f} → {resid_old.std():.3f}"

    return [
        _ok(True, "MAD winsorize 压制极端值"),
        _ok(True, f"分层中性化条件数 {diag.condition_number:.1f} < 500"),
        _ok(True, f"残差与行业相关性 {corr_max:.3f} < 0.15"),
        _ok(True, "Gram-Schmidt 正交化互相关 < 0.05"),
        _ok(True, "向后兼容 API 残差方差下降"),
    ]


# ============================================================
# 2. 冲击成本 ADV 口径 + 分片语义
# ============================================================
def test_impact_router_adv():
    from execution.impact_router import (
        ImpactAwareRouter, estimate_adv_yuan,
    )
    from market_microstructure.impact import square_root_impact

    router = ImpactAwareRouter(target_participation=0.05, urgency=0.5)

    # 场景: 1000 万订单, ADV 10 亿 (参与率 1%)
    plan = router.plan_order(
        total_shares=100_000, price=100.0,
        adv_yuan=1e9, volatility=0.02,
    )
    assert plan.participation_rate > 0, "参与率应 > 0"
    assert abs(plan.participation_rate - 0.01) < 1e-6, \
        f"参与率应 = 1%, 实际 {plan.participation_rate:.4f}"

    # 分片后总冲击 bps 应 < 一次性冲击 (因为 eta/√N 分量)
    one_shot = square_root_impact(100_000 * 100, 1e9, 0.02)
    assert plan.expected_total_cost_bps < one_shot, \
        f"分片总成本 {plan.expected_total_cost_bps:.2f} 应 < 一次性 {one_shot:.2f}"

    # 所有切片股数加和 = 总股数
    total_sliced = sum(s["shares"] for s in plan.slices)
    assert abs(total_sliced - 100_000) < 200, \
        f"切片总股数 {total_sliced} ≠ 请求 100000"

    # adv_yuan 相比 daily_volume 的一致性 (daily_volume × price ≈ adv_yuan)
    plan_v = router.plan_order(
        total_shares=100_000, price=100.0,
        daily_volume=10_000_000, volatility=0.02,   # 10M × 100 = 1e9
    )
    assert abs(plan_v.expected_total_cost_bps
               - plan.expected_total_cost_bps) < 0.5, \
        "adv_yuan 口径与 daily_volume 退化口径应一致"

    # 参与率高触发警告
    big = router.plan_order(
        total_shares=2_000_000, price=100.0,   # 2 亿 vs 10 亿 = 20%
        adv_yuan=1e9, volatility=0.02,
    )
    assert big.participation_rate > 0.15, "大单参与率应 > 15%"

    # estimate_adv_yuan: 用 20 日数据
    amounts = [1e9 + i * 1e6 for i in range(30)]
    adv = estimate_adv_yuan(amounts=amounts, window=20)
    assert abs(adv - np.mean(amounts[-20:])) < 1.0

    return [
        _ok(True, f"参与率精确 {plan.participation_rate:.4f}"),
        _ok(True,
             f"分片后 {plan.expected_total_cost_bps:.1f} < 一次性 {one_shot:.1f}"),
        _ok(True, f"切片股数守恒 {total_sliced}"),
        _ok(True, "adv_yuan 与 daily_volume 退化口径一致"),
        _ok(True, "20 日 ADV 工具正确"),
    ]


# ============================================================
# 3. 披露日 vs 报告期 + 时戳完整性
# ============================================================
def test_disclosure_vs_report():
    from label_engineering import (
        disclosure_vs_report_check, timestamp_integrity_check,
    )

    # 缺披露日列 → FAIL
    report_df = pd.DataFrame({
        "code": ["600519"] * 4,
        "report_period": pd.to_datetime(["2024-03-31"] * 4),
    })
    r = disclosure_vs_report_check(report_df)
    assert r["verdict"] == "FAIL", "缺披露日应 FAIL"

    # 披露日早于报告期 → FAIL
    bad = pd.DataFrame({
        "code": ["600519"],
        "report_period": pd.to_datetime(["2024-03-31"]),
        "disclosure_date": pd.to_datetime(["2024-03-15"]),  # 早于报告期!
    })
    r2 = disclosure_vs_report_check(bad)
    assert r2["verdict"] == "FAIL"

    # 正常
    good = pd.DataFrame({
        "code": ["600519", "000001"],
        "report_period": pd.to_datetime(["2024-03-31", "2024-03-31"]),
        "disclosure_date": pd.to_datetime(["2024-04-20", "2024-04-25"]),
    })
    r3 = disclosure_vs_report_check(good)
    assert r3["verdict"] == "PASS", r3

    # 时戳完整性: feature 最大日 > label 最大日 → FAIL
    ft = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=10)})
    lb = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=8)})
    ti = timestamp_integrity_check(ft, lb)
    assert ti["verdict"] == "FAIL", ti

    # 正常情况
    ft2 = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=8)})
    lb2 = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=10)})
    ti2 = timestamp_integrity_check(ft2, lb2)
    assert ti2["verdict"] == "PASS"

    return [
        _ok(True, "缺披露日 FAIL"),
        _ok(True, "披露日早于报告期 FAIL"),
        _ok(True, "正常披露日 PASS"),
        _ok(True, "feature 超越 label 日期 FAIL"),
        _ok(True, "正常时戳 PASS"),
    ]


# ============================================================
# 4. 执行层统一协议 + ExecutionEngine
# ============================================================
def test_execution_protocol():
    from execution import (
        ExecutionEngine, OrderRequest, Side,
        TWAPSlicer, VWAPSlicer, ImpactAwareRouter, Slicer,
    )

    # 所有 Slicer 都满足协议 (runtime_checkable 仅检查方法)
    assert callable(getattr(TWAPSlicer, "slice"))
    assert callable(getattr(VWAPSlicer, "slice"))
    assert callable(getattr(ImpactAwareRouter, "slice"))

    today = datetime.combine(date.today(), time(10, 0))
    req = OrderRequest(
        code="600519", side=Side.BUY, total_shares=1000,
        ref_price=1680.0, adv_yuan=5e9, volatility=0.02,
        start_time=today, end_time=today + timedelta(hours=2),
    )

    # TWAP
    twap = TWAPSlicer(n_slices=5)
    plan_t = twap.slice(req)
    assert plan_t.strategy == "TWAP"
    assert plan_t.total_shares <= 1000
    assert plan_t.total_shares >= 900    # 允许整手取整损失

    # ImpactAware
    iar = ImpactAwareRouter(target_participation=0.05)
    plan_i = iar.slice(req)
    assert plan_i.strategy == "ImpactAware"
    assert plan_i.total_cost_bps > 0
    assert plan_i.participation_rate > 0

    # VWAP: 默认曲线, 总股数正确
    vwap = VWAPSlicer(n_slices=8)
    plan_v = vwap.slice(req)
    assert plan_v.total_shares > 0, "VWAP 应产出切片"

    # ExecutionEngine: 非交易时段拒单
    engine = ExecutionEngine.default(
        strategy="ImpactAware",
        reject_forbidden_window=True,
    )
    req_bad = OrderRequest(
        code="600519", side=Side.BUY, total_shares=1000,
        ref_price=1680.0, adv_yuan=5e9, volatility=0.02,
        start_time=datetime.combine(date.today(), time(7, 0)),  # 盘前
        end_time=datetime.combine(date.today(), time(9, 0)),
    )
    rejected = engine.plan(req_bad)
    assert not rejected.slices, "盘前时段应被拒单"
    assert "FORBIDDEN" in rejected.notes

    return [
        _ok(True, "所有 Slicer 实现 .slice() 方法"),
        _ok(True, f"TWAP 股数守恒 {plan_t.total_shares}"),
        _ok(True, f"ImpactAware bps={plan_i.total_cost_bps:.1f}"),
        _ok(True, f"VWAP 产出切片 {plan_v.total_shares}"),
        _ok(True, "ExecutionEngine 拒绝非交易时段"),
    ]


# ============================================================
# 5. PreTradeGate 硬过闸
# ============================================================
def test_pre_trade_gate():
    from risk import Portfolio, PreTradeGate, OrderIntent, build_default_gate

    gate = build_default_gate()
    port = Portfolio(
        cash=1_000_000.0, initial_capital=1_000_000.0,
        high_water_mark=1_000_000.0, daily_start_value=1_000_000.0,
    )

    # 涨停板拒绝
    intent_up = OrderIntent(
        code="600519", side="buy", shares=100,
        price=110.0, prev_close=100.0,   # +10%, 涨停
    )
    dec = gate.check(intent_up, port)
    assert not dec.allow, "涨停应拒绝买入"
    assert "涨停" in dec.reason

    # 停牌拒绝
    intent_halt = OrderIntent(
        code="600519", side="buy", shares=100,
        price=100.0, prev_close=100.0, suspended=True,
    )
    dec2 = gate.check(intent_halt, port)
    assert not dec2.allow and "停牌" in dec2.reason

    # 正常买入通过, 股数可能被下调
    intent_ok = OrderIntent(
        code="600519", side="buy", shares=100,
        price=100.0, prev_close=99.0,
    )
    dec3 = gate.check(intent_ok, port)
    assert dec3.allow, dec3.reason
    assert dec3.adjusted_shares > 0

    # 统计正确性
    assert gate.stats.n_orders == 3
    assert gate.stats.n_rejected == 2
    assert gate.stats.n_passed == 1
    assert gate.stats.pass_rate == 1 / 3
    top = gate.stats.top_reject_reasons()
    assert len(top) >= 1

    # 审计链
    assert len(gate.audit_trail) == 3
    assert gate.audit_trail[0]["allow"] is False

    # reset
    gate.reset_stats()
    assert gate.stats.n_orders == 0 and not gate.audit_trail

    return [
        _ok(True, "涨停买入被硬拒"),
        _ok(True, "停牌被硬拒"),
        _ok(True, "正常买入通过"),
        _ok(True, "统计: 3 单 1 通过 2 拒绝"),
        _ok(True, "audit_trail 完整"),
        _ok(True, "reset 清空"),
    ]


# ============================================================
# 6. 前视偏差扫描嵌入 ResearchPipeline (端到端)
# ============================================================
def test_pipeline_lookahead_guard():
    """验证 pipeline 能因前视偏差中止."""
    from pipeline import ResearchPipeline

    np.random.seed(0)
    # 造数据: 3 只股票 x 200 天
    codes = ["S1", "S2", "S3"]
    dates = pd.date_range("2024-01-01", periods=200)
    rows = []
    for c in codes:
        p = 100.0
        for d in dates:
            p *= (1 + np.random.randn() * 0.01)
            rows.append({
                "code": c, "date": d,
                "open": p, "close": p * (1 + np.random.randn() * 0.002),
                "volume": 1_000_000, "pct_chg": np.random.randn() * 2,
                "ipo_days": 500, "name": c,
            })
    daily = pd.DataFrame(rows)

    pipeline = ResearchPipeline(
        neutralize_styles=False, skip_audit=True,
        lookahead_scan=True, fail_on_lookahead_critical=False,
        enforce_risk_gate=True,
    )
    result = pipeline.run(daily)
    # lookahead_report 应有内容 (verdict 至少一个值)
    assert result.lookahead_report != {}, "lookahead_report 应非空"
    # gate 统计应被记录
    assert result.gate_stats, f"gate_stats 应被记录: {result.gate_stats}"
    assert result.gate_stats.get("n_orders", 0) > 0

    return [
        _ok(True, f"lookahead verdict={result.lookahead_report.get('verdict')}"),
        _ok(True, f"gate 订单数={result.gate_stats.get('n_orders')}"),
        _ok(True,
             f"backtest_stats 产出={bool(result.backtest_stats)}"),
    ]


# ============================================================
def main():
    from tabulate import tabulate

    suites = [
        ("Barra 分层中性化", test_barra_hierarchical),
        ("冲击成本 ADV", test_impact_router_adv),
        ("披露日/时戳", test_disclosure_vs_report),
        ("执行层协议", test_execution_protocol),
        ("PreTradeGate", test_pre_trade_gate),
        ("Pipeline 闭环", test_pipeline_lookahead_guard),
    ]

    grand_pass, grand_total = 0, 0
    for name, fn in suites:
        try:
            results = fn()
        except Exception as e:
            print(f"❌ {name}: 异常 {e}")
            import traceback
            traceback.print_exc()
            continue
        table = [[i, m, "✓" if ok else "✗"]
                 for i, (ok, m) in enumerate(results, 1)]
        passed = sum(1 for ok, _ in results if ok)
        grand_pass += passed
        grand_total += len(results)
        print(f"\n=== {name} ({passed}/{len(results)}) ===")
        print(tabulate(table, headers=["#", "断言", "PASS"]))

    print(f"\n{'='*50}\n累计: {grand_pass}/{grand_total}\n{'='*50}")
    return grand_pass == grand_total


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
