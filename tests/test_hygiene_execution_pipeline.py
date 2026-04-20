"""验证 data_hygiene / execution / pipeline 三个模块."""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from tabulate import tabulate


def _a(cond, msg): return bool(cond), msg


# ==================== 1. data_hygiene ====================
def test_hygiene():
    from data_hygiene import (
        detect_survivorship_bias, delisted_stock_checker,
        scan_lookahead_bias, time_index_integrity_check,
        detect_price_jumps, verify_adjustment_factor,
        find_suspension_days, gap_aware_fill,
        clock_skew_detector, align_to_exchange_time,
        DataHealthChecker,
    )
    from data_hygiene.timezone_sync import (
        unify_tz_to_shanghai, detect_dst_artifacts,
    )
    from data_hygiene.survivorship import detect_point_in_time_issues
    from data_hygiene.lookahead import label_leakage_test
    from data_hygiene.adjustment import cross_validate_with_raw, fix_missing_adjustment
    from data_hygiene.gaps import suspension_recovery_mask, cross_sectional_fill

    # 造数据: 包含停牌, 跳点
    n = 200
    codes = ["300750", "600519", "000001"]
    dates = pd.date_range("2024-01-01", periods=n)
    rows = []
    for c in codes:
        for d in dates:
            rows.append({"code": c, "date": d,
                          "close": 100 + np.random.randn(),
                          "open": 100 + np.random.randn(),
                          "volume": 1_000_000,
                          "factor": 1.0})
    # 插入停牌
    rows[100]["volume"] = 0
    rows[101]["volume"] = 0
    # 插入跳点
    rows[150]["close"] = 150
    df = pd.DataFrame(rows)

    # 幸存偏差
    sb = detect_survivorship_bias(df, "2024-01-01", "2024-07-01")
    # 退市检查
    del_check = delisted_stock_checker(["300750", "600519", "000001"])

    # 时间完整性
    ti = time_index_integrity_check(df)
    # 价格跳点
    jumps = detect_price_jumps(df)
    # 复权校验
    adj = verify_adjustment_factor(df)
    # 停牌区间
    susp = find_suspension_days(df)
    # 停牌感知填充
    filled = gap_aware_fill(df)
    # 恢复后 mask
    recov = suspension_recovery_mask(df)

    # 时钟偏差
    ex_ts = np.arange(1000) * 1000
    loc_ts = ex_ts + np.random.randint(-10, 100, 1000)
    skew = clock_skew_detector(ex_ts, loc_ts)
    # 时区
    df_tz = unify_tz_to_shanghai(df.head(5))

    # 前视偏差扫描
    features = pd.DataFrame({
        "ok": np.random.randn(n),
        "leaky": np.roll(np.random.randn(n), -5),   # 平滑, 低 corr
    })
    label = pd.Series(np.random.randn(n))
    la = scan_lookahead_bias(features, label)

    # 复权数据修正
    events = [("300750", "2024-03-01", 0.5)]
    fixed = fix_missing_adjustment(df, events)

    # 一键体检
    checker = DataHealthChecker()
    report = checker.audit_full(df)

    return [
        _a("survivorship_risk" in sb, "幸存偏差检测"),
        _a("coverage" in del_check, "退市覆盖率"),
        _a(ti.get("verdict") in ("PASS", "FAIL"), f"时间完整性 {ti.get('verdict')}"),
        _a(len(jumps) >= 1, f"检测到跳点 {len(jumps)}"),
        _a(adj.get("verdict") in ("PASS", "FAIL"), f"复权校验 {adj.get('verdict')}"),
        _a(len(susp) >= 1, f"停牌区间 {len(susp)}"),
        _a(len(filled) == len(df), "gap_aware_fill 保持行数"),
        _a(recov.sum() <= len(df), "recovery mask"),
        _a("mean_ms" in skew, f"时钟偏差 {skew['mean_ms']:.1f}ms"),
        _a("verdict" in la, f"前视偏差 {la.get('verdict')}"),
        _a("close_adj" in fixed.columns, "除权修正列"),
        _a(hasattr(report, "summary"), "体检报告"),
    ]


# ==================== 2. execution ====================
def test_execution():
    from execution import (
        OptimalTradingWindow, is_tradeable_now, avoid_auction_window,
        TWAPSlicer, VWAPSlicer, POVSlicer,
        ImpactAwareRouter, split_order_by_participation,
        BacktestExecutionSim,
    )
    from execution.time_windows import WindowQuality

    # 时段判定
    opt = OptimalTradingWindow()
    q_open = opt.classify(datetime(2024, 4, 22, 9, 32))   # 开盘 2 分钟
    q_mid = opt.classify(datetime(2024, 4, 22, 10, 30))   # 上午正常
    q_close = opt.classify(datetime(2024, 4, 22, 14, 55))  # 尾盘
    q_lunch = opt.classify(datetime(2024, 4, 22, 12, 0))  # 午休

    # 成本乘数
    mult_open = opt.cost_multiplier(datetime(2024, 4, 22, 9, 32))
    mult_mid = opt.cost_multiplier(datetime(2024, 4, 22, 10, 30))

    # TWAP
    twap = TWAPSlicer()
    twap_slices = twap.slice(
        total_shares=100_000,
        start=datetime(2024, 4, 22, 9, 30),
        end=datetime(2024, 4, 22, 15, 0),
        n_slices=10,
    )

    # VWAP
    vwap = VWAPSlicer()
    vwap_slices = vwap.slice(
        total_shares=100_000,
        start=datetime(2024, 4, 22, 9, 30),
        end=datetime(2024, 4, 22, 15, 0),
        n_slices=20,
    )

    # POV
    pov = POVSlicer(participation_rate=0.10)
    dur = pov.estimate_duration_minutes(100_000, 10_000)
    next_sz = pov.next_slice_size(50_000)

    # 冲击路由
    router = ImpactAwareRouter(urgency=0.5)
    plan = router.plan_order(
        total_shares=100_000, price=50.0,
        daily_volume=10_000_000, volatility=0.025,
    )

    # 按参与率拆
    splits = split_order_by_participation(
        100_000,
        market_volume_forecast=[500_000, 800_000, 600_000, 400_000],
        target_participation=0.05,
    )

    # 执行仿真
    sim = BacktestExecutionSim()
    res_buy = sim.execute(
        action="buy", ref_price=50.0, shares=10_000,
        daily_volume=5_000_000, volatility=0.02,
        trade_time=datetime(2024, 4, 22, 10, 30),
    )
    res_buy_avoid = sim.execute(
        action="buy", ref_price=50.0, shares=10_000,
        daily_volume=5_000_000, volatility=0.02,
        trade_time=datetime(2024, 4, 22, 9, 32),   # AVOID 时段
    )
    res_lim_up = sim.execute(
        action="buy", ref_price=50.0, shares=10_000,
        daily_volume=5_000_000,
        is_limit_up=True,
    )

    return [
        _a(q_open == WindowQuality.AVOID, f"开盘 AVOID 实际 {q_open.value}"),
        _a(q_mid == WindowQuality.OPTIMAL, f"上午 OPTIMAL 实际 {q_mid.value}"),
        _a(q_close == WindowQuality.AVOID, f"尾盘 AVOID 实际 {q_close.value}"),
        _a(q_lunch == WindowQuality.FORBIDDEN, f"午休禁止 {q_lunch.value}"),
        _a(mult_open > mult_mid, f"开盘成本高于中段 {mult_open} > {mult_mid}"),
        _a(len(twap_slices) >= 8, f"TWAP 切片 {len(twap_slices)}"),
        _a(sum(s.target_shares for s in twap_slices) <= 100_000, "TWAP 总量"),
        _a(len(vwap_slices) >= 10, f"VWAP 切片 {len(vwap_slices)}"),
        _a(dur > 0, f"POV 预估 {dur:.1f} min"),
        _a(next_sz == 5_000, f"POV 下笔 {next_sz}"),
        _a(plan.expected_total_cost_bps > 0, f"冲击路由 {plan.expected_total_cost_bps:.1f} bps"),
        _a(sum(splits) <= 100_000, "参与率拆单总量"),
        _a(res_buy.cost_bps > 0, f"正常执行 cost {res_buy.cost_bps:.2f} bps"),
        _a(res_buy_avoid.cost_bps > res_buy.cost_bps,
           f"AVOID 时段成本更高 {res_buy_avoid.cost_bps:.1f} > {res_buy.cost_bps:.1f}"),
        _a(res_lim_up.filled_shares == 0, "涨停板未成交"),
    ]


# ==================== 3. pipeline ====================
def test_pipeline():
    from pipeline import (
        ResearchPipeline, DailyTradingPipeline,
        build_daily_report, build_research_report,
    )

    n = 250
    codes = ["300750", "600519", "000001", "000002", "600036",
             "601398", "600030", "300760", "002594", "002415"]
    dates = pd.date_range("2023-01-01", periods=n)
    rng = np.random.default_rng(42)
    rows = []
    for c in codes:
        price = 100.0
        for d in dates:
            price *= 1 + rng.normal(0.0005, 0.015)
            rows.append({
                "code": c, "date": d, "open": price * (1 + rng.normal(0, 0.005)),
                "close": price, "volume": int(rng.uniform(5e5, 5e6)),
                "pct_chg": rng.normal(0.05, 1.5),
                "turnover": rng.uniform(0.5, 3.0),
            })
    daily_df = pd.DataFrame(rows)

    # 研究 pipeline
    pipe = ResearchPipeline(skip_audit=True)
    result = pipe.run(daily_df=daily_df)
    research_report = build_research_report(result)

    # 实盘 pipeline
    index_df = daily_df[daily_df["code"] == "300750"].copy()
    index_df["high"] = index_df["close"] * 1.01
    index_df["low"] = index_df["close"] * 0.99
    stocks_today = daily_df[daily_df["date"] == daily_df["date"].max()]

    scored = {c: rng.random() for c in codes}

    daily_pipe = DailyTradingPipeline(top_k=5, use_llm=False)
    decision = daily_pipe.run(
        index_df=index_df.reset_index(drop=True),
        stocks_daily=stocks_today.reset_index(drop=True),
        candidates_scored=scored,
        total_capital=1_000_000,
    )
    daily_report = build_daily_report(decision)

    # 记录模拟交易反思
    trades = [{
        "code": "300750", "entry_date": "2024-04-01", "exit_date": "2024-04-05",
        "entry_price": 250, "exit_price": 265, "shares": 100,
        "holding_days": 4, "pnl_pct": 0.06,
        "entry_reasoning": "技术突破", "exit_trigger": "止盈",
        "regime": "bull_trending",
    }]
    saved = daily_pipe.record_trade_outcomes(trades)

    return [
        _a(hasattr(result, "ic_stats"), "研究结果有 ic_stats"),
        _a("backtest_stats" in result.__dict__, "研究结果有 backtest_stats"),
        _a(len(research_report) > 100, f"研究报告 {len(research_report)} 字符"),
        _a(decision.regime != "unknown" or decision.candidates, "决策产出"),
        _a(len(decision.candidates) <= 5, f"top_k 生效 {len(decision.candidates)}"),
        _a(len(daily_report) > 100, f"日报 {len(daily_report)} 字符"),
        _a(saved >= 1, f"反思保存 {saved}"),
        _a("research" in research_report.lower() or "研究" in research_report, "研究报告内容"),
    ]


def main():
    all_rows = []
    total_pass = total = 0
    for name, fn in [
        ("data_hygiene", test_hygiene),
        ("execution", test_execution),
        ("pipeline", test_pipeline),
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

    print("\n==== Hygiene + Execution + Pipeline ====")
    print(tabulate(all_rows, headers=["模块", "用例", "结果"]))
    print(f"\n通过率: {total_pass}/{total}")
    sys.exit(0 if total_pass == total else 1)


if __name__ == "__main__":
    main()
