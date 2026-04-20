"""用真实数据跑两个实验并产出对比报告.

实验 A (baseline): 3 个玩具因子 (momentum_5d + reversal_1d + vol_20d)
实验 B (升级版):  Alpha158-lite (30+ 个 qlib 风格因子) + Barra 残差化

输出:
    output/baseline_<ts>.html        实验 A 详情
    output/alpha158_barra_<ts>.html  实验 B 详情
    output/compare_<ts>.html         对比报告 (推荐直接看这个)
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from data_adapter.em_direct import bulk_fetch_daily, fetch_index_daily
from pipeline import (
    ResearchPipeline, build_compare_report, build_research_report_html,
)


DEFAULT_POOL = [
    "300750", "600519", "000858", "601318", "600036",
    "601398", "002594", "000333", "600900", "601888",
    "601012", "002415", "600030", "000001", "600276",
    "601166", "002475", "600309", "002352", "000651",
    "601899", "600028", "600050", "601988", "688981",
]


def main(
    n_stocks: int = 25, start: str = "20240101", end: str = "20260420",
):
    print(f"\n{'='*64}")
    print(f"  策略对比实验 - {time.strftime('%Y-%m-%d %H:%M')}")
    print(f"  股票池: {n_stocks} 只 · 时段: {start} → {end}")
    print(f"{'='*64}")

    codes = DEFAULT_POOL[:n_stocks]
    print(f"\n[1/4] 拉取日线 ({n_stocks} 只)...")
    t0 = time.time()
    cache_path = Path(__file__).resolve().parent.parent / "cache" / \
                 f"daily_{n_stocks}_{start}_{end}.pkl"
    cache_path.parent.mkdir(exist_ok=True)
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            daily_df = pickle.load(f)
        print(f"  命中缓存 ({time.time()-t0:.1f}s, {len(daily_df)} 行)")
    else:
        daily_df = bulk_fetch_daily(codes, start, end, sleep_ms=120)
        if daily_df.empty:
            print("❌ 拉数据失败")
            return
        with open(cache_path, "wb") as f:
            pickle.dump(daily_df, f)
        print(f"  耗时 {time.time()-t0:.1f}s, 写缓存 {cache_path.name}")

    # 指数
    idx_df = fetch_index_daily("000300", start, end)
    idx_df["date"] = pd.to_datetime(idx_df["date"])
    market_return = idx_df.set_index("date")["pct_chg"] / 100

    # 市值 (简化)
    market_cap = pd.Series({c: 3e10 for c in codes})
    industries = pd.Series({c: ("金融" if c.startswith(("601", "600036"))
                                  else "消费" if c.startswith(("600519", "000858"))
                                  else "科技") for c in codes})

    daily_df = daily_df.sort_values(["code", "date"]).reset_index(drop=True)
    ts = time.strftime("%Y%m%d_%H%M")
    out_dir = Path(__file__).resolve().parent.parent / "output"
    out_dir.mkdir(exist_ok=True)

    # ============ 实验 A: Baseline ============
    print(f"\n[2/4] 实验 A: Baseline (玩具因子 × 3)...")
    t0 = time.time()
    pipe_a = ResearchPipeline(
        neutralize_styles=False, skip_audit=False,
        lookahead_scan=True, fail_on_lookahead_critical=False,
        enforce_risk_gate=True,
        use_alpha158=False,
        neutralize_residual=False,
        compose_signal=False,        # baseline: 第一因子
        experiment_name="baseline",
    )
    result_a = pipe_a.run(daily_df=daily_df)
    print(f"  完成 {time.time()-t0:.1f}s  "
          f"Sharpe={result_a.backtest_stats.get('sharpe', 0):.2f}  "
          f"IC={result_a.ic_stats.get('ic_mean', 0):.4f}")
    html_a = out_dir / f"baseline_{ts}.html"
    build_research_report_html(result_a, out_path=html_a,
                                 project_name="Baseline (3 toy factors)")

    # ============ 实验 B: Alpha158-lite + Barra 残差化 ============
    print(f"\n[3/4] 实验 B: Alpha158-lite + Barra 残差化...")
    t0 = time.time()
    # Barra 残差化需要 ≥ 50 只股票才稳定; 小股票池下自动关闭
    use_residual = n_stocks >= 50
    pipe_b = ResearchPipeline(
        neutralize_styles=use_residual, neutralize_method="hierarchical",
        skip_audit=False,
        lookahead_scan=True, fail_on_lookahead_critical=False,
        enforce_risk_gate=True,
        use_alpha158=True,
        neutralize_residual=use_residual,
        compose_signal=True,         # 升级版: Rolling IC 加权合成
        experiment_name="alpha158_barra" if use_residual else "alpha158_ic",
    )
    exp_b_name = ("Alpha158 + Barra 残差" if use_residual
                   else "Alpha158 + IC 加权")
    result_b = pipe_b.run(
        daily_df=daily_df,
        market_cap=market_cap, market_return=market_return,
        industry_map=industries,
    )
    print(f"  完成 {time.time()-t0:.1f}s  "
          f"Sharpe={result_b.backtest_stats.get('sharpe', 0):.2f}  "
          f"IC={result_b.ic_stats.get('ic_mean', 0):.4f}")
    html_b = out_dir / f"alpha158_barra_{ts}.html"
    build_research_report_html(
        result_b, out_path=html_b, project_name=exp_b_name,
    )

    # ============ 对比 ============
    print(f"\n[4/4] 生成对比 HTML...")
    compare_path = out_dir / f"compare_{ts}.html"
    build_compare_report(
        result_a, result_b,
        name_a="Baseline (toy × 3)",
        name_b=exp_b_name,
        out_path=compare_path,
    )

    # 摘要
    def _stats(r):
        bt = r.backtest_stats
        return (f"Sharpe={bt.get('sharpe', 0):.2f}  "
                f"Annual={bt.get('annual_return', 0)*100:.1f}%  "
                f"MaxDD={bt.get('max_drawdown', 0)*100:.1f}%  "
                f"IC={r.ic_stats.get('ic_mean', 0):.4f}")

    print(f"\n{'='*64}")
    print(f"  对比结果")
    print(f"{'='*64}")
    print(f"  A (baseline):   {_stats(result_a)}")
    print(f"  B (alpha158+Barra): {_stats(result_b)}")
    print(f"\n  📄 A 详报: {html_a}")
    print(f"  📄 B 详报: {html_b}")
    print(f"  🆚 对比:   {compare_path}")
    print(f"  浏览器打开对比: open {compare_path}")

    return result_a, result_b


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--start", default="20240101")
    ap.add_argument("--end", default="20260420")
    args = ap.parse_args()
    main(n_stocks=args.n, start=args.start, end=args.end)
