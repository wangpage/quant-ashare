"""V4 benchmark - 用已有 200 只缓存跑 3 组对照实验, 验证 Sharpe 改造效果.

实验组:
    A. baseline:   toy 3 因子 + 等权 + 周频       (对齐 v1)
    B. alpha158:   Alpha158-lite + 等权 + 周频
    C. +barra:     Alpha158-lite + Barra 残差 + 等权 + 周频
    D. +rp-月频:   Alpha158 + Barra + 风险平价 + 月频 + buffer 0.3

预期: A < B < C < D, D 接近或超过 Sharpe 1.5+
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from data_adapter.em_direct import fetch_index_daily
from pipeline import ResearchPipeline

ROOT = Path(__file__).resolve().parent.parent


def _segment_stats(arr, tovs, cost_bps: float, freq: int) -> dict:
    """给收益序列 / 换手序列 计算扣成本 Sharpe / 年化 / 回撤."""
    import numpy as np
    if len(arr) == 0:
        return {"sharpe_net": float("nan"), "ann_ret_net": float("nan"),
                "max_dd_net": float("nan"), "n_periods": 0,
                "avg_turnover": float("nan")}
    arr = np.asarray(arr, dtype=float)
    tovs = np.asarray(tovs, dtype=float) if len(tovs) == len(arr) else np.zeros_like(arr)
    periods = max(1.0, 252.0 / freq)
    costs = tovs * cost_bps / 10000.0
    net = arr - costs
    ann_ret_net = float(net.mean() * periods)
    ann_vol_net = float(net.std() * np.sqrt(periods))
    sharpe_net = ann_ret_net / (ann_vol_net + 1e-9)
    nav = pd.Series(1 + net).cumprod()
    dd = (nav - nav.cummax()) / nav.cummax()
    return {
        "sharpe_net": sharpe_net,
        "ann_ret_net": ann_ret_net,
        "ann_vol_net": ann_vol_net,
        "max_dd_net": float(dd.min()),
        "n_periods": len(arr),
        "avg_turnover": float(tovs.mean()),
    }


def run_exp(label: str, daily, mkt_cap, mkt_ret, cost_bps: float = 10,
             oos_split: str = "2026-01-01", **kwargs) -> dict:
    """cost_bps: 单边换手成本 (bps). 实盘 A股 10~20: 佣金 2.5+印花税 10(卖)+滑点 5-10.

    oos_split: 以此日期切 IS / OOS. pipeline rolling IC window 是 60d,
               IS 里自然做完暖启动, OOS 区间不包含未来 leak.
    """
    t0 = time.time()
    pipe = ResearchPipeline(skip_audit=True, **kwargs)
    res = pipe.run(
        daily_df=daily, market_cap=mkt_cap, market_return=mkt_ret,
    )
    bt = res.backtest_stats
    ic = res.ic_stats
    freq = kwargs.get("rebalance_freq", 5)

    rets = res.stage_results.get("_bt_returns", [])
    tovs = res.stage_results.get("_bt_turnovers", [])
    dates = res.stage_results.get("_bt_dates", [])

    # 分三段: all / IS (< oos_split) / OOS (>= oos_split)
    all_s = _segment_stats(rets, tovs, cost_bps, freq)
    is_s = oos_s = {"sharpe_net": float("nan"), "ann_ret_net": float("nan"),
                     "max_dd_net": float("nan"), "n_periods": 0,
                     "avg_turnover": float("nan")}
    if len(dates) == len(rets) and dates:
        dts = pd.to_datetime(dates)
        cutoff = pd.Timestamp(oos_split)
        is_mask = dts < cutoff
        oos_mask = ~is_mask
        is_rets = [r for r, m in zip(rets, is_mask) if m]
        is_tovs = [t for t, m in zip(tovs, is_mask) if m]
        oos_rets = [r for r, m in zip(rets, oos_mask) if m]
        oos_tovs = [t for t, m in zip(tovs, oos_mask) if m]
        is_s = _segment_stats(is_rets, is_tovs, cost_bps, freq)
        oos_s = _segment_stats(oos_rets, oos_tovs, cost_bps, freq)

    out = {
        "label": label,
        "sharpe_gross": bt.get("sharpe", float("nan")),
        "sharpe_net_all": all_s["sharpe_net"],
        "sharpe_net_IS": is_s["sharpe_net"],
        "sharpe_net_OOS": oos_s["sharpe_net"],
        "ann_ret_IS": is_s["ann_ret_net"],
        "ann_ret_OOS": oos_s["ann_ret_net"],
        "dd_IS": is_s["max_dd_net"],
        "dd_OOS": oos_s["max_dd_net"],
        "n_IS": is_s["n_periods"],
        "n_OOS": oos_s["n_periods"],
        "tov_IS": is_s["avg_turnover"],
        "tov_OOS": oos_s["avg_turnover"],
        "ic_mean": ic.get("ic_mean", float("nan")),
        "ic_ir": ic.get("icir", float("nan")),
        "elapsed_s": round(time.time() - t0, 1),
    }
    return out


def main(n: int = 60, cost_bps: float = 20, oos_split: str = "2026-01-01"):
    print(f"\n{'='*64}\n  V4 基准对比 (n={n}, 成本={cost_bps}bps, "
          f"OOS>={oos_split})  {time.strftime('%H:%M')}\n{'='*64}")

    cache = ROOT / "cache" / "kline_20230101_20260420_n200.parquet"
    daily = pd.read_parquet(cache)
    # 稳定子集: 按 code 抽前 n 只
    first_codes = sorted(daily["code"].unique())[:n]
    daily = daily[daily["code"].isin(first_codes)].copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values(["code", "date"]).reset_index(drop=True)
    print(f"  池子 {daily['code'].nunique()}  行数 {len(daily)}  "
          f"期间 {daily['date'].min().date()} → {daily['date'].max().date()}")

    ik = ROOT / "cache" / "v4_idx_20230101_20260420.parquet"
    if ik.exists():
        idx = pd.read_parquet(ik)
    else:
        idx = fetch_index_daily("000300", "20230101", "20260420")
        idx.to_parquet(ik)
    idx["date"] = pd.to_datetime(idx["date"])
    mkt_ret = idx.set_index("date")["pct_chg"] / 100
    # 差异化市值代理: float_cap ≈ mean(amount) / mean(turnover_pct)
    # 因为 turnover = volume/float_share, amount = close*volume
    # 故 amount/turnover = close*float_share = float_cap
    cap_proxy = (
        daily.groupby("code").apply(
            lambda g: g["amount"].mean() / max(g["turnover"].mean(), 1e-3)
        ) if "turnover" in daily.columns else None
    )
    if cap_proxy is None or cap_proxy.isna().all():
        # 退化: 用 close×首日 volume×500 作为差异化代理 (秩差异是关键)
        cap_proxy = daily.groupby("code").apply(
            lambda g: float(g["close"].iloc[0]) *
                      float(g["volume"].iloc[0]) * 500
        )
    mkt_cap = cap_proxy.reindex(first_codes).fillna(cap_proxy.median())

    results = []

    # A. baseline
    results.append(run_exp(
        "A baseline (toy+eq+周)", daily, mkt_cap, mkt_ret,
        cost_bps=cost_bps, oos_split=oos_split,
        use_alpha158=False, neutralize_residual=False,
        portfolio_method="equal_weight", rebalance_freq=5, top_k=10,
    ))

    # B. Alpha158
    results.append(run_exp(
        "B +alpha158 (eq+周)", daily, mkt_cap, mkt_ret,
        cost_bps=cost_bps, oos_split=oos_split,
        use_alpha158=True, neutralize_residual=False,
        portfolio_method="equal_weight", rebalance_freq=5, top_k=10,
    ))

    # C. +Barra
    results.append(run_exp(
        "C +barra (eq+周)", daily, mkt_cap, mkt_ret,
        cost_bps=cost_bps, oos_split=oos_split,
        use_alpha158=True, neutralize_residual=True,
        portfolio_method="equal_weight", rebalance_freq=5, top_k=10,
    ))

    # D. 全套
    results.append(run_exp(
        "D +rp+月+buf (全套)", daily, mkt_cap, mkt_ret,
        cost_bps=cost_bps, oos_split=oos_split,
        use_alpha158=True, neutralize_residual=True,
        portfolio_method="risk_parity", rebalance_freq=20,
        top_k_ratio=0.2, top_k=max(5, int(n * 0.2)),
        turnover_buffer=0.3, cov_lookback=60,
    ))

    # E. IC-gated (激进, 不在 1+3+4 内, 作参考)
    results.append(run_exp(
        "E +IC-gate (激进参考)", daily, mkt_cap, mkt_ret,
        cost_bps=cost_bps, oos_split=oos_split,
        use_alpha158=True, neutralize_residual=True,
        portfolio_method="risk_parity", rebalance_freq=20,
        top_k_ratio=0.2, top_k=max(5, int(n * 0.2)),
        turnover_buffer=0.3, cov_lookback=60,
        ic_gate=True, ic_gate_window=20,
    ))

    # F. Alpha158 + Barra + 风险平价 + 周频 (拆开月频效应)
    results.append(run_exp(
        "F C+风险平价 (rp+周)", daily, mkt_cap, mkt_ret,
        cost_bps=cost_bps, oos_split=oos_split,
        use_alpha158=True, neutralize_residual=True,
        portfolio_method="risk_parity", rebalance_freq=5,
        top_k=10, turnover_buffer=0.3, cov_lookback=60,
    ))

    # G. Alpha158 + Barra + inverse_vol + 周频 (risk_parity 的轻量版)
    results.append(run_exp(
        "G C+逆波动率 (iv+周)", daily, mkt_cap, mkt_ret,
        cost_bps=cost_bps, oos_split=oos_split,
        use_alpha158=True, neutralize_residual=True,
        portfolio_method="inverse_vol", rebalance_freq=5,
        top_k=10, turnover_buffer=0.3, cov_lookback=60,
    ))

    # H. Alpha158 + Barra + 等权 + 周频 + buffer 0.5 (降换手主攻)
    results.append(run_exp(
        "H C+大buffer (eq+周+buf0.5)", daily, mkt_cap, mkt_ret,
        cost_bps=cost_bps, oos_split=oos_split,
        use_alpha158=True, neutralize_residual=True,
        portfolio_method="equal_weight", rebalance_freq=5,
        top_k=10, turnover_buffer=0.5,
    ))

    # ========== 顶级量化 "三连击" — 推 Sharpe 过 2.0 ==========
    # I. H + 信号 EMA(3) 平滑 (降噪 → 降换手 → 降成本)
    results.append(run_exp(
        "I H+EMA(3) (信号平滑)", daily, mkt_cap, mkt_ret,
        cost_bps=cost_bps, oos_split=oos_split,
        use_alpha158=True, neutralize_residual=True,
        portfolio_method="equal_weight", rebalance_freq=5,
        top_k=10, turnover_buffer=0.5,
        signal_ema_span=3,
    ))

    # J. H + vol target 15% (压波动, 核心杀招)
    results.append(run_exp(
        "J H+VolTgt15% (波动目标)", daily, mkt_cap, mkt_ret,
        cost_bps=cost_bps, oos_split=oos_split,
        use_alpha158=True, neutralize_residual=True,
        portfolio_method="equal_weight", rebalance_freq=5,
        top_k=10, turnover_buffer=0.5,
        vol_target=0.15, vol_target_window=20,
    ))

    # K. H + 风险调整 top-K (signal/vol 排序)
    results.append(run_exp(
        "K H+RiskAdj (signal/vol)", daily, mkt_cap, mkt_ret,
        cost_bps=cost_bps, oos_split=oos_split,
        use_alpha158=True, neutralize_residual=True,
        portfolio_method="equal_weight", rebalance_freq=5,
        top_k=10, turnover_buffer=0.5,
        signal_risk_adjust=True,
    ))

    # L. 三连击全开 (H + EMA + VolTgt + RiskAdj)
    results.append(run_exp(
        "L 三连击全开 🚀", daily, mkt_cap, mkt_ret,
        cost_bps=cost_bps, oos_split=oos_split,
        use_alpha158=True, neutralize_residual=True,
        portfolio_method="equal_weight", rebalance_freq=5,
        top_k=10, turnover_buffer=0.5,
        signal_ema_span=3, vol_target=0.15, vol_target_window=20,
        signal_risk_adjust=True,
    ))

    # M. 三连击 + 风险平价 (组合优化 + 三连击 会不会相加?)
    results.append(run_exp(
        "M 三连击+RP (rp+全开)", daily, mkt_cap, mkt_ret,
        cost_bps=cost_bps, oos_split=oos_split,
        use_alpha158=True, neutralize_residual=True,
        portfolio_method="risk_parity", rebalance_freq=5,
        top_k=10, turnover_buffer=0.5, cov_lookback=60,
        signal_ema_span=3, vol_target=0.15,
        signal_risk_adjust=True,
    ))

    # N. EMA(3) + RiskAdj 双杀 (丢掉拖后腿的 VolTgt)
    results.append(run_exp(
        "N H+EMA+RiskAdj (双杀)", daily, mkt_cap, mkt_ret,
        cost_bps=cost_bps, oos_split=oos_split,
        use_alpha158=True, neutralize_residual=True,
        portfolio_method="equal_weight", rebalance_freq=5,
        top_k=10, turnover_buffer=0.5,
        signal_ema_span=3, signal_risk_adjust=True,
    ))

    # O. EMA(5) 更强平滑 - 极限测试
    results.append(run_exp(
        "O H+EMA(5) (强平滑)", daily, mkt_cap, mkt_ret,
        cost_bps=cost_bps, oos_split=oos_split,
        use_alpha158=True, neutralize_residual=True,
        portfolio_method="equal_weight", rebalance_freq=5,
        top_k=10, turnover_buffer=0.5,
        signal_ema_span=5,
    ))

    # P. EMA(3) + RiskAdj + 风险平价 (最终候选)
    results.append(run_exp(
        "P Final (EMA+RA+RP)", daily, mkt_cap, mkt_ret,
        cost_bps=cost_bps, oos_split=oos_split,
        use_alpha158=True, neutralize_residual=True,
        portfolio_method="risk_parity", rebalance_freq=5,
        top_k=10, turnover_buffer=0.5, cov_lookback=60,
        signal_ema_span=3, signal_risk_adjust=True,
    ))

    # Q. EMA(3) + 弱 VolTgt (25% target, 不压太狠)
    results.append(run_exp(
        "Q H+EMA+WeakVolTgt25%", daily, mkt_cap, mkt_ret,
        cost_bps=cost_bps, oos_split=oos_split,
        use_alpha158=True, neutralize_residual=True,
        portfolio_method="equal_weight", rebalance_freq=5,
        top_k=10, turnover_buffer=0.5,
        signal_ema_span=3, vol_target=0.25, vol_target_window=40,
    ))

    df = pd.DataFrame(results)
    # 打印对比
    print("\n" + "="*120)
    cols = ["label", "sharpe_gross", "sharpe_net_all",
             "sharpe_net_IS", "sharpe_net_OOS",
             "ann_ret_IS", "ann_ret_OOS", "dd_OOS",
             "n_IS", "n_OOS", "tov_IS", "tov_OOS", "ic_ir", "elapsed_s"]
    df_print = df[cols].copy()
    for c in ["sharpe_gross", "sharpe_net_all", "sharpe_net_IS",
               "sharpe_net_OOS", "ic_ir"]:
        df_print[c] = df_print[c].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
    for c in ["ann_ret_IS", "ann_ret_OOS", "dd_OOS", "tov_IS", "tov_OOS"]:
        df_print[c] = df_print[c].apply(lambda x: f"{x:+.2%}" if pd.notna(x) else "—")
    print(df_print.to_string(index=False))
    print("="*120)

    # 过拟合诊断: OOS Sharpe / IS Sharpe, <0.5 = 严重过拟合, >0.7 = 稳健
    print("\n过拟合诊断 (OOS/IS Sharpe 比率):")
    for r in results:
        isv = r["sharpe_net_IS"]; oosv = r["sharpe_net_OOS"]
        if pd.notna(isv) and pd.notna(oosv) and abs(isv) > 0.1:
            ratio = oosv / isv
            flag = ("🟢 稳健" if ratio > 0.7 else
                    "🟡 衰减" if ratio > 0.3 else
                    "🔴 过拟合" if ratio > 0 else
                    "⚫ 反转")
            print(f"  {r['label']:30s}  IS={isv:.2f}  OOS={oosv:.2f}  "
                  f"ratio={ratio:+.2f}  {flag}")

    ts = time.strftime("%Y%m%d_%H%M")
    out = ROOT / "output" / f"bench_v4_oos_{ts}.md"
    out.parent.mkdir(exist_ok=True)
    md = [
        f"# V4 IS/OOS 对比 (n={n}, cost={cost_bps}bps, {time.strftime('%Y-%m-%d %H:%M')})",
        "",
        f"股票池: {n} 只 (首字母序)",
        f"期间: {daily['date'].min().date()} → {daily['date'].max().date()}",
        f"OOS 切分点: {oos_split} (即 2026 Q1 样本外)",
        f"成本: {cost_bps}bps / 单边换手 (A股实盘上限估计)",
        "",
        "## 核心指标 (Sharpe 扣成本)",
        "",
        "| 实验 | Sharpe(毛) | Sharpe全期 | **Sharpe IS** | **Sharpe OOS** | 年化 IS | 年化 OOS | 回撤 OOS | n_IS | n_OOS |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for r in results:
        def _f(v, fmt="{:.2f}"):
            return fmt.format(v) if pd.notna(v) else "—"
        md.append(
            f"| {r['label']} | {_f(r['sharpe_gross'])} | "
            f"{_f(r['sharpe_net_all'])} | "
            f"**{_f(r['sharpe_net_IS'])}** | "
            f"**{_f(r['sharpe_net_OOS'])}** | "
            f"{_f(r['ann_ret_IS'], '{:+.2%}')} | "
            f"{_f(r['ann_ret_OOS'], '{:+.2%}')} | "
            f"{_f(r['dd_OOS'], '{:+.2%}')} | "
            f"{r['n_IS']} | {r['n_OOS']} |"
        )

    md.append("")
    md.append("## 过拟合诊断 (OOS/IS Sharpe 比率)")
    md.append("")
    md.append("| 实验 | IS | OOS | ratio | 判断 |")
    md.append("|---|:---:|:---:|:---:|:---:|")
    for r in results:
        isv = r["sharpe_net_IS"]; oosv = r["sharpe_net_OOS"]
        if pd.notna(isv) and pd.notna(oosv) and abs(isv) > 0.1:
            ratio = oosv / isv
            flag = ("🟢 稳健" if ratio > 0.7 else
                    "🟡 衰减" if ratio > 0.3 else
                    "🔴 过拟合" if ratio > 0 else
                    "⚫ 反转")
            md.append(f"| {r['label']} | {isv:.2f} | {oosv:.2f} | "
                      f"{ratio:+.2f} | {flag} |")
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"\n📝 报告: {out}")
    return df


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--cost", type=float, default=20,
                     help="单边换手成本 bps (默认 20 = 保守实盘)")
    ap.add_argument("--oos", default="2026-01-01",
                     help="OOS 切分日期")
    args = ap.parse_args()
    main(args.n, cost_bps=args.cost, oos_split=args.oos)
