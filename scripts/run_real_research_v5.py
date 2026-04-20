"""V4 - 去幻觉 + 补关键缺口. 三刀定生死.

对应用户专业 review:
    "因子多 ≠ Sharpe 高, 反而更容易归零"
    "你缺的不是因子, 是 3 个东西: Label 设计 / 执行建模 / 因子去共线"

三刀:
  刀 1 (最高 ROI): IC 聚类去共线
      68 因子 → ~20 核心. 每簇只留 IR 最高的.
      砍 Alpha101-style 公开量价因子, 保留行为类 + 独特信号.

  刀 2: 条件 Label (非线性)
      传统: close[t+H]/open[t+1]-1 端点收益, 被噪音主导
      V4:   max_return × (1 - alpha·max_drawdown)
            奖励"先涨后跌"可套利形态 (散户恐慌卖 → 在高点出货)

  刀 3: 按成交量滑点 (执行建模)
      Almgren-Chriss sqrt 律: impact ∝ sqrt(size / ADV)
      替代固定 10bps, 让回测数字现实化
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from data_adapter.em_direct import bulk_fetch_daily
from data_adapter.sina_universe import fetch_all_ashare, filter_midcap_universe
from data_adapter.lhb import (
    fetch_lhb_range, build_lhb_features,
    LHB_FACTOR_NAMES, LHB_B2_FACTOR_NAMES,
)
from data_adapter.insider import (
    fetch_insider_range, build_insider_features, INSIDER_FACTOR_NAMES,
)
from factors.alpha_pandas import compute_pandas_alpha
from factors.alpha_reversal import compute_advanced_alpha
from factors.alpha_limit import compute_limit_alpha, LIMIT_FACTOR_NAMES

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"


# ========== 刀 2 (修正版): end_return + 大回撤惩罚 ==========
def make_conditional_label(daily_df: pd.DataFrame,
                            horizon: int = 20,
                            dd_clip: float = 0.25) -> pd.Series:
    """Label = end_return - max(0, |max_dd| - dd_clip) × 2

    逻辑:
        - 主体是 end_return (持有到期能真实拿到的)
        - 只在 **大回撤** 样本 (max_dd < -25%) 上扣分, 惩罚"坑"股票
        - 避免 max_ret 的陷阱 (模型学不到择时)
    """
    df = daily_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["code", "date"]).reset_index(drop=True)

    labels = []
    for code, g in df.groupby("code"):
        g = g.sort_values("date").copy()
        highs, lows = g["high"].values, g["low"].values
        closes = g["close"].values
        opens = g["open"].shift(-1).values
        n = len(g)
        end_ret = np.full(n, np.nan)
        max_dd = np.full(n, np.nan)
        for i in range(n - horizon - 1):
            bp = opens[i]
            if not (bp and bp > 0):
                continue
            w_l = lows[i + 1 : i + 1 + horizon]
            ep = closes[i + horizon]   # 期末 close
            end_ret[i] = ep / bp - 1
            max_dd[i]  = w_l.min() / bp - 1
        g["end_ret"] = end_ret
        g["max_dd"]  = max_dd
        labels.append(g[["date", "code", "end_ret", "max_dd"]])

    merged = pd.concat(labels, ignore_index=True)
    # 只惩罚超过 dd_clip 的部分 (大坑)
    penalty = np.maximum(0.0, -merged["max_dd"] - dd_clip) * 2
    merged["label_raw"] = merged["end_ret"] - penalty
    merged["label"] = merged.groupby("date")["label_raw"].transform(
        lambda s: s.clip(s.quantile(0.01), s.quantile(0.99))
    )
    return merged.set_index(["date", "code"])["label"]


# ========== 刀 1: IC 聚类去共线 ==========
def ic_cluster_select(feat_z, label, corr_threshold=0.6, min_ic=0.005):
    """按 IC 时序相关性聚类, 每簇留 IR 最高的."""
    common = feat_z.index.intersection(label.index)
    feat = feat_z.loc[common]
    y = label.loc[common]
    mask = y.notna()
    feat, y = feat[mask], y[mask]
    print(f"  样本 {len(feat)}, 特征 {feat.shape[1]}")

    # 预先把 feat / y 放到日期-code 的扁平索引上方便取截面
    ic_rows = []
    # 把 feat 的 (date, code) index 拆开按 date 分组
    for dt, grp in feat.groupby(level="date"):
        # grp 的 index 仍是 multi, 保留 code 维度
        grp_c = grp.reset_index(level="date", drop=True)  # index 变成 code
        try:
            y_d = y.loc[dt]  # Series index=code
        except KeyError:
            continue
        if isinstance(y_d, float):
            continue
        if len(grp_c) < 20:
            continue
        ic_row = {}
        for col in feat.columns:
            x = grp_c[col]
            idx = x.index.intersection(y_d.index)
            if len(idx) < 20:
                ic_row[col] = np.nan
                continue
            c = x.loc[idx].corr(y_d.loc[idx], method="spearman")
            ic_row[col] = c
        ic_rows.append((dt, ic_row))

    ic_df = pd.DataFrame([r for _, r in ic_rows],
                          index=[d for d, _ in ic_rows])
    ic_mean = ic_df.mean()
    ir = ic_mean / (ic_df.std() + 1e-9)

    print(f"\n  因子 IC 概览 (按 |IR| 排序, top 15):")
    ir_abs = ir.abs().sort_values(ascending=False)
    for name in ir_abs.head(15).index:
        print(f"    {name:25s} IC={ic_mean[name]:+.4f}  IR={ir[name]:+.2f}")

    keep0 = ic_mean[ic_mean.abs() >= min_ic].index.tolist()
    print(f"\n  IC 显著 (|IC|>={min_ic}): {len(keep0)}/{feat.shape[1]}")

    ic_corr = ic_df[keep0].corr().abs()
    selected = []
    ranked = ir_abs[keep0].sort_values(ascending=False).index.tolist()
    for name in ranked:
        too_similar = any(
            ic_corr.loc[name, s] > corr_threshold for s in selected
        )
        if not too_similar:
            selected.append(name)

    print(f"  聚类后保留 (corr<{corr_threshold}): {len(selected)}")
    for s in selected:
        tag = ""
        if s.startswith("LHB_"):
            tag = " [行为]"
        elif s in ("MOM12_1", "MOM6_1", "AMIHUD_20", "AMIHUD_60"):
            tag = " [经典异常]"
        print(f"    ✓ {s:25s} IC={ic_mean[s]:+.4f}  IR={ir[s]:+.2f}{tag}")
    return selected


# ========== 刀 3: 动态滑点 ==========
def impact_bps(trade_size_yuan: float, daily_amount_yuan: float,
                c: float = 5.0) -> float:
    """Almgren-Chriss sqrt 律. 基础 5 bps (可拆单分散 impact).
    size/ADV=1%→5bps, 10%→16bps, 100%→50bps.
    """
    if daily_amount_yuan <= 0 or trade_size_yuan <= 0:
        return 10.0
    pct = trade_size_yuan / daily_amount_yuan
    return float(min(c * np.sqrt(pct * 100) + 3.0, 80.0))  # +3 bps 佣金


# ========== 训练 ==========
def _fit_model(X, y):
    try:
        import lightgbm as lgb
        m = lgb.LGBMRegressor(
            n_estimators=400, learning_rate=0.03,
            num_leaves=24, max_depth=5, min_child_samples=150,
            subsample=0.7, colsample_bytree=0.6,
            reg_alpha=0.3, reg_lambda=0.3,
            random_state=42, verbose=-1,
        )
        m.fit(X.values, y.values)
        return m
    except (OSError, ImportError):
        from sklearn.ensemble import HistGradientBoostingRegressor
        m = HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.03,
            max_leaf_nodes=24, max_depth=5, min_samples_leaf=150,
            l2_regularization=0.3, random_state=42,
        )
        m.fit(X.values, y.values)
        return m


def rolling_predict(X, y, train_days=252, step_days=21):
    dates = X.index.get_level_values("date").unique().sort_values()
    if len(dates) < train_days + step_days:
        split = int(len(dates) * 0.6)
        tr = X.loc[dates[0]:dates[split - 1]]
        ty = y.loc[dates[0]:dates[split - 1]]
        mask = ty.notna() & tr.notna().all(axis=1)
        m = _fit_model(tr[mask], ty[mask])
        te = X.loc[dates[split]:dates[-1]]
        return pd.Series(m.predict(te.values), index=te.index)

    preds = []
    i = train_days
    while i < len(dates):
        j = min(i + step_days, len(dates))
        tr_s, tr_e = dates[i - train_days], dates[i - 1]
        te_s, te_e = dates[i], dates[j - 1]
        try:
            X_tr = X.loc[tr_s:tr_e]
            y_tr = y.loc[tr_s:tr_e]
            mask = y_tr.notna() & X_tr.notna().all(axis=1)
            X_tr, y_tr = X_tr[mask], y_tr[mask]
            if len(X_tr) < 1000:
                i = j; continue
            m = _fit_model(X_tr, y_tr)
            X_te = X.loc[te_s:te_e]
            preds.append(pd.Series(m.predict(X_te.values), index=X_te.index))
            print(f"  rolling tr[{tr_s.date()}→{tr_e.date()}] te[{te_s.date()}→{te_e.date()}] n={len(X_tr)}")
        except Exception as e:
            print(f"  err {e}")
        i = j
    return pd.concat(preds).sort_index() if preds else pd.Series(dtype=float)


# ========== 回测 ==========
def backtest_v4(pred, daily_df, top_ratio=0.05, rebalance_days=20,
                capital_yuan=10_000_000, limit_pct=0.097,
                vol_target=0.20):
    df = daily_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["code", "date"])
    df["ret1"] = df.groupby("code")["close"].pct_change()
    # 补 amount: sina fallback 返回 0, 用 volume × close 近似
    if "amount" not in df.columns or (df["amount"] == 0).all():
        df["amount"] = df["volume"] * df["close"]
    else:
        zero_mask = df["amount"] == 0
        df.loc[zero_mask, "amount"] = df.loc[zero_mask, "volume"] * df.loc[zero_mask, "close"]
    df["is_limit_up"]    = (df["ret1"] > limit_pct).astype(int)
    df["is_limit_down"]  = (df["ret1"] < -limit_pct).astype(int)
    df["is_one_word_up"] = ((df["open"] == df["high"]) &
                             (df["high"] == df["low"]) &
                             (df["ret1"] > limit_pct)).astype(int)
    df["is_halt"] = (df["volume"] == 0).astype(int)
    df_idx = df.set_index(["date", "code"])

    dates = pred.index.get_level_values("date").unique().sort_values()
    rebalance_dates = list(dates[::rebalance_days])

    rets_arr, bench_arr, turnovers, impact_bps_list = [], [], [], []
    n_blocked_buys = n_blocked_sells = 0
    prev_holdings: set = set()

    for dt in rebalance_dates:
        scores = pred.xs(dt, level="date").dropna()
        if len(scores) < 20:
            continue
        k = max(10, int(len(scores) * top_ratio))
        candidates = scores.nlargest(int(k * 1.5)).index.tolist()

        fut = [d for d in dates if d > dt]
        if len(fut) < rebalance_days:
            continue
        buy_date, sell_date = fut[0], fut[rebalance_days - 1]

        admitted = []
        for code in candidates:
            try:
                row = df_idx.loc[(buy_date, code)]
                if row["is_one_word_up"] or row["is_halt"]:
                    n_blocked_buys += 1; continue
                admitted.append(code)
            except KeyError:
                continue
            if len(admitted) >= k:
                break
        if not admitted:
            continue
        top = admitted
        per_pos = capital_yuan / len(top)

        rets, bps_list = [], []
        for code in top:
            try:
                buy_p = df_idx.loc[(buy_date, code), "open"]
                buy_amt = df_idx.loc[(buy_date, code), "amount"]
                sell_row = df_idx.loc[(sell_date, code)]
                sell_p = sell_row["close"]
                sell_amt = sell_row["amount"]
                if sell_row["is_limit_down"] or sell_row["is_halt"]:
                    n_blocked_sells += 1
                    sell_p = sell_p * 0.98
                buy_bps = impact_bps(per_pos, buy_amt)
                sell_bps = impact_bps(per_pos, sell_amt)
                total_bps = buy_bps + sell_bps
                bps_list.append(total_bps)
                if pd.notna(buy_p) and pd.notna(sell_p) and buy_p > 0:
                    g = sell_p / buy_p - 1
                    rets.append(g - total_bps / 10000)
            except KeyError:
                continue
        if not rets:
            continue

        port_ret = float(np.mean(rets))
        avg_bps = float(np.mean(bps_list))

        # benchmark: universe 等权 (无滑点, 公平对比)
        uni = scores.index.tolist()
        b = []
        for code in uni:
            try:
                bp = df_idx.loc[(buy_date, code), "open"]
                sp = df_idx.loc[(sell_date, code), "close"]
                if pd.notna(bp) and pd.notna(sp) and bp > 0:
                    b.append(sp / bp - 1)
            except KeyError:
                continue
        bench_ret = float(np.mean(b)) if b else 0

        new_h = set(top)
        turnover = len(new_h ^ prev_holdings) / max(len(new_h) + len(prev_holdings), 1)
        rets_arr.append(port_ret)
        bench_arr.append(bench_ret)
        turnovers.append(turnover)
        impact_bps_list.append(avg_bps)
        prev_holdings = new_h

    if not rets_arr:
        return {"error": "no samples"}

    arr = np.array(rets_arr)
    bench = np.array(bench_arr)
    excess = arr - bench
    periods = 252 / rebalance_days

    if vol_target is not None:
        ex_vol = excess.std() * np.sqrt(periods)
        if ex_vol > 0:
            scale = vol_target / ex_vol
            arr = arr * scale
            excess = arr - bench * scale

    ann_ret = float(arr.mean() * periods)
    ann_vol = float(arr.std() * np.sqrt(periods))
    sharpe = ann_ret / (ann_vol + 1e-9)
    b_ann = float(bench.mean() * periods)
    b_vol = float(bench.std() * np.sqrt(periods))
    b_sh = b_ann / (b_vol + 1e-9)
    ex_ann = float(excess.mean() * periods)
    ex_vol = float(excess.std() * np.sqrt(periods))
    ir = ex_ann / (ex_vol + 1e-9)
    eq = pd.Series(arr).cumsum() + 1
    dd = float((eq - eq.cummax()).min())
    eq_ex = pd.Series(excess).cumsum() + 1
    dd_ex = float((eq_ex - eq_ex.cummax()).min())

    return {
        "n_rebalances": len(arr),
        "annual_return": ann_ret, "annual_vol": ann_vol, "sharpe": sharpe,
        "max_drawdown": dd,
        "bench_return": b_ann, "bench_vol": b_vol, "bench_sharpe": b_sh,
        "excess_return": ex_ann, "excess_vol": ex_vol, "info_ratio": ir,
        "excess_max_dd": dd_ex,
        "avg_turnover": float(np.mean(turnovers)),
        "avg_impact_bps": float(np.mean(impact_bps_list)),
        "blocked_buys": int(n_blocked_buys),
        "blocked_sells": int(n_blocked_sells),
        "top_k_ratio": top_ratio,
    }


def main(pool, start, end, top_ratio, rebalance_days):
    print(f"\n{'='*64}\n  V4 (三刀·去幻觉) {time.strftime('%Y-%m-%d %H:%M')}")
    print(f"  pool={pool} 期间={start}→{end} Top={top_ratio:.0%} reb={rebalance_days}d")
    print('='*64)

    print("\n[1/7] 股票池...")
    kline_path = CACHE / f"kline_{start}_{end}_n{pool}.parquet"
    if kline_path.exists():
        daily = pd.read_parquet(kline_path)
        print(f"  kline 缓存 {len(daily)} 行, {daily['code'].nunique()} 只")
    else:
        df_mkt = fetch_all_ashare()
        codes = filter_midcap_universe(df_mkt)[:pool]
        daily = bulk_fetch_daily(codes, start, end, sleep_ms=70)
        kline_path.parent.mkdir(parents=True, exist_ok=True)
        daily.to_parquet(kline_path)

    lhb_path = CACHE / f"lhb_{start}_{end}.parquet"
    lhb_df = pd.read_parquet(lhb_path) if lhb_path.exists() else pd.DataFrame()
    print(f"\n[2/7] 龙虎榜 {len(lhb_df)} 条")

    # V5: 加 insider
    insider_path = CACHE / f"insider_{start}_{end}.parquet"
    insider_df = (pd.read_parquet(insider_path) if insider_path.exists()
                   else fetch_insider_range(start, end, insider_path))
    print(f"  insider {len(insider_df)} 条")

    print("\n[3/7] 计算全部因子 (V5: +涨停+insider)...")
    feat_tech = compute_pandas_alpha(daily)
    feat_rev = compute_advanced_alpha(daily)
    feat_limit = compute_limit_alpha(daily)   # 刀四
    feat_combo = feat_tech.join(feat_rev, how="outer").join(feat_limit, how="outer")

    trading_dates = pd.DatetimeIndex(sorted(daily["date"].unique()))
    if not lhb_df.empty:
        feat_lhb = build_lhb_features(lhb_df, trading_dates)
        feat_combo = feat_combo.join(feat_lhb, how="left")
        for f in LHB_FACTOR_NAMES + LHB_B2_FACTOR_NAMES:
            if f in feat_combo.columns:
                feat_combo[f] = feat_combo[f].fillna(0)
    if not insider_df.empty:
        feat_ins = build_insider_features(insider_df, trading_dates)
        feat_combo = feat_combo.join(feat_ins, how="left")
        for f in INSIDER_FACTOR_NAMES:
            if f in feat_combo.columns:
                feat_combo[f] = feat_combo[f].fillna(0)
    for f in LIMIT_FACTOR_NAMES:
        if f in feat_combo.columns:
            feat_combo[f] = feat_combo[f].fillna(0)

    def _z(s):
        mu, sd = s.mean(), s.std()
        return (s - mu) / sd if sd > 0 else s * 0
    feat_z = feat_combo.groupby(level="date").transform(_z).clip(-3, 3).fillna(0)
    print(f"  特征 {feat_z.shape}")

    print(f"\n[4/7] 刀 2: 条件 Label (horizon={rebalance_days})...")
    label = make_conditional_label(daily, horizon=rebalance_days, dd_clip=0.25)
    print(f"  label mean={label.mean():.4f} std={label.std():.4f}")

    print(f"\n[5/7] 刀 1: IC 聚类去共线...")
    aligned = feat_z.join(label.rename("label"), how="inner")
    valid = aligned["label"].notna() & aligned.drop(columns="label").notna().all(axis=1)
    feat_valid = aligned.drop(columns="label")[valid]
    y_valid = aligned["label"][valid]
    selected = ic_cluster_select(feat_valid, y_valid,
                                  corr_threshold=0.6, min_ic=0.005)

    X = feat_valid[selected]
    y = y_valid
    print(f"\n[6/7] 训练样本 {len(X)}, 精选因子 {len(selected)}")
    pred = rolling_predict(X, y, train_days=252, step_days=21)
    print(f"  OOS 预测 {len(pred)}")

    ic = pred.groupby(level="date").apply(
        lambda s: s.corr(y.loc[s.index], method="spearman")
    ).dropna()
    print(f"  IC mean={ic.mean():.4f} IR={ic.mean()/ic.std():.2f} IC>0={float((ic>0).mean()):.2%}")

    print(f"\n[7/7] 刀 3: 按成交量滑点回测...")
    stats = backtest_v4(pred, daily, top_ratio=top_ratio,
                         rebalance_days=rebalance_days,
                         vol_target=0.20)

    print("\n" + "="*64 + f"\n  V4 回测结果\n" + "="*64)
    for k, v in stats.items():
        if isinstance(v, float):
            if any(s in k for s in ["return", "drawdown", "vol", "turnover"]):
                print(f"  {k:20s} {v:>10.2%}")
            elif "bps" in k:
                print(f"  {k:20s} {v:>10.1f}")
            else:
                print(f"  {k:20s} {v:>10.4f}")
        else:
            print(f"  {k:20s} {v}")

    out = ROOT / "output" / f"research_v4_{time.strftime('%Y%m%d_%H%M')}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(
        f"# V4 (三刀·去幻觉)\n\n"
        f"- 股票池: {daily['code'].nunique()} 只\n"
        f"- 期间: {start} → {end}\n"
        f"- 精选因子: {len(selected)} 个 (IC 聚类 corr<0.6)\n"
        f"- Label: 条件 max×(1-0.7·dd) (非线性, 奖励先涨后跌形态)\n"
        f"- 滑点: Almgren-Chriss 动态 (按 ADV)\n\n"
        f"## 保留的因子\n\n" + "\n".join("- " + s for s in selected) + "\n\n"
        f"## IC\n- mean: {ic.mean():.4f}\n- IR: {ic.mean()/ic.std():.2f}\n- IC>0: {float((ic>0).mean()):.2%}\n\n"
        f"## 回测\n" + "\n".join(f"- {k}: {v}" for k, v in stats.items()),
        encoding="utf-8",
    )
    print(f"\n报告: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=500)
    ap.add_argument("--start", default="20230101")
    ap.add_argument("--end", default="20260420")
    ap.add_argument("--top", type=float, default=0.05)
    ap.add_argument("--reb", type=int, default=30)
    args = ap.parse_args()
    main(args.pool, args.start, args.end, args.top, args.reb)
