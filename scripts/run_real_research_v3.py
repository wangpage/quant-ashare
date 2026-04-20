"""V3 - 全市场中小盘 + 龙虎榜事件因子 + 可交易约束.

关键升级:
    1. 股票池: 硬编码 → sina 全市场中小盘 (流通 30-300 亿) top 500
    2. 因子: 36 → 41 (加 5 个龙虎榜事件因子)
    3. 回测: 加涨跌停不可成交约束 + 停牌剔除
    4. 调仓: 周频 → 月频 (rebalance=20), 大幅降成本
    5. 信号: T 收盘 → T+1 open 买入 (现实化)

学术依据:
    - Han-Hirshleifer-Walden 2022: A股小盘反转 OOS Sharpe 1.5+
    - 国内私募研究: 龙虎榜 + 中小盘 OOS Sharpe 2-3 (A股特色)
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
from data_adapter.sina_universe import (
    fetch_all_ashare, filter_midcap_universe,
)
from data_adapter.lhb import (
    fetch_lhb_range, build_lhb_features,
    LHB_FACTOR_NAMES, LHB_B2_FACTOR_NAMES,
)
from factors.alpha_pandas import compute_pandas_alpha, FACTOR_NAMES as TECH_FACTORS
from factors.alpha_reversal import compute_advanced_alpha, ADVANCED_FACTOR_NAMES

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"


# ---------- 1. 股票池 ----------
def build_universe_v3(n_pool: int = 500,
                       min_float_cap: float = 30,
                       max_float_cap: float = 300) -> list[str]:
    """sina 全市场 → 中小盘 → 流动性 top n."""
    df = fetch_all_ashare(max_pages=60)
    if df.empty:
        raise RuntimeError("sina 池子拉取失败")
    codes = filter_midcap_universe(
        df, min_float_cap=min_float_cap, max_float_cap=max_float_cap,
        min_amount=0.5,
    )
    return codes[:n_pool]


# ---------- 2. 标签 (T+1 open → T+horizon close) ----------
def make_label(daily_df: pd.DataFrame, horizon: int = 20) -> pd.Series:
    df = daily_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["code", "date"])
    open_next = df.groupby("code")["open"].shift(-1)
    close_fwd = df.groupby("code")["close"].shift(-horizon)
    label = close_fwd / open_next - 1
    df["label"] = label
    df["label"] = df.groupby("date")["label"].transform(
        lambda s: s.clip(s.quantile(0.01), s.quantile(0.99))
    )
    return df.set_index(["date", "code"])["label"]


# ---------- 3. 截面 z-score ----------
def cs_zscore(feat_df: pd.DataFrame) -> pd.DataFrame:
    def _z(s):
        mu, sd = s.mean(), s.std()
        return (s - mu) / sd if sd > 0 else s * 0
    return feat_df.groupby(level="date").transform(_z).clip(-3, 3).fillna(0)


# ---------- 4. LightGBM 滚动 ----------
def _fit_model(X, y):
    """更强正则 + 特征采样, 防止 LHB 因子 importance 过度集中."""
    try:
        import lightgbm as lgb
        m = lgb.LGBMRegressor(
            n_estimators=400, learning_rate=0.03,
            num_leaves=24, max_depth=5, min_child_samples=150,
            subsample=0.7, colsample_bytree=0.5,  # 每棵树只看半数因子
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


def rolling_predict(X: pd.DataFrame, y: pd.Series,
                     train_days: int = 252, step_days: int = 21) -> pd.Series:
    dates = X.index.get_level_values("date").unique().sort_values()
    if len(dates) < train_days + step_days:
        split = int(len(dates) * 0.6)
        tr = X.loc[dates[0]:dates[split-1]]
        ty = y.loc[dates[0]:dates[split-1]]
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
            X_tr, y_tr = X.loc[tr_s:tr_e], y.loc[tr_s:tr_e]
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


# ---------- 5. 可交易约束回测 ----------
def backtest_tradeable(
    pred: pd.Series, daily_df: pd.DataFrame,
    top_ratio: float = 0.1, rebalance_days: int = 20,
    cost_bps: float = 10,
    limit_pct: float = 0.097,
    vol_target: float | None = None,
    inertia_keep: float = 0.6,    # 旧持仓若排名在 (1+keep)*k 内保留
    vol_weight: bool = True,       # 波动率倒数加权
) -> dict:
    """加可交易约束 (涨停不买/跌停不卖/停牌剔除).

    - 信号生成: T 日收盘
    - 买入: T+1 open, 但若 T+1 一字涨停 (open == high == low 且 ret > 9.5%), 不能买
    - 卖出: rebalance 末日 close, 若该日跌停, 顺延一日
    """
    df = daily_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["code", "date"])
    df["ret1"] = df.groupby("code")["close"].pct_change()
    df["vol20"] = df.groupby("code")["ret1"].transform(
        lambda s: s.rolling(20).std())
    df["is_limit_up"]    = (df["ret1"] > limit_pct).astype(int)
    df["is_limit_down"]  = (df["ret1"] < -limit_pct).astype(int)
    df["is_one_word_up"] = ((df["open"] == df["high"]) &
                             (df["high"] == df["low"]) &
                             (df["ret1"] > limit_pct)).astype(int)
    df["is_halt"] = (df["volume"] == 0).astype(int)
    df_idx = df.set_index(["date", "code"])

    dates = pred.index.get_level_values("date").unique().sort_values()

    daily_returns = []
    turnovers = []
    n_blocked_buys = 0
    n_blocked_sells = 0
    prev_holdings: set = set()

    rebalance_dates = list(dates[::rebalance_days])

    for i, dt in enumerate(rebalance_dates):
        scores = pred.xs(dt, level="date").dropna()
        if len(scores) < 20:
            continue
        k = max(10, int(len(scores) * top_ratio))

        # 持仓惯性: 旧票仍在 top (1+keep)*k 且打分为正, 保留
        ranked = scores.rank(ascending=False)
        stay_threshold = k * (1 + inertia_keep)
        stay = [c for c in prev_holdings
                if (c in ranked.index and ranked[c] <= stay_threshold
                    and scores[c] > scores.median())]
        need = max(0, k - len(stay))
        fresh_pool = [c for c in scores.nlargest(int(k * 2)).index
                      if c not in prev_holdings]
        candidates = list(stay) + list(fresh_pool[:int(need * 1.5)])

        fut = [d for d in dates if d > dt]
        if len(fut) < rebalance_days:
            continue
        buy_date = fut[0]
        sell_date = fut[rebalance_days - 1]

        # 过滤: T+1 一字涨停 / 停牌 不能买 (旧持仓也要过滤, 如停牌需剔)
        admitted = []
        for code in candidates:
            try:
                row = df_idx.loc[(buy_date, code)]
                if row["is_one_word_up"] or row["is_halt"]:
                    n_blocked_buys += 1
                    continue
                admitted.append(code)
            except KeyError:
                continue
            if len(admitted) >= k:
                break

        if not admitted:
            continue
        top = admitted

        # 权重: 波动率倒数加权 或 等权
        weights: dict[str, float] = {}
        for code in top:
            try:
                v = df_idx.loc[(dt, code), "vol20"]
                v = float(v) if pd.notna(v) and v > 0 else None
            except KeyError:
                v = None
            if vol_weight and v is not None:
                weights[code] = 1.0 / v
            else:
                weights[code] = 1.0
        tw = sum(weights.values())
        weights = {c: w / tw for c, w in weights.items()}

        rets, w_used = [], []
        for code, w in weights.items():
            try:
                buy_p = df_idx.loc[(buy_date, code), "open"]
                sell_row = df_idx.loc[(sell_date, code)]
                sell_p = sell_row["close"]
                if sell_row["is_limit_down"] or sell_row["is_halt"]:
                    n_blocked_sells += 1
                    sell_p = sell_p * 0.98
                if pd.notna(buy_p) and pd.notna(sell_p) and buy_p > 0:
                    rets.append((sell_p / buy_p - 1) * w)
                    w_used.append(w)
            except KeyError:
                continue

        if not rets:
            continue
        gross = float(sum(rets)) / max(sum(w_used), 1e-9)

        new_h = set(top)
        turnover = len(new_h ^ prev_holdings) / max(len(new_h) + len(prev_holdings), 1)
        cost = turnover * cost_bps / 10000
        net = gross - cost

        daily_returns.append(net)
        turnovers.append(turnover)
        prev_holdings = new_h

    if not daily_returns:
        return {"error": "no samples"}

    arr = np.array(daily_returns)
    periods = 252 / rebalance_days

    # === 同期 universe 等权 benchmark (算 alpha) ===
    bench_rets = []
    for i, dt in enumerate(rebalance_dates):
        fut = [d for d in dates if d > dt]
        if len(fut) < rebalance_days:
            continue
        buy_date, sell_date = fut[0], fut[rebalance_days - 1]
        # 当日 universe = pred 当天的所有 code
        try:
            uni = pred.xs(dt, level="date").index.tolist()
        except KeyError:
            continue
        rs = []
        for code in uni:
            try:
                bp = df_idx.loc[(buy_date, code), "open"]
                sp = df_idx.loc[(sell_date, code), "close"]
                if pd.notna(bp) and pd.notna(sp) and bp > 0:
                    rs.append(sp / bp - 1)
            except KeyError:
                continue
        if rs:
            bench_rets.append(float(np.mean(rs)))
    bench_arr = np.array(bench_rets) if bench_rets else np.zeros(len(arr))

    # 对齐长度
    n = min(len(arr), len(bench_arr))
    arr_a, bench_a = arr[:n], bench_arr[:n]
    excess = arr_a - bench_a

    # vol targeting: 直接对超额收益
    if vol_target is not None:
        actual_vol = excess.std() * np.sqrt(periods)
        if actual_vol > 0:
            scale = vol_target / actual_vol
            arr_a = arr_a * scale
            excess = arr_a - bench_a * scale

    annual_ret    = float(arr_a.mean() * periods)
    annual_vol    = float(arr_a.std()  * np.sqrt(periods))
    sharpe        = annual_ret / (annual_vol + 1e-9)
    bench_ret     = float(bench_a.mean() * periods)
    bench_vol     = float(bench_a.std()  * np.sqrt(periods))
    bench_sharpe  = bench_ret / (bench_vol + 1e-9)
    excess_ret    = float(excess.mean() * periods)
    excess_vol    = float(excess.std()  * np.sqrt(periods))
    info_ratio    = excess_ret / (excess_vol + 1e-9)
    eq = pd.Series(arr_a).cumsum() + 1
    dd = float((eq - eq.cummax()).min())
    eq_ex = pd.Series(excess).cumsum() + 1
    dd_ex = float((eq_ex - eq_ex.cummax()).min())

    return {
        "n_rebalances": len(arr_a),
        "annual_return": annual_ret,
        "annual_vol": annual_vol,
        "sharpe": sharpe,
        "max_drawdown": dd,
        "bench_return": bench_ret,
        "bench_vol": bench_vol,
        "bench_sharpe": bench_sharpe,
        "excess_return": excess_ret,
        "excess_vol": excess_vol,
        "info_ratio": info_ratio,
        "excess_max_dd": dd_ex,
        "avg_turnover": float(np.mean(turnovers)),
        "blocked_buys": int(n_blocked_buys),
        "blocked_sells": int(n_blocked_sells),
        "top_k_ratio": top_ratio,
    }


# ---------- 主流程 ----------
def main(n_pool: int, start: str, end: str,
         top_ratio: float, rebalance_days: int):
    print(f"\n{'='*64}\n  V3 (中小盘 + 龙虎榜事件 + 可交易) - {time.strftime('%Y-%m-%d %H:%M')}")
    print(f"  pool={n_pool}  期间={start}→{end}  Top={top_ratio:.0%}  reb={rebalance_days}d\n{'='*64}")

    # 1. 池子
    print("\n[1/6] sina 拉全市场, 筛中小盘...")
    codes = build_universe_v3(n_pool=n_pool)
    print(f"  中小盘 top {len(codes)}: 示例 {codes[:8]}...")

    # 2. 日线 (带缓存)
    cache_kline = CACHE / f"kline_{start}_{end}_n{len(codes)}.parquet"
    if cache_kline.exists():
        daily = pd.read_parquet(cache_kline)
        print(f"\n[2/6] kline 缓存命中 {len(daily)} 行")
    else:
        print(f"\n[2/6] 拉日线 ({len(codes)} 只)...")
        t0 = time.time()
        daily = bulk_fetch_daily(codes, start, end, sleep_ms=70)
        print(f"  耗时 {time.time()-t0:.0f}s  {len(daily)} 行  {daily['code'].nunique()} 只")
        cache_kline.parent.mkdir(parents=True, exist_ok=True)
        daily.to_parquet(cache_kline)
    if daily.empty:
        return

    # 3. 龙虎榜
    cache_lhb = CACHE / f"lhb_{start}_{end}.parquet"
    print(f"\n[3/6] 拉龙虎榜...")
    lhb_df = fetch_lhb_range(start, end, cache_path=cache_lhb)
    print(f"  龙虎榜 {len(lhb_df)} 条, 涉及 {lhb_df['code'].nunique() if len(lhb_df) else 0} 只")

    # 4. 因子
    print(f"\n[4/6] 计算技术因子 + 反转因子 + 龙虎榜因子...")
    feat_tech = compute_pandas_alpha(daily)
    feat_rev  = compute_advanced_alpha(daily)
    feat_combo = feat_tech.join(feat_rev, how="outer")

    # 龙虎榜面板特征
    if not lhb_df.empty:
        all_dates = pd.DatetimeIndex(sorted(daily["date"].unique()))
        feat_lhb = build_lhb_features(lhb_df, all_dates)
        # join (lhb 可能只覆盖部分股票, 缺省值 fillna 0)
        feat_combo = feat_combo.join(feat_lhb, how="left")
        for f in LHB_FACTOR_NAMES + LHB_B2_FACTOR_NAMES:
            if f in feat_combo.columns:
                feat_combo[f] = feat_combo[f].fillna(0)

    feat_z = cs_zscore(feat_combo)
    print(f"  特征矩阵 {feat_z.shape}, 因子数 {feat_z.shape[1]}")

    # 5. 标签 + 模型
    label = make_label(daily, horizon=rebalance_days)
    aligned = feat_z.join(label.rename("label"), how="inner")
    feat_cols = [c for c in feat_z.columns if c in aligned.columns]
    X = aligned[feat_cols]
    y = aligned["label"]
    valid = y.notna() & X.notna().all(axis=1)
    X, y = X[valid], y[valid]
    print(f"\n[5/6] 训练样本 {len(X)}, 因子 {len(feat_cols)}")
    pred = rolling_predict(X, y, train_days=252, step_days=21)
    print(f"  OOS 预测 {len(pred)}")

    # IC
    ic = pred.groupby(level="date").apply(
        lambda s: s.corr(y.loc[s.index], method="spearman")
    ).dropna()
    print(f"  IC mean={ic.mean():.4f}  IR={ic.mean()/ic.std():.2f}  IC>0={float((ic>0).mean()):.2%}")

    # 6. 回测
    print(f"\n[6/6] 可交易回测 (cost=10bps, 月频)...")
    stats = backtest_tradeable(pred, daily,
                                top_ratio=top_ratio,
                                rebalance_days=rebalance_days,
                                cost_bps=10,
                                vol_target=0.20)

    print("\n" + "="*64 + "\n  V3 回测结果\n" + "="*64)
    for k, v in stats.items():
        if isinstance(v, float):
            if any(s in k for s in ["return","drawdown","vol","turnover"]):
                print(f"  {k:20s} {v:>10.2%}")
            else:
                print(f"  {k:20s} {v:>10.4f}")
        else:
            print(f"  {k:20s} {v}")

    out = ROOT / "output" / f"research_v3_{time.strftime('%Y%m%d_%H%M')}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(
        f"# V3 (中小盘+龙虎榜+可交易)\n\n"
        f"- 池子: {len(codes)} 只 (sina 中小盘 top, 流通 30-300 亿)\n"
        f"- 期间: {start} → {end}\n"
        f"- 因子: {len(feat_cols)} 个 (技术+反转+龙虎榜)\n"
        f"- 调仓: 月频 ({rebalance_days}d)\n"
        f"- 成本: 10bps 单边\n"
        f"- 约束: 涨停不买/跌停不卖/停牌剔除\n\n"
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
    ap.add_argument("--top", type=float, default=0.1)
    ap.add_argument("--reb", type=int, default=20)
    args = ap.parse_args()
    main(n_pool=args.pool, start=args.start, end=args.end,
         top_ratio=args.top, rebalance_days=args.reb)
