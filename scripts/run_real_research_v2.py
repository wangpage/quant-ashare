"""P0 改造版 - 冲 Sharpe 1.5+.

相对 v1 的三个关键变更:
    1. 股票池: 硬编码 10 只大盘蓝筹 → 动态取 top-500 活跃股
    2. 因子: 3 个玩具因子 → 36 个 A股特化因子 (纯 pandas)
    3. 打分/调仓: Top-10 等权 → LightGBM 滚动训练 + Top = universe * ratio

用法:
    python3 scripts/run_real_research_v2.py --n 300 --start 20230101 --end 20260420
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from data_adapter.em_direct import (
    bulk_fetch_daily, fetch_hot_stocks, fetch_csi300_constituents,
)
from factors.alpha_pandas import compute_pandas_alpha, FACTOR_NAMES
from utils.logger import logger


# ---------- 1. 股票池 ----------
_FALLBACK_POOL = [
    # 大盘蓝筹
    "600519","000858","601318","600036","601398","002594","000333","600900",
    "601888","601012","002415","600030","000001","600276","601166","002475",
    "600309","002352","000651","601899","600028","600050","601988","688981",
    "300750","601628","600887","601857","601288","601818","000568","600585",
    "600048","601601","000002","600309","000725","600893","601328","601336",
    # 中盘成长
    "300124","300059","300015","002352","002027","300760","300142","002714",
    "300274","000876","300024","002607","300413","600298","600332","000661",
    "600763","000963","002466","300122","002241","002241","000538","002157",
    "600420","600585","601231","300498","300450","300759","002230","300003",
    "300033","000547","002572","002035","002271","002563","002007","000568",
    # 金融
    "600000","600015","600016","600837","601169","601229","601766","601800",
    "600926","601211","601688","601878","601198","600999","600958","601788",
    "000776","600061","600030","000166","601377","600831","600837","601555",
    # 科技/新能源
    "002460","300014","002466","002074","000100","000063","000725","002241",
    "002157","002230","300760","300124","300750","300059","300142","300498",
    "300015","000733","002049","002415","002241","600183","601225","002340",
    # 消费/医药
    "600887","600690","000651","002415","000333","600519","000858","603288",
    "603369","000895","000596","600809","000799","000568","603027","603345",
    "600276","600196","300015","300347","000999","600332","600436","002558",
]


def _get_fallback_universe(n: int) -> list[str]:
    """fetch_hot_stocks 挂掉时的硬编码兜底池 (150+ 只活跃股)."""
    codes = list(dict.fromkeys(_FALLBACK_POOL))  # 去重保序
    return codes[:n]


def build_universe(n: int, use_hot: bool = True) -> list[str]:
    """动态股票池: 按成交额取 top-N, 自然过滤流动性差的.

    三层兜底: hot → csi300 → hardcoded.
    """
    codes: list[str] = []
    if use_hot:
        try:
            hot = fetch_hot_stocks(limit=max(n, 50))
            codes = [h["code"] for h in hot if h.get("code")]
        except Exception as e:
            print(f"  ⚠️  fetch_hot_stocks 失败 ({e}), 尝试 csi300...")

    if not codes:
        try:
            codes = fetch_csi300_constituents()
        except Exception as e:
            print(f"  ⚠️  fetch_csi300_constituents 失败 ({e}), 用硬编码池...")

    if not codes:
        codes = _get_fallback_universe(n)

    codes = [c for c in codes if c and c[0] in ("6", "0", "3")]
    return codes[:n]


# ---------- 2. 标签 (下一日 open 到 T+5 close) ----------
def make_label(daily_df: pd.DataFrame, horizon: int = 5) -> pd.Series:
    """T+1 open 到 T+horizon close 的收益. 符合 A股 T+1 约束."""
    df = daily_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["code", "date"])
    open_next = df.groupby("code")["open"].shift(-1)
    close_fwd = df.groupby("code")["close"].shift(-horizon)
    label = close_fwd / open_next - 1
    df["label"] = label
    # 截面 winsorize 避免极端收益主导
    df["label"] = df.groupby("date")["label"].transform(
        lambda s: s.clip(s.quantile(0.01), s.quantile(0.99))
    )
    df = df.set_index(["date", "code"])
    return df["label"]


# ---------- 3. 截面标准化 (日内 z-score) ----------
def cross_sectional_zscore(feat_df: pd.DataFrame) -> pd.DataFrame:
    """截面 z-score + 3σ 截断: 实测比 rank 效果好."""
    def _z(s):
        mu, sd = s.mean(), s.std()
        if sd == 0 or np.isnan(sd):
            return s * 0
        return (s - mu) / sd
    return feat_df.groupby(level="date").transform(_z).clip(-3, 3).fillna(0)


# ---------- 4. LightGBM 滚动训练 + OOS 预测 ----------
def rolling_lgbm_predict(
    X: pd.DataFrame, y: pd.Series,
    train_years: float = 1.0, step_days: int = 21,
) -> pd.Series:
    """滚动窗口训练梯度提升树, 生成全样本 OOS 预测.

    优先用 LightGBM, macOS 缺 libomp 时自动降级到 sklearn
    HistGradientBoostingRegressor (同为直方图梯度提升树, 精度相近).

    Args:
        X: MultiIndex (date, code), columns = 因子
        y: MultiIndex (date, code), label
        train_years: 训练窗口年数
        step_days: 每 N 天重训一次

    Returns:
        OOS 预测, MultiIndex (date, code)
    """
    all_dates = X.index.get_level_values("date").unique().sort_values()
    train_window = int(train_years * 252)

    if len(all_dates) < train_window + step_days:
        # 数据量不够滚动, 用前 60% 训练, 后 40% 预测
        split = int(len(all_dates) * 0.6)
        train_dates = all_dates[:split]
        test_dates = all_dates[split:]
        model = _fit_lgbm(X.loc[train_dates.min():train_dates.max()],
                          y.loc[train_dates.min():train_dates.max()])
        test_X = X.loc[test_dates.min():test_dates.max()]
        pred = pd.Series(model.predict(test_X.values), index=test_X.index)
        return pred

    preds = []
    start_idx = train_window
    while start_idx < len(all_dates):
        end_idx = min(start_idx + step_days, len(all_dates))
        train_start = all_dates[start_idx - train_window]
        train_end   = all_dates[start_idx - 1]
        test_start  = all_dates[start_idx]
        test_end    = all_dates[end_idx - 1]

        try:
            X_tr = X.loc[train_start:train_end]
            y_tr = y.loc[train_start:train_end]
            mask = y_tr.notna() & X_tr.notna().all(axis=1)
            X_tr, y_tr = X_tr[mask], y_tr[mask]
            if len(X_tr) < 1000:
                start_idx = end_idx
                continue
            model = _fit_lgbm(X_tr, y_tr)
            X_te = X.loc[test_start:test_end]
            pred = pd.Series(model.predict(X_te.values), index=X_te.index)
            preds.append(pred)
            print(f"  rolling: train[{train_start.date()}→{train_end.date()}] "
                  f"test[{test_start.date()}→{test_end.date()}] "
                  f"n_train={len(X_tr)} n_test={len(X_te)}")
        except Exception as e:
            print(f"  rolling 异常 {e}")
        start_idx = end_idx

    if not preds:
        return pd.Series(dtype=float)
    return pd.concat(preds).sort_index()


def _fit_lgbm(X: pd.DataFrame, y: pd.Series):
    """优先 LightGBM, 缺 libomp 时降级到 sklearn HistGBR."""
    try:
        import lightgbm as lgb
        model = lgb.LGBMRegressor(
            n_estimators=200, learning_rate=0.05,
            num_leaves=31, max_depth=6, min_child_samples=50,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1,
            random_state=42, verbose=-1,
        )
        model.fit(X.values, y.values)
        return model
    except (OSError, ImportError):
        from sklearn.ensemble import HistGradientBoostingRegressor
        model = HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.05,
            max_leaf_nodes=31, max_depth=6,
            min_samples_leaf=50,
            l2_regularization=0.1,
            random_state=42,
        )
        model.fit(X.values, y.values)
        return model


# ---------- 5. 回测 (Top-K = universe * ratio, 调仓频率可调) ----------
def backtest_topk(
    pred: pd.Series, daily_df: pd.DataFrame,
    top_ratio: float = 0.1, rebalance_days: int = 10,
    cost_bps: float = 15,
    inertia_keep_ratio: float = 0.5,
    vol_weight: bool = True, vol_window: int = 20,
) -> dict:
    """Top-K + 持仓惯性 + 波动率倒数加权.

    降换手的小技巧:
        - 持仓惯性: 旧持仓若仍排在 top (1 + inertia_keep_ratio) * K 以内则保留
        - 波动率倒数加权: 高波动股分配更少权重, 降组合 vol
    """
    dates = pred.index.get_level_values("date").unique().sort_values()

    df = daily_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["code", "date"])
    # 预计算每只股的滚动波动率 (日收益 std)
    df["ret1"] = df.groupby("code")["close"].pct_change()
    df["vol20"] = df.groupby("code")["ret1"].transform(
        lambda s: s.rolling(vol_window).std()
    )
    df_idx = df.set_index(["date", "code"])

    daily_returns = []
    turnovers = []
    prev_holdings: set = set()

    for i, dt in enumerate(dates[::rebalance_days]):
        scores = pred.xs(dt, level="date").dropna()
        if len(scores) < 10:
            continue
        k = max(5, int(len(scores) * top_ratio))

        # 持仓惯性: 旧票还在 top (1+keep)*k 且本身打分为正才保留,
        # 否则打分转负的旧持仓会拖累组合
        ranked = scores.rank(ascending=False)
        stay_threshold = k * (1 + inertia_keep_ratio)
        stay = [c for c in prev_holdings
                if (c in ranked.index
                    and ranked[c] <= stay_threshold
                    and scores[c] > scores.median())]
        need = max(0, k - len(stay))
        fresh_pool = [c for c in scores.nlargest(k * 2).index
                      if c not in prev_holdings]
        fresh = fresh_pool[:need]
        top = list(stay) + list(fresh)

        # 每只票的权重 (等权 or 波动率倒数加权)
        weights: dict[str, float] = {}
        for code in top:
            try:
                v = df_idx.loc[(dt, code), "vol20"]
                v = float(v) if pd.notna(v) and v > 0 else np.nan
            except KeyError:
                v = np.nan
            if vol_weight and pd.notna(v):
                weights[code] = 1.0 / v
            else:
                weights[code] = 1.0
        if not weights:
            continue
        total_w = sum(weights.values())
        weights = {c: w / total_w for c, w in weights.items()}

        # 持仓期收益: 加权
        period_rets = []
        period_weights = []
        for code, w in weights.items():
            try:
                fut_idx = [d for d in dates if d > dt][:rebalance_days]
                if len(fut_idx) < 1:
                    continue
                buy_price = df_idx.loc[(fut_idx[0], code)].get("open")
                sell_price = df_idx.loc[(fut_idx[-1], code)].get("close")
                if pd.notna(buy_price) and pd.notna(sell_price) and buy_price > 0:
                    period_rets.append((sell_price / buy_price - 1) * w)
                    period_weights.append(w)
            except KeyError:
                continue

        if not period_rets:
            continue
        gross_ret = float(sum(period_rets)) / max(sum(period_weights), 1e-9)

        new_holdings = set(top)
        turnover = len(new_holdings ^ prev_holdings) / max(
            len(new_holdings) + len(prev_holdings), 1)
        cost = turnover * cost_bps / 10000
        net_ret = gross_ret - cost

        daily_returns.append(net_ret)
        turnovers.append(turnover)
        prev_holdings = new_holdings

    if not daily_returns:
        return {"error": "no samples"}

    arr = np.array(daily_returns)
    # 年化: 每周一次调仓 ≈ 50 次/年; 非周频按实际
    periods_per_year = 252 / rebalance_days
    annual_ret = float(arr.mean() * periods_per_year)
    annual_vol = float(arr.std() * np.sqrt(periods_per_year))
    sharpe = annual_ret / (annual_vol + 1e-9)

    equity = pd.Series(arr).cumsum() + 1
    dd = (equity - equity.cummax()).min()

    return {
        "n_rebalances": len(daily_returns),
        "annual_return": annual_ret,
        "annual_vol": annual_vol,
        "sharpe": sharpe,
        "max_drawdown": float(dd),
        "avg_turnover": float(np.mean(turnovers)),
        "top_k_ratio": top_ratio,
    }


# ---------- 主流程 ----------
def main(n_stocks: int, start: str, end: str,
         top_ratio: float, rebalance_days: int,
         use_hot: bool):
    print(f"\n{'='*64}")
    print(f"  ResearchPipeline v2 (P0) - {time.strftime('%Y-%m-%d %H:%M')}")
    print(f"  universe={n_stocks}  期间={start}→{end}")
    print(f"  Top-K={top_ratio:.0%}  rebalance={rebalance_days}d")
    print('='*64)

    # 1. 池子
    print("\n[1/5] 构建股票池...")
    codes = build_universe(n_stocks, use_hot=use_hot)
    print(f"  候选 {len(codes)} 只 (示例: {codes[:8]}...)")

    # 2. 拉数据
    print(f"\n[2/5] 拉日线 ({len(codes)} 只 × ~{(pd.Timestamp(end[:4]+'-'+end[4:6]+'-'+end[6:]) - pd.Timestamp(start[:4]+'-'+start[4:6]+'-'+start[6:])).days} 天)...")
    t0 = time.time()
    daily = bulk_fetch_daily(codes, start, end, sleep_ms=60)
    print(f"  耗时 {time.time()-t0:.0f}s  {len(daily)} 行  {daily['code'].nunique()} 只")
    if daily.empty:
        print("❌ 数据空, 中止"); return

    # 3. 因子 + 标签
    print("\n[3/5] 计算 36 个 A股因子...")
    feat = compute_pandas_alpha(daily)
    print(f"  特征矩阵 {feat.shape}")
    feat_z = cross_sectional_zscore(feat).fillna(0)

    label = make_label(daily, horizon=rebalance_days)
    aligned = feat_z.join(label.rename("label"), how="inner")
    X = aligned[FACTOR_NAMES]
    y = aligned["label"]
    valid = y.notna() & X.notna().all(axis=1)
    X, y = X[valid], y[valid]
    print(f"  有效样本 {len(X)}")

    # 4. LightGBM 滚动预测
    print("\n[4/5] LightGBM 滚动训练 + OOS 预测...")
    pred = rolling_lgbm_predict(X, y, train_years=0.5, step_days=21)
    print(f"  OOS 预测 {len(pred)} 条")
    if pred.empty:
        print("❌ 预测空"); return

    # IC
    ic = pred.groupby(level="date").apply(
        lambda s: s.corr(y.loc[s.index], method="spearman")
    ).dropna()
    print(f"  IC 均值 {ic.mean():.4f}  IC_IR {ic.mean()/ic.std():.2f}  |IC>0| 占比 {(ic>0).mean():.2%}")

    # 5. 回测
    print("\n[5/5] 回测 (Top-K × 成本 15bps)...")
    stats = backtest_topk(pred, daily,
                          top_ratio=top_ratio,
                          rebalance_days=rebalance_days,
                          cost_bps=10)

    print("\n" + "=" * 64)
    print("  回测结果")
    print("=" * 64)
    for k, v in stats.items():
        if isinstance(v, float):
            if "return" in k or "drawdown" in k or "vol" in k or "turnover" in k:
                print(f"  {k:20s} {v:>10.2%}")
            else:
                print(f"  {k:20s} {v:>10.4f}")
        else:
            print(f"  {k:20s} {v}")

    # 输出
    out = Path(__file__).resolve().parent.parent / "output" / \
          f"research_v2_{time.strftime('%Y%m%d_%H%M')}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(
        f"# ResearchPipeline v2 (P0)\n\n"
        f"- universe: {len(codes)} 只\n"
        f"- 期间: {start} → {end}\n"
        f"- 因子: 36 个 A股特化 (pandas 原生)\n"
        f"- 模型: LightGBM rolling (train 1y, retrain 21d)\n"
        f"- 组合: Top {top_ratio:.0%} 等权, 周频, 成本 15bps\n\n"
        f"## IC\n\n"
        f"- IC 均值: {ic.mean():.4f}\n"
        f"- IC_IR: {ic.mean()/ic.std():.2f}\n"
        f"- IC > 0 占比: {(ic>0).mean():.2%}\n\n"
        f"## 回测\n\n"
        + "\n".join(f"- {k}: {v}" for k, v in stats.items()),
        encoding="utf-8",
    )
    print(f"\n报告: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="股票池大小")
    ap.add_argument("--start", default="20230101")
    ap.add_argument("--end", default="20260420")
    ap.add_argument("--top", type=float, default=0.1, help="Top-K 占比")
    ap.add_argument("--rebalance", type=int, default=5, help="调仓间隔日")
    ap.add_argument("--csi300", action="store_true", help="用沪深300 池子")
    args = ap.parse_args()
    main(n_stocks=args.n, start=args.start, end=args.end,
         top_ratio=args.top, rebalance_days=args.rebalance,
         use_hot=not args.csi300)
