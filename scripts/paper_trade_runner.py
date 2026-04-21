"""Paper Trade Runner - 自维护模拟账户, 每日一跑.

与 daily_paper_trade.py 的区别:
    旧: 只记录"今日 top 25 信号" → positions_log.csv
    新: 维护真·账户 (现金/持仓), 模拟 T+1 买入 + 持仓到期卖出,
        每日滚动更新, 生成真实 P&L 曲线.

状态文件:
    output/paper_trade/account.json
    {
      "start_date": "2026-04-21",
      "initial_cash": 1000000,
      "cash": 875634,
      "positions": {
        "300750": {"shares": 100, "cost": 432.50, "buy_date": "2026-04-10",
                    "target_exit": "2026-05-20", "rank": 3}
      },
      "equity_history": [{"date": "...", "nav": 1005200}, ...],
      "trade_log": [{"date":"..","code":"..","side":"buy","price":..,"shares":..}]
    }

调用:
    python3 scripts/paper_trade_runner.py                    # 今日收盘后跑
    python3 scripts/paper_trade_runner.py --asof 2026-04-18  # 重放历史某日
    python3 scripts/paper_trade_runner.py --reset            # 清零重建账户
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
PAPER = ROOT / "output" / "paper_trade"
PAPER.mkdir(parents=True, exist_ok=True)
ACCOUNT_PATH = PAPER / "account.json"
TRADE_LOG_PATH = PAPER / "trade_log.csv"

# 配置
INITIAL_CASH = 1_000_000    # 100 万起始
TOP_K = 25                    # 每次持仓 25 只
HOLDING_DAYS = 30             # 持仓 30 交易日
COST_BPS_SINGLE = 10          # 单边 10 bps (佣金 3 + 印花 5 + 滑点 2)


# ---------- 账户状态 ----------
def load_account() -> dict:
    if ACCOUNT_PATH.exists():
        return json.loads(ACCOUNT_PATH.read_text(encoding="utf-8"))
    return {
        "start_date": None,
        "initial_cash": INITIAL_CASH,
        "cash": INITIAL_CASH,
        "positions": {},       # code → {shares, cost, buy_date, target_exit}
        "equity_history": [],  # [{date, nav, cash, position_value}]
        "trade_log": [],       # 内存快照, 详尽版写 csv
    }


def save_account(acc: dict):
    ACCOUNT_PATH.write_text(
        json.dumps(acc, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def append_trade_log(trade: dict):
    """追加交易到 CSV, 便于事后分析."""
    is_new = not TRADE_LOG_PATH.exists()
    df = pd.DataFrame([trade])
    df.to_csv(TRADE_LOG_PATH, mode="a", header=is_new, index=False)


# ---------- 定价查询 ----------
def load_daily_cache() -> pd.DataFrame:
    kline_path = CACHE / "kline_20230101_20260420_n500.parquet"
    df = pd.read_parquet(kline_path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def get_price(daily: pd.DataFrame, code: str, dt: pd.Timestamp,
               field: str = "open") -> float | None:
    """查某只股某天的价格. field: open/close/high/low."""
    sub = daily[(daily["code"] == code) & (daily["date"] == dt)]
    if sub.empty:
        return None
    return float(sub[field].iloc[0])


def get_latest_close(daily: pd.DataFrame, code: str,
                      before: pd.Timestamp) -> float | None:
    sub = daily[(daily["code"] == code) & (daily["date"] <= before)]
    if sub.empty:
        return None
    return float(sub.sort_values("date").iloc[-1]["close"])


# ---------- 交易 ----------
def execute_sell(acc: dict, code: str, price: float,
                  shares: int, dt: pd.Timestamp, reason: str) -> float:
    """卖出: 现金 += shares × price × (1 - cost), 返回净收益."""
    gross = price * shares
    cost = gross * COST_BPS_SINGLE / 10000
    net = gross - cost
    acc["cash"] += net

    pos = acc["positions"].get(code, {})
    buy_cost = pos.get("cost", price) * shares
    pnl = net - buy_cost
    pnl_pct = pnl / buy_cost if buy_cost > 0 else 0

    trade = {
        "date": str(dt.date()), "code": code, "side": "sell",
        "shares": shares, "price": price, "gross": gross, "cost": cost,
        "net": net, "pnl": pnl, "pnl_pct": pnl_pct, "reason": reason,
    }
    acc["trade_log"].append(trade)
    append_trade_log(trade)
    del acc["positions"][code]
    return pnl


def execute_buy(acc: dict, code: str, price: float,
                 amount: float, dt: pd.Timestamp,
                 target_exit: pd.Timestamp, rank: int) -> bool:
    """买入: amount 元资金买 floor(amount / price / 100) × 100 股 (A股最小 100)."""
    shares = int(amount / price / 100) * 100
    if shares < 100:
        return False
    gross = shares * price
    cost = gross * COST_BPS_SINGLE / 10000
    total_out = gross + cost
    if total_out > acc["cash"]:
        return False
    acc["cash"] -= total_out
    acc["positions"][code] = {
        "shares": shares,
        "cost": price,
        "buy_date": str(dt.date()),
        "target_exit": str(target_exit.date()),
        "rank": rank,
    }
    trade = {
        "date": str(dt.date()), "code": code, "side": "buy",
        "shares": shares, "price": price, "gross": gross,
        "cost": cost, "net": -total_out, "rank": rank,
    }
    acc["trade_log"].append(trade)
    append_trade_log(trade)
    return True


# ---------- 每日流程 ----------
def mark_to_market(acc: dict, daily: pd.DataFrame, dt: pd.Timestamp) -> dict:
    """计算当前 NAV (现金 + 持仓市值)."""
    pos_value = 0.0
    for code, pos in acc["positions"].items():
        p = get_latest_close(daily, code, dt)
        if p is None:
            p = pos["cost"]   # 取不到用成本价
        pos_value += p * pos["shares"]
    nav = acc["cash"] + pos_value
    return {
        "date": str(dt.date()),
        "nav": round(nav, 2),
        "cash": round(acc["cash"], 2),
        "position_value": round(pos_value, 2),
        "n_positions": len(acc["positions"]),
    }


def sell_expired(acc: dict, daily: pd.DataFrame, dt: pd.Timestamp) -> float:
    """T+1 开盘卖出: 持仓到期 (target_exit ≤ dt) 的全部卖出."""
    total_pnl = 0.0
    expired = [c for c, p in acc["positions"].items()
                if pd.Timestamp(p["target_exit"]) <= dt]
    for code in expired:
        pos = acc["positions"][code]
        p = get_price(daily, code, dt, "open")
        if p is None:
            p = get_latest_close(daily, code, dt)
            reason = "expired_no_open_fallback_close"
        else:
            reason = "expired"
        if p is not None:
            pnl = execute_sell(acc, code, p, pos["shares"], dt, reason)
            total_pnl += pnl
    if expired:
        print(f"  卖出 {len(expired)} 只到期持仓, PnL {total_pnl:+.0f}")
    return total_pnl


def buy_new_signals(acc: dict, top: pd.Series, daily: pd.DataFrame,
                     dt: pd.Timestamp, target_exit: pd.Timestamp,
                     max_positions: int = TOP_K):
    """剔除已持有的, 从空余仓位买入新信号."""
    existing = set(acc["positions"].keys())
    candidates = [c for c in top.index if c not in existing]
    slots = max_positions - len(acc["positions"])
    if slots <= 0:
        print(f"  仓位已满 {len(acc['positions'])}/{max_positions}, 不开新仓")
        return
    candidates = candidates[:slots]
    if not candidates:
        return

    # 每只等权分配: 现金 / slots
    per_slot = acc["cash"] / slots * 0.98  # 留 2% buffer 防止超买
    bought = 0
    for i, code in enumerate(candidates):
        price = get_price(daily, code, dt, "open")
        if price is None:
            continue
        ok = execute_buy(acc, code, price, per_slot, dt, target_exit,
                          rank=top.index.tolist().index(code) + 1)
        if ok:
            bought += 1
    print(f"  买入 {bought}/{len(candidates)} 只新持仓, 每只 ~{per_slot/1e4:.1f} 万")


# ---------- 主流程 ----------
def run_one_day(asof: pd.Timestamp, top: pd.Series, daily: pd.DataFrame,
                 acc: dict, trading_dates: list[pd.Timestamp]):
    """每日流程: 卖到期 → 买新 → mark to market."""
    # T+1 开盘 (asof 的下一交易日)
    next_td = next((d for d in trading_dates if d > asof), None)
    if next_td is None:
        print(f"  ⚠️  {asof.date()} 之后无交易日, 仅 mark-to-market")
        mtm = mark_to_market(acc, daily, asof)
        acc["equity_history"].append(mtm)
        return

    # 卖出到期 (按 T+1 open)
    sell_expired(acc, daily, next_td)

    # 算目标持仓到期日 = next_td + HOLDING_DAYS 交易日
    idx = trading_dates.index(next_td)
    target_idx = min(idx + HOLDING_DAYS, len(trading_dates) - 1)
    target_exit = trading_dates[target_idx]

    # 买入新信号 (按 T+1 open)
    buy_new_signals(acc, top, daily, next_td, target_exit)

    # MTM at T+1 close
    mtm = mark_to_market(acc, daily, next_td)
    acc["equity_history"].append(mtm)
    print(f"  {next_td.date()} NAV={mtm['nav']/1e4:.1f} 万 "
          f"(现金 {mtm['cash']/1e4:.1f} 万, 持仓 {mtm['n_positions']} 只)")


def generate_signal_today(asof: pd.Timestamp, daily: pd.DataFrame) -> pd.Series:
    """用当前缓存生成 asof 的 top-K 信号.
    复用 run_holdout_v6 的核心逻辑, 但只跑最后一个窗口.
    """
    # 导入这里延后以减少启动开销
    from scripts.run_real_research_v5 import (
        make_conditional_label, ic_cluster_select, _fit_model,
    )
    from factors.alpha_pandas import compute_pandas_alpha
    from factors.alpha_reversal import compute_advanced_alpha
    from factors.alpha_limit import compute_limit_alpha, LIMIT_FACTOR_NAMES
    from factors.adaptive_polarity import apply_adaptive_polarity
    from data_adapter.lhb import (
        build_lhb_features, LHB_FACTOR_NAMES, LHB_B2_FACTOR_NAMES,
    )
    from data_adapter.insider import (
        build_insider_features, INSIDER_FACTOR_NAMES,
    )

    # label horizon 与持仓周期一致
    horizon = HOLDING_DAYS

    # 因子
    feat_tech = compute_pandas_alpha(daily)
    feat_rev = compute_advanced_alpha(daily)
    feat_limit = compute_limit_alpha(daily)
    feat_combo = feat_tech.join(feat_rev, how="outer").join(feat_limit, how="outer")
    trading_dates = pd.DatetimeIndex(sorted(daily["date"].unique()))

    # 龙虎榜 + insider
    lhb_files = [p for p in CACHE.glob("lhb_2*.parquet") if "taxonomy" not in p.name]
    if lhb_files:
        lhb_df = pd.read_parquet(sorted(lhb_files)[-1])
        feat_lhb = build_lhb_features(lhb_df, trading_dates)
        feat_combo = feat_combo.join(feat_lhb, how="left")
        for f in LHB_FACTOR_NAMES + LHB_B2_FACTOR_NAMES:
            if f in feat_combo.columns:
                feat_combo[f] = feat_combo[f].fillna(0)
    ins_files = list(CACHE.glob("insider_*.parquet"))
    if ins_files:
        ins_df = pd.read_parquet(sorted(ins_files)[-1])
        feat_ins = build_insider_features(ins_df, trading_dates)
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
    # feat_z 保留全部日期 (末尾用于预测, 不需要 label)
    label = make_conditional_label(daily, horizon=horizon, dd_clip=0.25)

    # 训练期: label 必须 notna
    label_valid_idx = label.dropna().index
    # 有效特征的 X (所有 feature 列非 NaN)
    feat_ok_mask = feat_z.notna().all(axis=1)
    feat_ok_idx = feat_z[feat_ok_mask].index

    # IC 聚类只用 ≤ asof - horizon 的训练数据
    all_dates = feat_z.index.get_level_values("date").unique().sort_values()
    all_before = all_dates[all_dates <= asof]
    if len(all_before) < 252 + horizon:
        raise ValueError(f"历史数据不足 ({len(all_before)} < 252+{horizon})")
    ic_cutoff = all_before[-(horizon + 1)]

    train_mask_for_ic = (
        feat_z.index.get_level_values("date") <= ic_cutoff
    )
    # 对齐 label (必须 notna)
    train_idx_for_ic = feat_z[train_mask_for_ic & feat_ok_mask].index
    train_idx_for_ic = train_idx_for_ic.intersection(label_valid_idx)

    selected = ic_cluster_select(
        feat_z.loc[train_idx_for_ic], label.loc[train_idx_for_ic],
        corr_threshold=0.6, min_ic=0.005,
    )
    feat_sel = feat_z[selected]

    # 自适应极性: 只用训练期 label 算 IC, 但 feat_adapt 覆盖全部日期
    # apply_adaptive_polarity 自己会 shift(horizon) 防 leak, 可以传完整 label
    feat_adapt, _ = apply_adaptive_polarity(
        feat_sel, label, horizon=horizon, window=90,
        z_threshold=0.8, z_cap=3.0, inertia=0.6,
    )
    all_zero = (feat_adapt.abs().sum(axis=1) < 1e-9)
    feat_adapt = feat_adapt[~all_zero]

    # 训练: [asof - 252 交易日 - horizon, asof - horizon]
    tr_dates = feat_adapt.index.get_level_values("date").unique().sort_values()
    tr_before = tr_dates[tr_dates <= asof]
    if len(tr_before) <= horizon:
        raise ValueError(f"adapt 后训练日不足")
    tr_end = tr_before[-(horizon + 1)]
    tr_start = tr_before[max(0, len(tr_before) - 252 - horizon)]
    X_tr = feat_adapt.loc[tr_start:tr_end]
    y_tr_for_train = label.loc[X_tr.index.intersection(label_valid_idx)]
    X_tr_aligned = X_tr.loc[y_tr_for_train.index]
    mask = y_tr_for_train.notna() & X_tr_aligned.notna().all(axis=1)
    if mask.sum() < 500:
        raise ValueError(f"训练样本不足 {mask.sum()}")
    model = _fit_model(X_tr_aligned[mask], y_tr_for_train[mask])

    # 预测 asof 当日截面 — 允许末尾 label NaN 的日期 (这正是 paper trade 需要的)
    avail = feat_adapt.index.get_level_values("date").unique()
    asof_used = avail[avail <= asof].max()
    X_te = feat_adapt.xs(asof_used, level="date", drop_level=True)
    pred = pd.Series(model.predict(X_te.values), index=X_te.index)
    pred.index.name = "code"
    return pred.nlargest(TOP_K), asof_used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default=None, help="YYYY-MM-DD")
    ap.add_argument("--reset", action="store_true", help="重置账户")
    args = ap.parse_args()

    if args.reset:
        if ACCOUNT_PATH.exists():
            ACCOUNT_PATH.unlink()
        if TRADE_LOG_PATH.exists():
            TRADE_LOG_PATH.unlink()
        print("✓ 账户已重置")

    print(f"\n{'='*64}\n  📊 Paper Trade Runner\n{'='*64}")

    asof = pd.Timestamp(args.asof) if args.asof else pd.Timestamp.today().normalize()
    print(f"  asof: {asof.date()}")

    acc = load_account()
    if acc["start_date"] is None:
        acc["start_date"] = str(asof.date())
        print(f"  🎬 初始化账户: 起始资金 ¥{INITIAL_CASH/1e4:.0f} 万")

    daily = load_daily_cache()
    trading_dates = sorted(daily["date"].unique())

    # 跳过非交易日
    if asof not in trading_dates:
        # 找最近的交易日 ≤ asof
        nearest = max((d for d in trading_dates if d <= asof), default=None)
        if nearest is None:
            print(f"❌ {asof.date()} 之前无交易日"); return
        print(f"  ⚠️  {asof.date()} 非交易日, 退到 {nearest.date()}")
        asof = nearest

    # 生成信号
    print(f"\n[1/3] 生成 {asof.date()} top {TOP_K} 信号...")
    try:
        top, sig_date = generate_signal_today(asof, daily)
        print(f"  信号日: {sig_date.date()}, top {len(top)} 股均分打分")
    except Exception as e:
        print(f"❌ 信号生成失败: {e}"); return

    # 执行一日流程
    print(f"\n[2/3] 执行 T+1 卖出到期 + 买入新信号...")
    run_one_day(sig_date, top, daily, acc, trading_dates)

    # 保存
    save_account(acc)

    # 打印总结
    print(f"\n[3/3] 账户概览")
    nav_series = pd.DataFrame(acc["equity_history"])
    if len(nav_series):
        nav = nav_series["nav"].iloc[-1]
        nav0 = INITIAL_CASH
        tot_ret = (nav - nav0) / nav0
        dur = len(nav_series)
        print(f"  起始资金:   ¥{nav0/1e4:>8.1f} 万")
        print(f"  当前 NAV:   ¥{nav/1e4:>8.1f} 万")
        print(f"  累计收益:   {tot_ret:+.2%}")
        print(f"  现金:       ¥{acc['cash']/1e4:>8.1f} 万")
        print(f"  持仓:       {len(acc['positions'])} 只")
        print(f"  已跑天数:    {dur} 日")

    print(f"\n  账户状态: {ACCOUNT_PATH}")
    print(f"  交易日志: {TRADE_LOG_PATH}")


if __name__ == "__main__":
    main()
