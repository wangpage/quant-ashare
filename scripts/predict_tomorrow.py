"""predict_tomorrow: 一键式"明日交易建议" 全链路编排.

把项目 7 个模块串起来:
  [0] watchlist + paper_trade account
  [1] RegimeDetector → regime + position_mult
  [2] 5 类信号并行:
      A) ML 因子 (generate_signal_today return_full_pred=True)
      B) Watchlist 18 因子 (compute_signals_df)
      C) Radar long/avoid
      D) Theme 扩散 (可选, 依赖网络)
      E) Sentiment batch (可选, 依赖网络)
  [3] 综合 rank → Top 10 进辩论
  [4] Top 10 跑完整 TradingAgentTeam (4 路 analyst + bull/bear + judge + trader + risk)
  [5] PreTradeGate 过滤 → 可执行清单
  [6] 飞书 bot 推送 + JSON 存档

用法:
  python3 scripts/predict_tomorrow.py                    # 默认完整
  python3 scripts/predict_tomorrow.py --dry-run          # 不推飞书
  python3 scripts/predict_tomorrow.py --no-theme         # 跳 theme (东财限速时)
  python3 scripts/predict_tomorrow.py --no-sentiment     # 跳 sentiment
  python3 scripts/predict_tomorrow.py --skip-agent       # 不跑 Agent 辩论, 只产综合排名
  python3 scripts/predict_tomorrow.py --top 15           # 辩论 top-15
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# 加载 .env
_env = ROOT / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


# ==================== 综合打分权重 ====================
SCORE_WEIGHTS = {
    "watchlist_v2": 0.35,
    "ml_factor":    0.25,
    "radar_long":   0.20,
    "theme":        0.10,
    "sentiment":    0.10,
}
RADAR_AVOID_PENALTY = 10.0  # 硬否决


# ==================== 辅助 ====================
def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_watchlist(csv_path: Path) -> list[dict]:
    out = []
    with open(csv_path, encoding="utf-8") as f:
        rdr = csv.reader(f)
        next(rdr)  # header
        for row in rdr:
            if len(row) < 2:
                continue
            name = row[0].strip()
            full_code = row[1].strip()
            if not full_code or "." not in full_code:
                continue
            code = full_code.split(".")[0]
            if not code.isdigit() or len(code) != 6:
                continue
            if name.startswith("深证") or name.startswith("上证"):
                continue
            out.append({
                "code": code, "name": name,
                "px": row[2].strip() if len(row) > 2 else "",
                "chg": row[3].strip() if len(row) > 3 else "",
            })
    return out


def load_account() -> dict:
    p = ROOT / "output" / "paper_trade" / "account.json"
    if not p.exists():
        return {"cash": 1_000_000, "positions": {}, "total_value": 1_000_000}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"cash": 1_000_000, "positions": {}, "total_value": 1_000_000}


# ==================== [1] Regime ====================
def stage_regime(daily: pd.DataFrame, asof: pd.Timestamp) -> dict:
    """跑 RegimeDetector, 返回 {regime, position_mult, allow_new_long, context}."""
    try:
        from market_regime import RegimeDetector
    except Exception as e:
        log(f"⚠️  regime 模块导入失败: {e}")
        return {"regime": "unknown", "position_mult": 0.5,
                 "allow_new_long": True, "context": "(regime 不可用)"}

    # 准备 index_df (HS300 v4_idx parquet)
    idx_files = sorted(ROOT.glob("cache/v4_idx_*.parquet"))
    if idx_files:
        idx_df = pd.read_parquet(idx_files[-1])
        if "date" in idx_df.columns:
            idx_df["date"] = pd.to_datetime(idx_df["date"])
    else:
        log("⚠️  无指数缓存, regime 退化")
        return {"regime": "unknown", "position_mult": 0.5,
                 "allow_new_long": True, "context": "(无指数数据)"}

    # 准备 stocks_daily (今日截面, 含 pct_chg)
    today_bars = daily[daily["date"] == asof].copy()
    if "pct_chg" not in today_bars.columns:
        # 粗算
        today_bars["pct_chg"] = 0.0

    try:
        detector = RegimeDetector()
        sig = detector.detect(idx_df, stocks_daily=today_bars)
        return {
            "regime": sig.regime.value if hasattr(sig.regime, "value") else str(sig.regime),
            "position_mult": float(sig.position_mult),
            "allow_new_long": bool(sig.allow_new_long),
            "preferred_strategy": getattr(sig, "preferred_strategy", ""),
            "context": sig.to_agent_context(),
            "confidence": float(getattr(sig, "confidence", 0.0)),
        }
    except Exception as e:
        log(f"⚠️  RegimeDetector 失败: {e}")
        traceback.print_exc()
        return {"regime": "unknown", "position_mult": 0.5,
                 "allow_new_long": True, "context": f"(regime 失败: {e})"}


# ==================== [2] 5 类信号 ====================
def stage_ml_factor(asof: pd.Timestamp, daily: pd.DataFrame) -> pd.Series | None:
    """全 universe pred series. 返回 None 表示失败."""
    try:
        from scripts.paper_trade_runner import generate_signal_today
        pred, sig_date = generate_signal_today(asof, daily, return_full_pred=True)
        log(f"  ML 因子: {len(pred)} 只, rank1 {pred.index[0]} score {pred.iloc[0]:+.4f}")
        return pred
    except Exception as e:
        log(f"⚠️  ML 因子失败: {e}")
        return None


def stage_watchlist_v2(codes: list[str]) -> pd.DataFrame:
    """watchlist v2 18 因子. 返回 DataFrame (index=code, cols alpha_z...)."""
    try:
        from scripts.watchlist_signal_v2 import compute_signals_df
        sig = compute_signals_df(codes, daily=None, use_minute=False, quiet=True)
        log(f"  watchlist_v2: {len(sig)} 只, top z={sig['alpha_z'].iloc[0]:+.2f}"
            if not sig.empty else f"  watchlist_v2: 空")
        return sig
    except Exception as e:
        log(f"⚠️  watchlist_v2 失败: {e}")
        return pd.DataFrame()


def stage_radar() -> dict:
    """拉 radar long / avoid."""
    try:
        from llm_layer.radar_candidates import (
            get_radar_long_candidates, get_radar_avoid_codes,
        )
        longs = get_radar_long_candidates(
            since_hours=48, min_conf=0.5, top=30, min_half_life_hours=24,
        )
        avoids = get_radar_avoid_codes(since_hours=48, min_conf=0.5)
        log(f"  radar: long {len(longs)} 只, avoid {len(avoids)} 只")
        return {"longs": longs, "avoids": avoids}
    except Exception as e:
        log(f"⚠️  radar 失败: {e}")
        return {"longs": [], "avoids": set()}


def stage_theme(daily: pd.DataFrame, asof: pd.Timestamp,
                 skip: bool = False) -> list:
    """detect_emerging_themes. 网络失败或 skip=True 返回 []."""
    if skip:
        log("  theme: 跳过 (--no-theme)")
        return []
    try:
        from data_adapter.theme_builder import build_theme_map
        theme_map = build_theme_map(mode="quick", min_pct_chg=3.0)
        if not theme_map:
            log("  theme: map 为空, 跳过")
            return []
        from thematic_investing.emerging_themes import detect_emerging_themes
        # 把 daily 转成 panel (code, date 多级索引, columns close/volume/turnover_rate)
        panel = daily.copy()
        panel = panel.rename(columns={"turnover": "turnover_rate"})
        # detect_emerging_themes 要求时间序列, 只保留最近 120 天
        panel["date"] = pd.to_datetime(panel["date"])
        recent = panel[panel["date"] >= asof - pd.Timedelta(days=120)]
        if "turnover_rate" not in recent.columns:
            recent["turnover_rate"] = 0.0
        sigs = detect_emerging_themes(recent, theme_map, as_of=asof)
        log(f"  theme: {len(sigs)} 个题材信号")
        return sigs
    except Exception as e:
        log(f"⚠️  theme 失败: {e}")
        return []


def stage_sentiment_batch(codes: list[str], skip: bool = False) -> dict:
    """批量 snownlp 舆情打分. code -> {score, conf, n}."""
    if skip:
        return {}
    try:
        from llm_layer.sentiment import NewsSentimentAnalyzer
        analyzer = NewsSentimentAnalyzer(backend="snownlp")
        out = {}
        t0 = time.time()
        for c in codes:
            try:
                s = analyzer.score(c)
                out[c] = {"score": s.score, "conf": s.confidence, "n": s.sample_size}
            except Exception:
                pass
            if time.time() - t0 > 45:  # 超时保护
                log(f"⚠️  sentiment 超时, 已跑 {len(out)}/{len(codes)}")
                break
        log(f"  sentiment: {len(out)}/{len(codes)} 只打分成功")
        return out
    except Exception as e:
        log(f"⚠️  sentiment 失败: {e}")
        return {}


# ==================== [3] 综合排名 ====================
def _z(series: pd.Series) -> pd.Series:
    mu, sd = series.mean(), series.std()
    return (series - mu) / (sd + 1e-9) if sd > 0 else series * 0


def stage_compose(
    watchlist: list[dict],
    ml_pred: pd.Series | None,
    wl_sig: pd.DataFrame,
    radar: dict,
    theme_sigs: list,
    sentiment_map: dict,
) -> pd.DataFrame:
    """综合打分. 候选池 = watchlist ∪ ML top50 ∪ radar long."""
    pool = {w["code"]: w["name"] for w in watchlist}
    if ml_pred is not None:
        for c in ml_pred.head(50).index:
            pool.setdefault(c, "")
    for c in radar.get("longs", []):
        pool.setdefault(c["code"], c.get("name", ""))

    df = pd.DataFrame([{"code": c, "name": n} for c, n in pool.items()])
    df = df.set_index("code")

    # watchlist v2 alpha_z
    if not wl_sig.empty:
        df["wl_z"] = wl_sig["alpha_z"].reindex(df.index).fillna(0.0)
    else:
        df["wl_z"] = 0.0

    # ML factor z
    if ml_pred is not None:
        ml_series = ml_pred.reindex(df.index).fillna(ml_pred.median())
        df["ml_z"] = _z(ml_series)
    else:
        df["ml_z"] = 0.0

    # Radar long boost
    long_map = {c["code"]: c for c in radar.get("longs", [])}
    df["radar_long_conf"] = df.index.map(
        lambda c: long_map.get(c, {}).get("conf", 0.0)
    )

    # Radar avoid flag
    avoids = radar.get("avoids", set())
    df["radar_avoid"] = df.index.map(lambda c: c in avoids)

    # Theme emerging boost
    theme_boost = {}
    for ts in theme_sigs:
        if ts.stage == "emerging":
            for ld in ts.leaders:
                theme_boost[ld] = theme_boost.get(ld, 0.0) + 1.0
    df["theme_boost"] = df.index.map(lambda c: theme_boost.get(c, 0.0))

    # Sentiment
    df["sentiment_score"] = df.index.map(
        lambda c: sentiment_map.get(c, {}).get("score", 0.0)
    )

    # 综合
    composite = (
        SCORE_WEIGHTS["watchlist_v2"] * df["wl_z"]
        + SCORE_WEIGHTS["ml_factor"]  * df["ml_z"]
        + SCORE_WEIGHTS["radar_long"] * df["radar_long_conf"]
        + SCORE_WEIGHTS["theme"]      * df["theme_boost"]
        + SCORE_WEIGHTS["sentiment"]  * df["sentiment_score"]
        - RADAR_AVOID_PENALTY * df["radar_avoid"].astype(float)
    )
    df["composite"] = composite
    return df.sort_values("composite", ascending=False)


# ==================== [4] Agent 辩论 ====================
async def run_debate(code: str, name: str, factor_score: float,
                     radar_summary: str, regime_ctx: dict,
                     portfolio_state: dict, team) -> dict:
    """一只股跑 decide_async 四路辩论, 返回精简 dict."""
    from llm_layer.candidate_data_builder import build_data_blob
    blob = build_data_blob(
        code=code, name=name, factor_score=factor_score,
        include_sentiment=True,  # 跑 Agent 的用 sentiment
        sentiment_backend="snownlp",
    )
    try:
        dec = await team.decide_async(
            code=code, name=name,
            fundamentals=blob["fundamentals"],
            kline=blob["kline"],
            factor_score=factor_score,
            indicators=blob["indicators"],
            sentiment_data=blob["sentiment_data"],
            portfolio_state=portfolio_state,
            macro_signals={
                "regime": regime_ctx.get("regime", "unknown"),
                "market_trend": "neutral",
                "money_effect": "neutral",
                "events": regime_ctx.get("context", ""),
            },
            radar_summary=blob["radar_summary"],
        )
        views = dec.analyst_views or {}
        return {
            "code": code, "name": name,
            "action": dec.action,
            "conviction": dec.conviction,
            "score": dec.score,
            "size_pct": dec.size_pct,
            "price": dec.price,
            "stop_loss": dec.stop_loss_price,
            "take_profit": dec.take_profit_price,
            "holding_days": dec.holding_days,
            "reasoning": dec.reasoning[:300] if dec.reasoning else "",
            "risk_decision": dec.risk_decision,
            "views": {
                k: (v.get("view") if isinstance(v, dict) else None)
                for k, v in views.items()
            },
            "fundamentals_used": blob["fundamentals"][:80],
            "radar_summary_used": blob["radar_summary"][:150],
        }
    except Exception as e:
        log(f"    [debate] {code} 失败: {e}")
        return {"code": code, "name": name, "action": "error", "error": str(e)[:200]}


async def stage_agent_debate(
    top_candidates: pd.DataFrame,
    regime_ctx: dict,
    portfolio_state: dict,
    concurrency: int = 3,
    debate_rounds: int = 0,
) -> list[dict]:
    """并发跑 Agent 4 路辩论. concurrency 控制同时有几只在跑."""
    from llm_layer.agents import TradingAgentTeam
    backend = os.getenv("RADAR_TRIAGE_BACKEND", "qwen")
    team = TradingAgentTeam(
        backend=backend,
        analyst_model=os.getenv("RADAR_TRIAGE_MODEL", "qwen-turbo"),
        researcher_model=os.getenv("RADAR_DEEP_MODEL", "qwen-plus"),
        trader_model=os.getenv("RADAR_DEEP_MODEL", "qwen-plus"),
        risk_model=os.getenv("RADAR_TRIAGE_MODEL", "qwen-turbo"),
        debate_rounds=debate_rounds,
    )
    sem = asyncio.Semaphore(concurrency)
    done_count = [0]
    total = len(top_candidates)

    async def _one(code, row):
        async with sem:
            done_count[0] += 1
            idx = done_count[0]
            name = row["name"] if row["name"] else "--"
            log(f"  [{idx}/{total}] debate {code} {name} (start)")
            t0 = time.time()
            r = await run_debate(
                code=code, name=row["name"],
                factor_score=float(row["ml_z"]),
                radar_summary="", regime_ctx=regime_ctx,
                portfolio_state=portfolio_state, team=team,
            )
            r["code"] = code
            log(f"  [{idx}/{total}] {code} done in {time.time()-t0:.0f}s "
                f"→ {r.get('action','?')}")
            return r

    tasks = [_one(code, row) for code, row in top_candidates.iterrows()]
    return await asyncio.gather(*tasks)


# ==================== [5] Gate 过滤 ====================
def stage_gate(
    decisions: list[dict],
    account: dict,
    daily: pd.DataFrame,
    asof: pd.Timestamp,
    fresh_cash: float | None = None,
) -> list[dict]:
    """
    fresh_cash: None = 用真账户(含持仓+剩余 cash); >0 = 模拟一个全新
                N 元的空仓账户, 只检查标的自身约束(涨停/停牌/仓位比例).
                推荐预测场景下用 fresh_cash=1_000_000 避免已满仓干扰.
    """
    try:
        from risk import build_default_gate, Portfolio, OrderIntent
        from risk.a_share_rules import Position
    except Exception as e:
        log(f"⚠️  风控模块不可用: {e}")
        return [{**d, "gate_severity": "SKIP"} for d in decisions]

    try:
        initial = account.get("initial_cash", 1_000_000)

        # fresh_cash 模式: 全新空仓, 只做标的自身约束
        if fresh_cash is not None and fresh_cash > 0:
            portfolio = Portfolio(
                cash=fresh_cash,
                initial_capital=fresh_cash,
                positions={},
                high_water_mark=fresh_cash,
                daily_start_value=fresh_cash,
            )
            today_bars = daily[daily["date"] == asof].set_index("code")
            log(f"  Portfolio (fresh): cash ¥{fresh_cash:,.0f}, 模拟空仓")
            gate = build_default_gate()
            today = asof.date() if hasattr(asof, "date") else asof
            return _run_gate_loop(decisions, gate, portfolio, today_bars, today,
                                    fresh_cash, fallback_daily=daily)

        # 真账户模式
        positions: dict = {}
        today_bars = daily[daily["date"] == asof].set_index("code")
        for code, pos in (account.get("positions") or {}).items():
            shares = int(pos.get("shares", 0) or 0)
            if shares <= 0:
                continue
            # 今日收盘价
            if code in today_bars.index:
                px = float(today_bars.loc[code, "close"])
            else:
                px = float(pos.get("cost", 0) or 0)
            positions[code] = Position(
                code=code, shares=shares,
                avg_cost=float(pos.get("cost", px) or px),
                current_price=px,
                open_date=date.fromisoformat(
                    str(pos.get("buy_date", "2024-01-01"))[:10]
                ),
            )

        # predict_tomorrow 场景: 把"今天收盘时的 total_value"当作起点, 避免
        # 历史熔断/回撤误触发(那些机制适用于回测/实盘连续交易流,
        # 预测场景只关心"明天能不能下单",不关心过去账户亏损).
        positions_mv = sum(p.market_value for p in positions.values())
        today_value = account.get("cash", initial) + positions_mv

        portfolio = Portfolio(
            cash=account.get("cash", initial),
            initial_capital=initial,
            positions=positions,
            high_water_mark=today_value,     # 清零历史水位线
            daily_start_value=today_value,   # → daily_pnl_pct = 0
        )
        log(f"  Portfolio: cash ¥{portfolio.cash:,.0f}, "
            f"持仓 {len(positions)} 只, total ¥{portfolio.total_value:,.0f}, "
            f"daily_pnl {portfolio.daily_pnl_pct:+.2%}")
    except Exception as e:
        log(f"⚠️  Portfolio 构造失败: {e}, 用默认")
        traceback.print_exc()
        portfolio = None

    if portfolio is None:
        return [{**d, "gate_severity": "SKIP"} for d in decisions]

    gate = build_default_gate()
    today_bars = daily[daily["date"] == asof].set_index("code")
    today = asof.date() if hasattr(asof, "date") else asof
    return _run_gate_loop(decisions, gate, portfolio, today_bars, today,
                            portfolio.cash, fallback_daily=daily)


def _run_gate_loop(decisions, gate, portfolio, today_bars, today,
                    available_cash, fallback_daily=None):
    """fallback_daily: 完整 daily DataFrame, 用于当今日无 bar 时退到最近一日."""
    from risk import OrderIntent
    out = []
    for d in decisions:
        if d.get("action") != "buy":
            out.append({**d, "gate_severity": "NO_BUY"})
            continue
        code = d["code"]
        if code in today_bars.index:
            b = today_bars.loc[code]
        elif fallback_daily is not None:
            sub = fallback_daily[fallback_daily["code"] == code]
            if sub.empty:
                out.append({**d, "gate_severity": "SKIP",
                            "gate_reason": "daily cache 无该股"})
                continue
            b = sub.sort_values("date").iloc[-1]
        else:
            out.append({**d, "gate_severity": "SKIP",
                        "gate_reason": "无今日 bar 数据"})
            continue
        price = float(b["close"])
        prev_close = (price / (1 + float(b.get("pct_chg", 0)) / 100)
                       if b.get("pct_chg") else price)

        # 按建议仓位的 5% 算试探 shares
        shares = int((available_cash * 0.05 / max(price, 1e-6)) // 100 * 100)
        if shares <= 0:
            shares = 100

        try:
            intent = OrderIntent(
                code=code, side="buy", shares=shares,
                price=price, prev_close=prev_close,
                industry="",
                suspended=False,
                conviction=float(d.get("conviction") or 0.5),
            )
            dec = gate.check(intent, portfolio, today)
            out.append({
                **d,
                "gate_severity": dec.severity,
                "gate_allow": dec.allow,
                "gate_adjusted_shares": dec.adjusted_shares,
                "gate_reason": dec.reason,
                "suggest_price": price,
            })
        except Exception as e:
            out.append({**d, "gate_severity": "ERROR",
                        "gate_reason": str(e)[:120]})
    return out


# ==================== [6] 报告 ====================
def render_report(regime: dict, composed: pd.DataFrame,
                   top_decisions: list[dict],
                   theme_sigs: list, wl_sig: pd.DataFrame,
                   asof_str: str, args) -> str:
    """纯 bullet list 渲染 (飞书 markdown 不支持 GFM table)."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    L = [f"# 🎯 明日交易建议  (asof {asof_str})", f"_生成 {now}_", ""]

    # 1. 市场状态
    L.append("## 📊 市场状态")
    L.append(f"- **regime**: `{regime.get('regime', 'unknown')}`  "
             f"(信心 {regime.get('confidence', 0)*100:.0f}%)")
    L.append(f"- **仓位乘数**: {regime.get('position_mult', 0.5):.2f}  |  "
             f"**允许新多头**: {'✅' if regime.get('allow_new_long') else '❌'}")
    L.append(f"- context: _{regime.get('context', '')[:200]}_")
    L.append("")

    # 2. 综合排名 Top 5 (bullet)
    L.append("## 🏆 综合排名 Top 5  (ML+Watchlist+Radar 合议)")
    for i, (code, r) in enumerate(composed.head(5).iterrows(), 1):
        name = r.get("name") or "--"
        avoid_tag = "· 🔴 radar avoid" if r.get("radar_avoid") else ""
        L.append(f"**{i}. `{code}` {name}**  · 综合 `{r['composite']:+.3f}`")
        L.append(f"   - Watchlist z={r['wl_z']:+.2f}  ·  ML z={r['ml_z']:+.2f}  "
                 f"·  Radar conf={r['radar_long_conf']:.2f}  {avoid_tag}")
    L.append("")

    # 3. Agent 辩论结果
    action_emoji = {"buy": "🟢 买入", "sell": "🔴 卖出",
                     "hold": "⚪ 观望", "error": "⚠️ 错误"}
    risk_emoji = {"approve": "通过", "hold": "持有",
                   "reject": "❌ 否决", "modify": "⚠️ 调整"}
    sev_emoji = {"PASS": "✅", "WARN": "⚠️", "HARD_REJECT": "❌",
                  "SKIP": "⏭", "NO_BUY": "—", "ERROR": "⚠️", "SKIP": "⏭"}

    L.append(f"## 🤖 4 路 Agent 辩论 (top {len(top_decisions)})")
    L.append("_fund/tech/sent/event 4 分析师 → bull/bear 辩论 → judge → trader → risk_mgr_")
    L.append("")
    for d in top_decisions:
        code = d["code"]; name = d.get("name") or "--"
        act = action_emoji.get(d.get("action", ""), d.get("action", "?"))
        conv = d.get("conviction", 0) or 0
        score = d.get("score", 0) or 0
        risk = risk_emoji.get(d.get("risk_decision", ""),
                               d.get("risk_decision", "?"))
        views = d.get("views") or {}
        vote = " · ".join(
            f"{k[:4]}:{v}" for k, v in views.items() if v
        ) or "无"
        L.append(f"- **`{code}` {name}**  → {act}  "
                 f"(conv {conv:.2f}, score {score:+.2f}, risk={risk})")
        L.append(f"  - 分析师投票: {vote}")
        if d.get("gate_severity"):
            sev = d["gate_severity"]
            emoji = sev_emoji.get(sev, sev)
            L.append(f"  - Gate: {emoji} {d.get('gate_reason','')[:60]}")
    L.append("")

    # 4. 买入决定 / 无通过 诊断
    buys = [d for d in top_decisions
             if d.get("action") == "buy" and d.get("gate_severity") == "PASS"]
    if buys:
        L.append(f"## 🟢 最终可买入 ({len(buys)} 只)")
        for b in buys:
            px = b.get("suggest_price", 0)
            L.append(f"- **`{b['code']}` {b.get('name','')}**  ¥{px:.2f}  "
                     f"(信心 {b.get('conviction',0):.2f})")
            if b.get("reasoning"):
                L.append(f"  - {b['reasoning'][:150]}")
        L.append("")
    else:
        L.append("## ⚠️ 明日无明确可买标的")
        L.append("")
        L.append('Agent 团队对 Top 5 均未形成"buy"合议. '
                 "最接近通过的 2 只(说明为什么没选中):")
        nearest = sorted(top_decisions,
                          key=lambda t: -(t.get("conviction") or 0))[:2]
        for t in nearest:
            views = t.get("views") or {}
            rs = views.get("research", "?")
            L.append(f"- `{t['code']}` {t.get('name','--')}  "
                     f"conv={t.get('conviction',0):.2f}  "
                     f"action={t.get('action')}  researcher→{rs}")
        L.append("")
        L.append("_常见解释: fundamentals/sentiment 数据不可得时, 对应 analyst "
                 "无法形成看多论据, researcher 倾向 bear → trader 选 hold/sell._")
        L.append("")

    # 5. Watchlist v2 top 5
    if not wl_sig.empty:
        L.append("## 📋 Watchlist v2 18 因子 Top 5")
        for code, r in wl_sig.head(5).iterrows():
            tag = f"{r.get('cat_sign','')}{r.get('top_category','')}"
            L.append(f"- `{code}`  alpha_z=**{r['alpha_z']:+.2f}**  主导 {tag}")
        L.append("")

    # 6. ML 因子 top 5 (从 composed 里 ml_z 排)
    ml_ranked = composed.nlargest(5, "ml_z")
    if len(ml_ranked):
        L.append("## 🤖 ML 因子 Top 5 (全市场横截面)")
        for code, r in ml_ranked.iterrows():
            name = r.get("name") or "--"
            tag = " ⭐ 自选" if name and name != "--" else ""
            L.append(f"- `{code}` {name}  ml_z=**{r['ml_z']:+.2f}**{tag}")
        L.append("")

    # 7. Radar 信号
    radar_longs = composed[composed["radar_long_conf"] > 0].head(5)
    radar_avoids = composed[composed["radar_avoid"]].head(3)
    if len(radar_longs) or len(radar_avoids):
        L.append("## 📡 Radar 事件信号")
        for code, r in radar_longs.iterrows():
            L.append(f"- 🟢 `{code}` {r.get('name') or '--'}  "
                     f"radar conf={r['radar_long_conf']:.2f}")
        for code, r in radar_avoids.iterrows():
            L.append(f"- 🔴 `{code}` {r.get('name') or '--'}  **avoid**")
        L.append("")

    # 8. 题材
    if theme_sigs:
        emerging = [t for t in theme_sigs if t.stage == "emerging"]
        if emerging:
            L.append(f"## 🔥 扩散中的题材 ({len(emerging)} 个)")
            for t in emerging[:5]:
                leaders_str = ", ".join(t.leaders[:3])
                L.append(f"- **{t.theme}** (20日动量 {t.momentum_20d:+.1f}%)  "
                         f"leaders: {leaders_str}")
            L.append("")

    L.append("---")
    L.append(f"_权重: "
             + " · ".join(f"{k}={v}" for k, v in SCORE_WEIGHTS.items()) + "_")
    L.append("_由 predict_tomorrow.py 调度: Regime + ML + Watchlist v2 + "
             "Radar + Agent 4 路辩论 + PreTradeGate_")
    return "\n".join(L)


# ==================== [7] 推送 ====================
def push_lark(md: str, dry_run: bool = False) -> bool:
    if dry_run:
        log("dry-run, 不推飞书")
        return True
    USER_ID = os.environ.get("LARK_USER_OPEN_ID",
                             "ou_5be0f87dc7cec796b7ea97d0a9b5302f")
    cmd = ["lark-cli", "im", "+messages-send",
           "--as", "bot", "--user-id", USER_ID, "--markdown", md]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if p.returncode == 0:
            log("✓ 飞书已送达")
            return True
        log(f"❌ 飞书失败: {p.stderr[:200]}")
    except Exception as e:
        log(f"❌ 飞书异常: {e}")
    return False


# ==================== main ====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="不推飞书")
    ap.add_argument("--no-theme", action="store_true", help="跳 theme 扫描")
    ap.add_argument("--no-sentiment", action="store_true", help="跳 sentiment")
    ap.add_argument("--skip-agent", action="store_true",
                    help="不跑 agent 辩论, 只产综合排名")
    ap.add_argument("--top", type=int, default=10, help="辩论 top-K")
    ap.add_argument("--debate-all", action="store_true",
                    help="对全部 watchlist 跑 Agent 辩论, 忽略 --top")
    ap.add_argument("--concurrency", type=int, default=3,
                    help="Agent 辩论并发数 (qwen-turbo 建议 3-5)")
    ap.add_argument("--debate-rounds", type=int, default=0,
                    help="多轮 bull/bear 辩论轮数. 0 = 一轮判决, 1-2 = 多轮反驳")
    ap.add_argument("--fresh-cash", type=float, default=None,
                    help="用 N 元全新空仓模拟 gate 检查 (而非真账户). "
                         "推荐 1e6. 默认 None = 用真账户状态")
    ap.add_argument("--watchlist-csv",
                    default="/Users/page/Desktop/股票/股票自选.csv")
    ap.add_argument("--out-prefix", default="/tmp/predict_tomorrow")
    args = ap.parse_args()

    log("=" * 60)
    log(f"🎯 predict_tomorrow 启动")

    # [0] 加载状态
    watchlist = load_watchlist(Path(args.watchlist_csv))
    wl_codes = [w["code"] for w in watchlist]
    log(f"watchlist: {len(watchlist)} 只")
    account = load_account()
    log(f"账户: cash ¥{account.get('cash', 0):,.0f}")

    # 加载全市场 kline + union watchlist_kline (覆盖非 n500 的自选股)
    log("[0] 加载 daily kline...")
    try:
        from scripts.paper_trade_runner import load_daily_cache
        daily = load_daily_cache()
        # merge watchlist_kline (最新日期 parquet)
        wl_klines = sorted(ROOT.glob("cache/watchlist_kline_*.parquet"))
        if wl_klines:
            wl_df = pd.read_parquet(wl_klines[-1])
            if "date" in wl_df.columns:
                wl_df["date"] = pd.to_datetime(wl_df["date"])
            # 去重: 同 (code, date) 取 watchlist 版本 (因其更新频率高)
            daily = pd.concat([daily, wl_df], ignore_index=True, sort=False)
            # 强制 code 为 6 位零填字符串, 避免 int 化丢前导 0 (002025 → 2025)
            daily["code"] = daily["code"].astype(str).str.zfill(6)
            daily = daily.sort_values(["code", "date"]).drop_duplicates(
                subset=["code", "date"], keep="last"
            ).reset_index(drop=True)
            log(f"  merge watchlist_kline {wl_klines[-1].name}: "
                f"daily 扩至 {daily.shape}")

        tdays = sorted(daily["date"].unique())
        asof = pd.Timestamp("2026-04-22")
        if asof not in set(tdays):
            asof = max(d for d in tdays if d <= asof)
        log(f"daily shape {daily.shape}, asof {asof.date()}")
    except Exception as e:
        log(f"❌ daily 加载失败: {e}")
        return 1

    # [1] regime
    log("[1] RegimeDetector...")
    regime = stage_regime(daily, asof)
    log(f"  regime={regime['regime']} pos_mult={regime['position_mult']:.2f}")

    # [2] 5 类信号
    log("[2A] ML 因子全量打分 (70s)...")
    ml_pred = stage_ml_factor(asof, daily)

    log("[2B] Watchlist v2 18 因子...")
    wl_sig = stage_watchlist_v2(wl_codes)

    log("[2C] Radar long/avoid...")
    radar = stage_radar()

    log("[2D] 题材扩散...")
    theme_sigs = stage_theme(daily, asof, skip=args.no_theme)

    log("[2E] Sentiment batch...")
    # 候选池: watchlist + ML top 20 + radar long (仅这些跑 sentiment, 避免太慢)
    batch_codes = list(set(wl_codes) |
                        set(ml_pred.head(20).index.tolist() if ml_pred is not None else []) |
                        set(c["code"] for c in radar.get("longs", [])))
    sentiment_map = stage_sentiment_batch(batch_codes, skip=args.no_sentiment)

    # [3] 综合排名
    log("[3] 综合打分...")
    composed = stage_compose(watchlist, ml_pred, wl_sig, radar,
                              theme_sigs, sentiment_map)
    log(f"  候选池 {len(composed)} 只, top5:")
    for code, r in composed.head(5).iterrows():
        log(f"    {code} {r['name']:8s}  comp={r['composite']:+.3f}  "
            f"wl_z={r['wl_z']:+.2f} ml_z={r['ml_z']:+.2f} "
            f"radar_conf={r['radar_long_conf']:.2f}")

    # 保存综合排名全表
    composed.to_csv(f"{args.out_prefix}_composed.csv")

    # [4] Agent 辩论
    if args.skip_agent:
        log("[4] 跳过 agent 辩论 (--skip-agent)")
        top_decisions = [
            {"code": code, "name": row["name"], "action": "buy" if row["composite"] > 0 else "hold",
             "conviction": min(1.0, max(0.3, (row["composite"] + 1) / 2)),
             "score": float(row["composite"]), "reasoning": "综合打分", "views": {}}
            for code, row in composed.head(args.top).iterrows()
        ]
    else:
        # 决定跑哪些: --debate-all 取全 watchlist ∩ composed
        if args.debate_all:
            wl_set = {w["code"] for w in watchlist}
            target = composed[composed.index.isin(wl_set)]
            log(f"[4] --debate-all: Agent 辩论全部 watchlist "
                f"({len(target)} 只, 并发 {args.concurrency}, "
                f"rounds={args.debate_rounds})")
        else:
            target = composed.head(args.top)
            log(f"[4] Agent 4 路辩论 top {args.top} "
                f"(并发 {args.concurrency}, rounds={args.debate_rounds})...")
        try:
            top_decisions = asyncio.run(
                stage_agent_debate(
                    target, regime, account,
                    concurrency=args.concurrency,
                    debate_rounds=args.debate_rounds,
                )
            )
        except Exception as e:
            log(f"❌ Agent 辩论整体失败: {e}")
            traceback.print_exc()
            top_decisions = []

    # [5] Gate
    log(f"[5] PreTradeGate 过滤 ({len(top_decisions)} 个决策, "
        f"{'fresh cash ¥' + f'{args.fresh_cash:,.0f}' if args.fresh_cash else '真账户'})")
    top_decisions = stage_gate(top_decisions, account, daily, asof,
                                 fresh_cash=args.fresh_cash)

    # [6] 报告
    log("[6] 生成报告...")
    asof_str = str(asof.date()) if hasattr(asof, "date") else str(asof)
    md = render_report(regime, composed, top_decisions,
                        theme_sigs, wl_sig, asof_str, args)
    out_md = f"{args.out_prefix}.md"
    Path(out_md).write_text(md, encoding="utf-8")
    log(f"报告 {len(md)} 字 → {out_md}")

    # JSON 存档
    out_json = f"{args.out_prefix}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "asof": asof_str,
            "generated_at": datetime.now().isoformat(),
            "regime": regime,
            "composed_top20": composed.head(20).reset_index().to_dict(orient="records"),
            "top_decisions": top_decisions,
            "weights": SCORE_WEIGHTS,
        }, f, ensure_ascii=False, indent=2, default=str)
    log(f"JSON 存档 → {out_json}")

    # [7] 推送
    push_lark(md, dry_run=args.dry_run)

    log("=" * 60)
    log("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
