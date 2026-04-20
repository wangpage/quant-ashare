"""Phase 2 测试: Memory Curator + Regime Detector.

覆盖:
  T1. MemoryStore 增删查改 + FTS5 全文检索
  T2. MemoryCurator 反思 (fallback 模式) + 每周 nudge (需 mock LLM)
  T3. SkillFactory 簇聚合 + 提炼
  T4. RegimeDetector 7 种 regime 分类 + 仓位乘数
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from tabulate import tabulate

from memory import MemoryStore, MemoryCurator, SkillFactory
from memory.curator import TradeOutcome
from market_regime import RegimeDetector, MarketRegime


def _a(cond, msg): return bool(cond), msg


# ==================== T1: MemoryStore ====================
def test_memory_store():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    store = MemoryStore(db_path=tmp.name)

    id1 = store.add(kind="reflection",
                    content="连板3天后次日低开3%应当观望",
                    code="300750", outcome_pnl=-0.04,
                    metadata={"regime": "bull_trending"})
    id2 = store.add(kind="rule",
                    content="大盘跌破20日均线减仓到30%",
                    metadata={"source": "manual"})
    id3 = store.add(kind="reflection",
                    content="涨停板次日放量该持股",
                    code="300750", outcome_pnl=0.08,
                    metadata={"regime": "bull_trending"})

    recent = store.recent(kind="reflection", limit=10)
    fts_results = store.search("连板", limit=5)

    stats = store.stats()
    return [
        _a(id1 > 0, "插入返回 id"),
        _a(len(recent) == 2, f"recent 拉 2 条反思, 实际 {len(recent)}"),
        _a(len(fts_results) >= 1, f"FTS5 搜'连板'命中, 实际 {len(fts_results)}"),
        _a(stats["memories_total"] == 3, "总数"),
        _a(stats["by_kind"].get("reflection") == 2, "按类统计"),
    ]


def test_memory_search_chinese():
    """中文 FTS5 关键字检索."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    store = MemoryStore(db_path=tmp.name)
    store.add(kind="reflection", content="宁德时代动力电池产能过剩, 价格战", code="300750")
    store.add(kind="reflection", content="贵州茅台批价下跌, 白酒周期拐点", code="600519")
    r1 = store.search("动力电池")
    r2 = store.search("白酒")
    return [
        _a(len(r1) == 1, f"动力电池 命中 1, 实际 {len(r1)}"),
        _a(len(r2) == 1, f"白酒 命中 1, 实际 {len(r2)}"),
    ]


# ==================== T2: MemoryCurator ====================
def test_curator_fallback():
    """LLM 不可用时 fallback 保底."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    store = MemoryStore(db_path=tmp.name)
    c = MemoryCurator(llm_backend=None, store=store)

    profit = TradeOutcome(
        code="300750", entry_date="2026-04-10", exit_date="2026-04-13",
        entry_price=250.0, exit_price=270.0, shares=100,
        holding_days=3, pnl_pct=0.08,
        entry_reasoning="量价突破", exit_trigger="止盈",
        market_regime="bull_trending",
    )
    id_profit = c.reflect_on_trade(profit)

    loss = TradeOutcome(
        code="600519", entry_date="2026-04-10", exit_date="2026-04-11",
        entry_price=1500.0, exit_price=1440.0, shares=100,
        holding_days=1, pnl_pct=-0.04,
        entry_reasoning="追高", exit_trigger="止损",
        market_regime="bull_quiet",
    )
    id_loss = c.reflect_on_trade(loss)

    recs = store.recent(kind="reflection")
    return [
        _a(id_profit > 0 and id_loss > 0, "两笔反思都成功存入"),
        _a(len(recs) == 2, "共 2 条"),
        _a(any("盈利" in r.content for r in recs), "盈利有对应反思"),
        _a(any("亏损" in r.content for r in recs), "亏损有对应反思"),
        _a(all(r.metadata.get("fallback") for r in recs), "标记了 fallback"),
    ]


def test_curator_with_mock_llm():
    """Mock LLM 返回 Hermes XML, 验证抽取."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    store = MemoryStore(db_path=tmp.name)

    def mock_llm(prompt):
        return """<THINKING>盈利来自趋势跟随</THINKING>
<REASONING>量能配合, 板块共振</REASONING>
<REFLECTION>
[bull_trending中连板次日放量] -> [敢于持仓至第3日]
[量能不到昨日1.5倍] -> [减仓或离场]
</REFLECTION>
<SCORE>8</SCORE>
<SOLUTION>趋势跟随成功案例</SOLUTION>"""

    c = MemoryCurator(llm_backend=mock_llm, store=store)
    trade = TradeOutcome(
        code="300750", entry_date="2026-04-10", exit_date="2026-04-13",
        entry_price=250.0, exit_price=270.0, shares=100,
        holding_days=3, pnl_pct=0.08,
        entry_reasoning="量价突破",
        exit_trigger="止盈",
        market_regime="bull_trending",
    )
    id_ = c.reflect_on_trade(trade)
    rec = store.recent(kind="reflection")[0]

    return [
        _a(id_ > 0, "写入成功"),
        _a("连板次日放量" in rec.content, "抽取到 REFLECTION"),
        _a(rec.metadata.get("quality_score") == 8.0, "抽取到质量分"),
        _a(rec.metadata.get("regime") == "bull_trending", "保存 regime"),
    ]


def test_weekly_nudge():
    """每周整合: 多条反思合并成规则."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    store = MemoryStore(db_path=tmp.name)

    # 预先塞 5 条反思
    for i in range(5):
        store.add(kind="reflection",
                  content=f"反思{i}: 跌破20日均线减仓",
                  outcome_pnl=-0.02)

    def mock_llm(prompt):
        return """<THINKING>重复出现的模式</THINkING>
<REFLECTION>
跌破20日均线 -> 减仓到30%
涨停板放量次日低开3%+ -> 不追
</REFLECTION>
<SOLUTION>
跌破20日均线应该减仓到30%
涨停板放量次日低开3%+不应追入
大盘下跌日避免新开仓
</SOLUTION>"""

    c = MemoryCurator(llm_backend=mock_llm, store=store)
    new_ids = c.weekly_nudge(days=7)
    rules = store.recent(kind="rule", days=1)

    return [
        _a(len(new_ids) >= 1, f"生成至少1条规则, 实际 {len(new_ids)}"),
        _a(len(rules) == len(new_ids), "数量匹配"),
        _a(any("减仓" in r.content for r in rules), "规则内容含减仓"),
    ]


# ==================== T3: SkillFactory ====================
def test_skill_factory():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    store = MemoryStore(db_path=tmp.name)

    # 塞 12 条同模式反思: bull_trending + short hold + up direction
    # 其中 8 条盈利, 4 条亏损 -> 胜率 8/12 = 67% 达标
    for i in range(8):
        store.add(kind="reflection",
                  content=f"成功案例{i}: 放量突破追入",
                  outcome_pnl=0.06,
                  metadata={"regime": "bull_trending", "holding_days": 2})
    for i in range(4):
        store.add(kind="reflection",
                  content=f"失败案例{i}: 放量假突破",
                  outcome_pnl=-0.03,
                  metadata={"regime": "bull_trending", "holding_days": 2})

    factory = SkillFactory(llm_backend=None, store=store,
                           min_samples=5, min_success_rate=0.5)
    # 先触发簇
    skills_generated = factory.generate_skills(days=30)

    # 再召回
    ctx = factory.recall_skills_for_context(regime="bull_trending",
                                            horizon_days=3)

    return [
        _a(len(skills_generated) >= 1, f"生成至少 1 个 skill, 实际 {len(skills_generated)}"),
        _a(any(s.success_rate >= 0.5 for s in skills_generated), "胜率达标"),
        _a("bull_trending" in ctx or "无" in ctx, "上下文可用"),
    ]


# ==================== T4: RegimeDetector ====================
def _synth_index(trend: str = "up", n: int = 250,
                 vol: float = 0.012) -> pd.DataFrame:
    """合成指数 DataFrame, 用于测试."""
    np.random.seed(42)
    rng = np.random.default_rng(42)
    if trend == "up":
        drift = 0.0020                   # 50% 年化, 趋势清晰
    elif trend == "down":
        drift = -0.0020
    elif trend == "crash":
        drift = -0.0005
    else:
        drift = 0.0

    rets = rng.normal(drift, vol, n)
    if trend == "crash":
        rets[-1] = -0.07
        rets[-5:-1] = -0.015
    close = 3000 * np.cumprod(1 + rets)
    df = pd.DataFrame({
        "close": close,
        "high": close * (1 + rng.uniform(0, 0.01, n)),
        "low": close * (1 - rng.uniform(0, 0.01, n)),
    })
    return df


def _synth_stocks(pct_up: float = 0.55,
                  pct_limit_up: float = 0.005) -> pd.DataFrame:
    np.random.seed(1)
    n = 4000
    n_up = int(n * pct_up)
    n_lim = int(n * pct_limit_up)
    pct = np.concatenate([
        np.random.uniform(0, 5, n_up - n_lim),
        np.full(n_lim, 10.0),                    # 涨停
        np.random.uniform(-5, 0, n - n_up),
    ])
    np.random.shuffle(pct)
    return pd.DataFrame({"pct_chg": pct})


def test_regime_bull():
    d = RegimeDetector()
    idx = _synth_index(trend="up", vol=0.010)
    stocks = _synth_stocks(pct_up=0.60, pct_limit_up=0.008)
    sig = d.detect(idx, stocks, total_turnover_yi=9000)
    return [
        _a(sig.regime in (MarketRegime.BULL_TRENDING, MarketRegime.BULL_QUIET),
           f"上升趋势 regime 实际 {sig.regime.value}"),
        _a(sig.position_mult >= 0.7, f"仓位乘数 {sig.position_mult}"),
        _a(sig.allow_new_long, "允许做多"),
    ]


def test_regime_bear():
    d = RegimeDetector()
    idx = _synth_index(trend="down", vol=0.018)
    stocks = _synth_stocks(pct_up=0.30, pct_limit_up=0.001)
    sig = d.detect(idx, stocks, total_turnover_yi=5000)
    return [
        _a(sig.regime in (MarketRegime.BEAR_TRENDING, MarketRegime.BEAR_QUIET),
           f"下降 regime 实际 {sig.regime.value}"),
        _a(sig.position_mult <= 0.5, f"仓位压降 {sig.position_mult}"),
    ]


def test_regime_crash():
    d = RegimeDetector()
    idx = _synth_index(trend="crash")
    sig = d.detect(idx)
    return [
        _a(sig.regime == MarketRegime.CRASH, f"识别崩盘 实际 {sig.regime.value}"),
        _a(sig.position_mult == 0.0, "禁止开仓"),
        _a(not sig.allow_new_long, "不允许做多"),
    ]


def test_regime_euphoria():
    d = RegimeDetector()
    idx = _synth_index(trend="up", vol=0.020)
    stocks = _synth_stocks(pct_up=0.80, pct_limit_up=0.04)
    sig = d.detect(idx, stocks, total_turnover_yi=13000)
    return [
        _a(sig.regime == MarketRegime.EUPHORIA,
           f"狂热 实际 {sig.regime.value}"),
        _a(sig.position_mult <= 0.5, "狂热仓位应受压降"),
    ]


def test_regime_agent_context():
    d = RegimeDetector()
    sig = d.detect(_synth_index(trend="up"),
                   _synth_stocks(pct_up=0.6),
                   total_turnover_yi=9000)
    ctx = sig.to_agent_context()
    return [
        _a("regime=" in ctx, "context 格式 regime="),
        _a("仓位乘数" in ctx, "context 格式 仓位"),
        _a("赚钱效应" in ctx, "context 格式 赚钱效应"),
    ]


def main():
    all_rows = []
    total_pass = total = 0
    for name, fn in [
        ("memory_store", test_memory_store),
        ("memory_fts_chinese", test_memory_search_chinese),
        ("curator_fallback", test_curator_fallback),
        ("curator_with_llm", test_curator_with_mock_llm),
        ("weekly_nudge", test_weekly_nudge),
        ("skill_factory", test_skill_factory),
        ("regime_bull", test_regime_bull),
        ("regime_bear", test_regime_bear),
        ("regime_crash", test_regime_crash),
        ("regime_euphoria", test_regime_euphoria),
        ("regime_context", test_regime_agent_context),
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

    print("\n==== Phase 2: Memory Curator + Regime Detector ====")
    print(tabulate(all_rows, headers=["模块", "用例", "结果"]))
    print(f"\n通过率: {total_pass}/{total}")
    sys.exit(0 if total_pass == total else 1)


if __name__ == "__main__":
    main()
