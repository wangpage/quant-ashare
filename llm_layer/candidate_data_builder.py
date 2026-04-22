"""给 TradingAgentTeam.decide_async 组装真实数据 blob.

现状: daily_trading.py:121-127 里 fundamentals/kline/sentiment_data 都是占位字符串.
本模块给每只股票构造 5 个字段的真数据文本, 让 4 路分析师能真正分析.

调用方:
  >>> from llm_layer.candidate_data_builder import build_data_blob
  >>> blob = build_data_blob(code='300750', name='宁德时代')
  >>> # blob = {fundamentals, kline, indicators, sentiment_data, radar_summary}

每个字段都独立降级, 不会因一个数据源失败而整个崩溃.
"""
from __future__ import annotations

from typing import Any

from memory.storage import MemoryStore
from utils.logger import logger

from .market_context import build_context_text, compute_indicators


def _build_indicators_text(code: str) -> str:
    """把 compute_indicators 的 dict 翻译成技术指标描述文本."""
    try:
        ind = compute_indicators(code)
    except Exception as e:
        return f"技术指标查询失败: {e}"

    if ind.get("bars_in_cache", 0) == 0:
        return "该股无本地 K 线数据, 无法计算技术指标"

    parts = []
    close = ind.get("close")
    if close is not None:
        parts.append(f"收盘 {close}")

    # 趋势
    r5 = ind.get("ret_5d_pct")
    r20 = ind.get("ret_20d_pct")
    r60 = ind.get("ret_60d_pct")
    if r5 is not None:
        parts.append(f"5日 {r5:+.2f}%")
    if r20 is not None:
        parts.append(f"20日 {r20:+.2f}%")
    if r60 is not None:
        parts.append(f"60日 {r60:+.2f}%")

    # 超额
    ex = ind.get("excess_vs_hs300_20d_pct")
    if ex is not None:
        parts.append(f"20日超额HS300 {ex:+.2f}%")

    # 位置
    p250 = ind.get("pos_in_250d_range")
    p60 = ind.get("pos_in_60d_range")
    if p250 is not None:
        pct = int(p250 * 100)
        cue = "近历史高点" if pct >= 90 else ("近历史低点" if pct <= 10 else "中位")
        parts.append(f"250日区间 {pct}% ({cue})")
    if p60 is not None:
        parts.append(f"60日区间 {int(p60*100)}%")

    if ind.get("at_near_250d_high"):
        parts.append("⚠️ 接近 250 日高点")

    # 量能
    vr = ind.get("vol_ratio_5d")
    if vr is not None:
        cue = "放量" if vr >= 1.5 else ("缩量" if vr <= 0.7 else "常量")
        parts.append(f"量比 {vr:.2f}x ({cue})")

    amt = ind.get("amount_yi_latest")
    rank = ind.get("amount_rank_in_20d")
    if amt is not None:
        tag = f" 近20日TOP{rank}" if rank and rank <= 3 else ""
        parts.append(f"成交 {amt:.2f}亿{tag}")

    consec = ind.get("consec_up_days", 0)
    if consec >= 3:
        parts.append(f"连涨 {consec} 日")

    return " | ".join(parts)


def _build_fundamentals_text(code: str, name: str) -> str:
    """基本面文本. 复用 data_adapter.fundamentals. akshare 限速时降级到 '不可得'."""
    try:
        from data_adapter.fundamentals import fetch_fundamentals_text
        return fetch_fundamentals_text(code, name_hint=name)
    except Exception as e:
        logger.warning(f"fundamentals 调用失败 {code}: {e}")
        return f"{code} {name} (基本面查询失败)"


def _build_sentiment_text(code: str, name: str,
                           backend: str = "snownlp") -> str:
    """舆情面文本. 用 SnowNLP 对 akshare stock_news_em 打分.

    akshare 挂了就降级到"无舆情信号".
    """
    try:
        from llm_layer.sentiment import NewsSentimentAnalyzer
        analyzer = NewsSentimentAnalyzer(backend=backend)
        s = analyzer.score(code)
        if s is None or s.sample_size == 0:
            return f"{code} {name}: 当日无舆情样本"
        sentiment_tag = ("偏多" if s.score > 0.15
                          else "偏空" if s.score < -0.15 else "中性")
        return (f"{code} {name}: 舆情 {sentiment_tag} (score={s.score:+.2f}, "
                f"conf={s.confidence:.2f}, 样本 {s.sample_size} 条, "
                f"来源 {s.backend})")
    except Exception as e:
        logger.debug(f"sentiment 失败 {code}: {e}")
        return f"{code} {name}: 舆情数据不可得"


def _build_radar_summary(code: str, name: str,
                          store: MemoryStore | None = None) -> str:
    """事件面文本. 复用 radar_events_helper."""
    try:
        from .radar_events_helper import build_radar_summary_for_code
        return build_radar_summary_for_code(
            code, name, since_hours=72, store=store,
        )
    except Exception as e:
        logger.warning(f"radar_summary 失败 {code}: {e}")
        return f"{code} {name}: 事件数据查询失败"


def build_data_blob(
    code: str, name: str = "",
    factor_score: float = 0.0,
    store: MemoryStore | None = None,
    sentiment_backend: str = "snownlp",
    include_sentiment: bool = True,
) -> dict[str, Any]:
    """一次性构造 decide_async 需要的全部字段.

    Args:
        code, name: 标的
        factor_score: ML 因子打分 (透传, agent 技术面分析师直接用)
        store: MemoryStore (若 None 则新建)
        sentiment_backend: "snownlp" (fast free) / "anthropic" / "openai" / "qwen"
        include_sentiment: False 时 sentiment_data 填占位避免网络拖慢

    Returns:
        {
            'code', 'name',
            'fundamentals',      # str
            'kline',             # str (build_context_text 的单行)
            'indicators',        # str (展开的技术指标描述)
            'sentiment_data',    # str
            'radar_summary',     # str (近 72h 相关事件)
            'factor_score',      # float
        }
    """
    store = store or MemoryStore()
    blob = {
        "code": code,
        "name": name,
        "factor_score": factor_score,
        "fundamentals": _build_fundamentals_text(code, name),
        "kline": build_context_text(code),
        "indicators": _build_indicators_text(code),
        "radar_summary": _build_radar_summary(code, name, store=store),
    }
    if include_sentiment:
        blob["sentiment_data"] = _build_sentiment_text(code, name, sentiment_backend)
    else:
        blob["sentiment_data"] = f"{code} {name}: 舆情分析已跳过 (--no-sentiment)"
    return blob


def build_data_blobs_batch(
    items: list[dict], store: MemoryStore | None = None,
    sentiment_backend: str = "snownlp",
    include_sentiment: bool = True,
) -> list[dict]:
    """批量, items 必须是 [{code, name, factor_score?}, ...]."""
    store = store or MemoryStore()
    out = []
    for it in items:
        out.append(build_data_blob(
            code=it["code"],
            name=it.get("name", ""),
            factor_score=float(it.get("factor_score", 0.0) or 0.0),
            store=store,
            sentiment_backend=sentiment_backend,
            include_sentiment=include_sentiment,
        ))
    return out


if __name__ == "__main__":
    import sys, json as _j
    codes = sys.argv[1:] or ["300750:宁德时代", "603163:圣晖集成"]
    for spec in codes:
        parts = spec.split(":", 1)
        code = parts[0]
        name = parts[1] if len(parts) > 1 else ""
        blob = build_data_blob(code, name, include_sentiment=False)
        print("=" * 60)
        print(f"{code} {name}")
        for k in ("fundamentals", "kline", "indicators",
                  "sentiment_data", "radar_summary"):
            print(f"\n[{k}]")
            print(blob[k][:400])
