"""核心聚合: watchlist_signals_v2 + llm_shortline + review + account + market_overview
→ AnalystBrief (供 im_formatter 消费)."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from analyst.market_overview import MarketSnapshot, get_snapshot
from utils.config import PROJECT_ROOT
from utils.logger import logger

_V2_DIR = PROJECT_ROOT / "output" / "paper_trade" / "watchlist_signals_v2"
_LLM_DIR = PROJECT_ROOT / "output" / "paper_trade" / "llm_shortline"
_REVIEW_DIR = PROJECT_ROOT / "output" / "review"
_ACCOUNT = PROJECT_ROOT / "output" / "paper_trade" / "account.json"


@dataclass
class StockPick:
    code: str
    name: str
    rank: int
    alpha_z: float
    top_category: str           # 打板/席位/反转/板块 等
    cat_sign: str               # "+" 或 "-"
    latest_close: float
    action: Optional[str] = None           # buy/watch/avoid (来自 llm_shortline)
    conviction: Optional[float] = None     # 0-1
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    holding_days: Optional[int] = None
    reason: str = ""
    risk: str = ""


@dataclass
class ReviewStats:
    v2_ic: Optional[float] = None
    v2_long_hit: Optional[float] = None          # 看多命中率 %
    v2_long_short_spread: Optional[float] = None # 多空价差 %
    llm_conviction_spearman: Optional[float] = None


@dataclass
class AccountSummary:
    nav_wan: Optional[float] = None
    total_ret_pct: Optional[float] = None
    position_count: int = 0


@dataclass
class AnalystBrief:
    date: str
    market: MarketSnapshot
    top_picks: list[StockPick] = field(default_factory=list)
    review: ReviewStats = field(default_factory=ReviewStats)
    account: AccountSummary = field(default_factory=AccountSummary)
    risk_notes: list[str] = field(default_factory=list)
    degraded_sources: list[str] = field(default_factory=list)  # 缺失/降级的数据源


def build(date: str, top_n: int = 3) -> AnalystBrief:
    """构建分析师简报.

    Args:
        date: YYYY-MM-DD
        top_n: 推送的精选股数量 (3-5)
    """
    brief = AnalystBrief(date=date, market=get_snapshot())

    if not brief.market.has_data:
        brief.degraded_sources.append("市场全景")

    v2_df = _load_v2(date)
    if v2_df is None:
        brief.degraded_sources.append("watchlist_signals_v2")
    else:
        llm_df = _load_llm(date)
        if llm_df is None:
            brief.degraded_sources.append("llm_shortline")
        brief.top_picks = _merge_picks(v2_df, llm_df, top_n=top_n)

    brief.review = _parse_latest_review()
    brief.account = _read_account()
    brief.risk_notes = _extract_risk_notes(brief)
    return brief


def _load_v2(date: str) -> Optional[pd.DataFrame]:
    path = _V2_DIR / f"{date}.csv"
    if not path.exists():
        # 回退: 取最新一份(防周末/节假日无产出)
        candidates = sorted(_V2_DIR.glob("*.csv"))
        if not candidates:
            logger.warning(f"brief_builder: v2 目录无产物")
            return None
        path = candidates[-1]
        logger.info(f"brief_builder: {date} 无 v2, 回退到 {path.name}")
    try:
        return pd.read_csv(path, dtype={"code": str})
    except Exception as e:
        logger.warning(f"brief_builder: 读 v2 失败 {e}")
        return None


def _load_llm(date: str) -> Optional[pd.DataFrame]:
    path = _LLM_DIR / f"{date}.csv"
    if not path.exists():
        candidates = sorted(_LLM_DIR.glob("*.csv"))
        if not candidates:
            return None
        path = candidates[-1]
        logger.info(f"brief_builder: {date} 无 llm_shortline, 回退到 {path.name}")
    try:
        # 处理 BOM, code 保留字符串形态
        return pd.read_csv(path, encoding="utf-8-sig", dtype={"code": str})
    except Exception as e:
        logger.warning(f"brief_builder: 读 llm 失败 {e}")
        return None


def _merge_picks(v2_df: pd.DataFrame, llm_df: Optional[pd.DataFrame],
                  top_n: int) -> list[StockPick]:
    v2_sorted = v2_df.sort_values("alpha_z", ascending=False).head(top_n).reset_index(drop=True)
    llm_map: dict[str, dict] = {}
    if llm_df is not None:
        for _, r in llm_df.iterrows():
            code = str(r.get("code", "")).strip()
            if code:
                llm_map[code] = r.to_dict()

    picks: list[StockPick] = []
    for i, row in v2_sorted.iterrows():
        code = str(row["code"])
        llm_rec = llm_map.get(code, {})
        picks.append(StockPick(
            code=code,
            name=str(row.get("name", "")),
            rank=i + 1,
            alpha_z=float(row.get("alpha_z", 0) or 0),
            top_category=str(row.get("top_category", "")),
            cat_sign=str(row.get("cat_sign", "")),
            latest_close=float(row.get("latest_close", 0) or 0),
            action=_safe_str(llm_rec.get("action")),
            conviction=_safe_float(llm_rec.get("conviction")),
            stop_loss=_safe_float(llm_rec.get("stop_loss")),
            take_profit=_safe_float(llm_rec.get("take_profit")),
            holding_days=_safe_int(llm_rec.get("holding")),
            reason=_safe_str(llm_rec.get("reason")) or "",
            risk=_safe_str(llm_rec.get("risk")) or "",
        ))
    return picks


def _safe_float(v) -> Optional[float]:
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        s = str(v).strip()
        if not s or s.lower() == "nan":
            return None
        return float(s)
    except (ValueError, TypeError):
        return None


def _safe_int(v) -> Optional[int]:
    f = _safe_float(v)
    return int(f) if f is not None else None


def _safe_str(v) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    s = str(v).strip()
    return s if s and s.lower() != "nan" else None


_IC_ROW_RE = re.compile(r"\|\s*v2\s*\|\s*\d+\s*\|\s*([-+\d.]+)\s*\|\s*[-+\d.]+\s*\|\s*\+?([-\d.]+)%")
_SPREAD_RE = re.compile(r"\|\s*v2\s*\|.*\|\s*([-+\d.]+)%\s*\|")
_SPEARMAN_RE = re.compile(r"conviction→收益\s*Spearman\s*=\s*([-+\d.]+)")


def _parse_latest_review() -> ReviewStats:
    stats = ReviewStats()
    if not _REVIEW_DIR.exists():
        return stats
    candidates = sorted(_REVIEW_DIR.glob("*.md"))
    if not candidates:
        return stats
    try:
        text = candidates[-1].read_text(encoding="utf-8")
    except Exception:
        return stats

    m = _IC_ROW_RE.search(text)
    if m:
        try:
            stats.v2_ic = float(m.group(1))
            stats.v2_long_hit = float(m.group(2))
        except ValueError:
            pass

    # 多空价差: v2 行最右侧百分比(多空价差前一列)
    for line in text.split("\n"):
        if line.strip().startswith("| v2 |") or "| v2 |" in line:
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if len(parts) >= 7:
                tail = parts[-2].rstrip("%").strip()
                try:
                    stats.v2_long_short_spread = float(tail)
                except ValueError:
                    pass
            break

    m = _SPEARMAN_RE.search(text)
    if m:
        try:
            stats.llm_conviction_spearman = float(m.group(1))
        except ValueError:
            pass
    return stats


def _read_account() -> AccountSummary:
    if not _ACCOUNT.exists():
        return AccountSummary()
    try:
        data = json.loads(_ACCOUNT.read_text(encoding="utf-8"))
        eq = data.get("equity_history", [])
        nav = eq[-1]["nav"] if eq else data.get("initial_cash", 0)
        nav0 = data.get("initial_cash", 1e6)
        return AccountSummary(
            nav_wan=nav / 1e4,
            total_ret_pct=(nav - nav0) / nav0 * 100 if nav0 else None,
            position_count=len(data.get("positions", {}) or {}),
        )
    except Exception:
        return AccountSummary()


def _extract_risk_notes(brief: AnalystBrief) -> list[str]:
    """规则式抽取关键风险点 (非 LLM)."""
    notes: list[str] = []

    # 大盘逆风
    if brief.market.indices:
        sh = next((x for x in brief.market.indices if x["name"] == "上证指数"), None)
        if sh and sh["pct_chg"] < -0.8:
            notes.append(f"大盘逆风：上证 {sh['pct_chg']:.2f}%，谨慎追高")

    # high conviction 但 low alpha_z (模型-LLM 背离)
    for p in brief.top_picks:
        if p.conviction and p.conviction >= 0.7 and p.alpha_z < 1.5:
            notes.append(f"{p.name} LLM 高信心但因子打分偏低，存在分歧")

    # LLM action 分布失衡
    actions = [p.action for p in brief.top_picks if p.action]
    if actions and actions.count("avoid") >= len(actions) / 2:
        notes.append("候选池半数以上被 LLM 判为回避，今日可空仓观望")

    # regime
    if brief.market.regime and any(k in brief.market.regime for k in ["冰点", "熊", "下降"]):
        notes.append(f"市场节奏：{brief.market.regime}")

    # review 昨日命中率偏低
    if brief.review.v2_long_hit is not None and brief.review.v2_long_hit < 45:
        notes.append(f"昨日看多命中率 {brief.review.v2_long_hit:.0f}%，模型短期漂移")

    return notes[:5]  # 最多 5 条


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"))
    ap.add_argument("--top-n", type=int, default=3)
    args = ap.parse_args()
    b = build(args.date, top_n=args.top_n)
    print(json.dumps({
        "date": b.date,
        "picks": [p.__dict__ for p in b.top_picks],
        "review": b.review.__dict__,
        "account": b.account.__dict__,
        "risk_notes": b.risk_notes,
        "degraded": b.degraded_sources,
        "market_indices": b.market.indices,
        "market_northbound": b.market.northbound_yi,
        "market_sectors": b.market.sectors_top5,
    }, ensure_ascii=False, indent=2, default=str))
