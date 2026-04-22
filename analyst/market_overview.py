"""市场全景业务整形: 把 data_adapter 的原始数据 → MarketSnapshot."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from data_adapter import market_overview as _mo
from utils.config import PROJECT_ROOT


@dataclass
class MarketSnapshot:
    date: str
    indices: list[dict] = field(default_factory=list)         # [{name, close, pct_chg}]
    northbound_yi: Optional[float] = None                     # 亿元净流入, None 表未更新
    sectors_top5: list[dict] = field(default_factory=list)    # [{name, pct_chg}]
    regime: Optional[str] = None                              # 来自 output/regime_summary.txt

    @property
    def has_data(self) -> bool:
        return bool(self.indices) or bool(self.sectors_top5) or self.northbound_yi is not None


def get_snapshot() -> MarketSnapshot:
    raw = _mo.fetch_all(persist=True)
    snap = MarketSnapshot(date=raw["date"])

    if raw["indices"] is not None and not raw["indices"].empty:
        snap.indices = [
            {"name": r["name"], "close": float(r["close"]), "pct_chg": float(r["pct_chg"])}
            for _, r in raw["indices"].iterrows()
        ]

    if raw["northbound"] is not None:
        snap.northbound_yi = float(raw["northbound"]["net_inflow_yi"])

    if raw["sectors"] is not None and not raw["sectors"].empty:
        snap.sectors_top5 = [
            {"name": r["name"], "pct_chg": float(r["pct_chg"])}
            for _, r in raw["sectors"].iterrows()
        ]

    snap.regime = _read_regime()
    return snap


def _read_regime() -> Optional[str]:
    """读 regime_summary.txt 首行, 抽出核心字段拼成简洁描述.

    原始行形如: [2026-04-20 19:59:44] bull_trending | 信心 17% | 仓位 1.00 | trend=up | ...
    清洗后: bull_trending · 信心 17% · 仓位 1.00
    """
    path = PROJECT_ROOT / "output" / "regime_summary.txt"
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return None
        first = text.split("\n")[0]
        # 去掉时间戳前缀
        import re
        first = re.sub(r"^\[.*?\]\s*", "", first)
        parts = [p.strip() for p in first.split("|") if p.strip()]
        # 只取前 3 段(状态/信心/仓位)
        keep = parts[:3]
        cleaned = " · ".join(keep)
        return cleaned[:80] if cleaned else None
    except Exception:
        return None


if __name__ == "__main__":
    s = get_snapshot()
    print(s)
