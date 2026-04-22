"""AnalystBrief → 飞书 IM 一屏 Markdown 卡片.

风格: 排序打分 + 风险提示 (中性), 禁止"建议买入/卖出"等死命令.
词表黑名单 post-process 会过滤任何漏网的动词。
"""
from __future__ import annotations

import re

from analyst.brief_builder import AnalystBrief, StockPick

# 禁用词 → 中性替代
_BANNED = [
    (re.compile(r"建议买入"), "建议关注"),
    (re.compile(r"建议卖出"), "建议减持关注"),
    (re.compile(r"满仓"), "重仓"),
    (re.compile(r"\bAll\s*in\b", re.IGNORECASE), "集中"),
    (re.compile(r"梭哈"), "加码"),
    (re.compile(r"立即买入"), "重点关注"),
    (re.compile(r"马上买"), "关注"),
]


def format_im(brief: AnalystBrief) -> str:
    lines: list[str] = []
    lines.append(f"📈 **A股分析师 · {brief.date} 盘后**")
    lines.append("")

    # 1. 市场全景
    lines.extend(_fmt_market(brief))

    # 2. 昨日命中
    review_line = _fmt_review(brief)
    if review_line:
        lines.append(review_line)

    # 3. 账户
    acct_line = _fmt_account(brief)
    if acct_line:
        lines.append(acct_line)

    lines.append("")

    # 4. 明日关注
    if brief.top_picks:
        lines.append(f"**明日关注 Top {len(brief.top_picks)}**")
        for p in brief.top_picks:
            lines.extend(_fmt_pick(p))
    else:
        lines.append("**明日关注**：今日量化层无候选(数据源缺失)")

    # 5. 风险提示
    if brief.risk_notes:
        lines.append("")
        lines.append("**风险提示**")
        for note in brief.risk_notes:
            lines.append(f"· {note}")

    # 6. 降级标记
    if brief.degraded_sources:
        lines.append("")
        lines.append(f"⚠️ 数据源缺失: {' / '.join(brief.degraded_sources)}")

    lines.append("")
    lines.append("_仅供参考，非投资建议_")

    text = "\n".join(lines)
    return _sanitize(text)


def _fmt_market(brief: AnalystBrief) -> list[str]:
    out: list[str] = []
    m = brief.market
    if m.indices:
        parts = [f"{idx['name']} {_pct(idx['pct_chg'])}" for idx in m.indices]
        out.append("**市场** " + " / ".join(parts))
    else:
        out.append("**市场** —")

    extras: list[str] = []
    if m.northbound_yi is not None:
        extras.append(f"北向 ¥{m.northbound_yi:+.1f}亿")
    else:
        extras.append("北向 —")
    if m.sectors_top5:
        top_names = " / ".join(f"{s['name']} {_pct(s['pct_chg'])}" for s in m.sectors_top5[:3])
        extras.append(f"热点 {top_names}")
    if extras:
        out.append(" · ".join(extras))

    if m.regime:
        out.append(f"_节奏：{m.regime}_")
    return out


def _fmt_review(brief: AnalystBrief) -> str:
    r = brief.review
    bits: list[str] = []
    if r.v2_long_hit is not None:
        bits.append(f"多头命中 {r.v2_long_hit:.0f}%")
    if r.v2_long_short_spread is not None:
        bits.append(f"多空 {r.v2_long_short_spread:+.2f}%")
    if r.llm_conviction_spearman is not None:
        bits.append(f"LLM Spearman {r.llm_conviction_spearman:+.2f}")
    if not bits:
        return ""
    return "**昨日复盘** " + " · ".join(bits)


def _fmt_account(brief: AnalystBrief) -> str:
    a = brief.account
    if a.nav_wan is None:
        return ""
    ret = f" ({a.total_ret_pct:+.1f}%)" if a.total_ret_pct is not None else ""
    return f"**账户** ¥{a.nav_wan:.1f}万{ret} · 持仓 {a.position_count}只"


def _fmt_pick(p: StockPick) -> list[str]:
    cat = f"{p.top_category}{p.cat_sign}" if p.top_category else ""
    conv = _conv_tag(p.conviction)
    action_tag = _action_tag(p.action)
    header = f"{p.rank}. `{p.code} {p.name}` 打分 {p.alpha_z:.2f} · {cat}{conv}{action_tag}"
    out = [header]

    # 入场/止损/目标
    price_bits = [f"现价 ¥{p.latest_close:.2f}"]
    if p.stop_loss:
        price_bits.append(f"止损 ¥{p.stop_loss:.2f}")
    if p.take_profit:
        price_bits.append(f"目标 ¥{p.take_profit:.2f}")
    if p.holding_days:
        price_bits.append(f"{p.holding_days}日")
    out.append("   · " + " / ".join(price_bits))

    if p.reason:
        out.append(f"   · 逻辑：{_truncate(p.reason, 70)}")
    if p.risk:
        out.append(f"   · 风险：{_truncate(p.risk, 60)}")
    return out


def _conv_tag(conv) -> str:
    if conv is None:
        return ""
    if conv >= 0.7:
        return " · 信心高"
    if conv >= 0.5:
        return " · 信心中"
    return " · 信心低"


def _action_tag(action) -> str:
    m = {"buy": " · 偏多", "watch": " · 观察", "avoid": " · 回避"}
    return m.get(action or "", "")


def _pct(v) -> str:
    try:
        return f"{float(v):+.2f}%"
    except (ValueError, TypeError):
        return "—"


def _truncate(s: str, n: int) -> str:
    s = s.strip().replace("\n", " ")
    return s if len(s) <= n else s[:n] + "…"


def _sanitize(text: str) -> str:
    out = text
    for pat, repl in _BANNED:
        out = pat.sub(repl, out)
    return out


if __name__ == "__main__":
    import argparse
    from datetime import datetime
    from analyst.brief_builder import build
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"))
    ap.add_argument("--top-n", type=int, default=3)
    args = ap.parse_args()
    brief = build(args.date, top_n=args.top_n)
    print(format_im(brief))
