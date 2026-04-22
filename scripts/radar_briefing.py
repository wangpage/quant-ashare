"""Radar 情报简报: 查今日 radar_event + analysis, 生成 markdown, 飞书推送.

用法:
  python scripts/radar_briefing.py                      # 推送到默认 LARK_USER
  python scripts/radar_briefing.py --dry-run            # 只打印 markdown, 不发
  python scripts/radar_briefing.py --hours 6            # 只看最近 6h (默认从今天 9:00)
  python scripts/radar_briefing.py --min-conf 0.6       # 候选置信度阈值
  python scripts/radar_briefing.py --top 15             # 候选清单上限
  python scripts/radar_briefing.py --to ou_xxx          # 指定飞书 user_id

调度建议:
  - 盘中复盘: 12:00, 14:30 各跑一次 (cron 加两行)
  - 盘后总结: 15:30 跑一次, 作为 cron_daily 的补充
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_env = ROOT / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

from utils.logger import logger  # noqa: E402
from memory.storage import MemoryStore  # noqa: E402


DEFAULT_LARK_USER = "ou_5be0f87dc7cec796b7ea97d0a9b5302f"


# ==================== 数据聚合 ====================
def today_open_ts() -> int:
    """今天 9:00 本地时间的 Unix 秒."""
    now = datetime.now()
    open_dt = now.replace(hour=9, minute=0, second=0, microsecond=0)
    return int(open_dt.timestamp())


def collect(since_ts: int, limit: int = 500) -> dict:
    """从 memory.db 拉 radar 事件 + analysis, 做聚合."""
    store = MemoryStore()
    events = store.query_radar_events(since_ts=since_ts, limit=limit)

    agg = {
        "total": len(events),
        "analyzed": 0,
        "deep_analyzed": 0,
        "by_type": Counter(),
        "by_tradability": Counter(),
        "candidates": [],      # [{code, name, direction, conf, half_life, event_id,
                               #   event_title, event_ts, thesis, entry, risk_flags}]
        "supply_chains": [],   # [{event_id, title, chain}]
        "noise_ids": [],       # 可以简单统计数
        "ts_start": since_ts,
        "ts_end": int(time.time()),
    }

    for e in events:
        a = e.metadata.get("analysis") or {}
        tr = a.get("triage") or {}
        if not tr:
            continue
        agg["analyzed"] += 1
        agg["by_type"][tr.get("event_type", "?")] += 1
        agg["by_tradability"][tr.get("tradability", "?")] += 1

        if tr.get("event_type") == "noise":
            agg["noise_ids"].append(e.id)
            continue

        d = a.get("deep") or {}
        if not d:
            continue
        agg["deep_analyzed"] += 1

        chain = d.get("supply_chain", "")
        if chain:
            agg["supply_chains"].append({
                "event_id": e.id,
                "title": e.content[:60],
                "chain": chain,
                "thesis": d.get("thesis", ""),
                "event_ts": e.ts,
            })

        for t in d.get("targets", []) or []:
            agg["candidates"].append({
                "code": t.get("code") or "--",
                "name": t.get("name") or "",
                "direction": t.get("direction", ""),
                "conf": float(t.get("conf") or 0),
                "half_life_hours": int(t.get("half_life_hours") or 0),
                "thesis": t.get("thesis", ""),
                "entry": t.get("entry", ""),
                "event_id": e.id,
                "event_title": e.content[:60],
                "event_ts": e.ts,
                "priced_in": d.get("already_priced_in", ""),
                "risk_flags": d.get("risk_flags", ""),
            })

    return agg


# ==================== Markdown 生成 ====================
def _fmt_hhmm(ts: int) -> str:
    return time.strftime("%H:%M", time.localtime(ts))


def _window_desc(ts_start: int, ts_end: int) -> str:
    s = time.strftime("%m-%d %H:%M", time.localtime(ts_start))
    e = time.strftime("%H:%M", time.localtime(ts_end))
    return f"{s} → {e}"


def render_markdown(agg: dict, min_conf: float, top: int,
                    title_prefix: str = "") -> str:
    """生成飞书友好的 markdown. 避免过长表格, 控制在 2000 字内."""
    lines: list[str] = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    window = _window_desc(agg["ts_start"], agg["ts_end"])

    prefix = f"{title_prefix} " if title_prefix else ""
    lines.append(f"# {prefix}📡 Radar 情报简报  {now}")
    lines.append("")
    lines.append(f"**窗口**: {window}  |  **事件**: {agg['total']} 条")
    lines.append(
        f"(已分析 {agg['analyzed']}, 深度 {agg['deep_analyzed']}, "
        f"噪音过滤 {len(agg['noise_ids'])})"
    )
    lines.append("")

    # 类型分布
    if agg["by_type"]:
        type_parts = [
            f"{k}×{v}" for k, v in
            agg["by_type"].most_common() if k != "noise"
        ]
        if type_parts:
            lines.append("**类型**: " + " / ".join(type_parts))
            lines.append("")

    # 候选清单
    cands = [c for c in agg["candidates"] if c["conf"] >= min_conf]
    # 去重: 同 code+direction 只保留最高 conf
    seen: dict = {}
    for c in cands:
        key = (c["code"], c["direction"])
        if key not in seen or c["conf"] > seen[key]["conf"]:
            seen[key] = c
    cands = sorted(seen.values(), key=lambda x: -x["conf"])[:top]

    if cands:
        lines.append(f"## 🔥 可交易候选 (conf≥{min_conf}, Top {len(cands)})")
        lines.append("")
        # 飞书 markdown 不支持 GFM table, 用 bullet
        for c in cands:
            direction_tag = {
                "long": "🟢 long", "short": "🔴 short", "avoid": "⚠️ avoid",
            }.get(c["direction"], c["direction"])
            entry = f" · 入场 **{c['entry']}**" if c.get("entry") and c["entry"] != "--" else ""
            lines.append(
                f"- **`{c['code']}` {c['name'] or '--'}**  {direction_tag}  "
                f"conf={c['conf']:.2f}  半衰期{c['half_life_hours']}h"
                f"{entry}"
            )
            lines.append(
                f"  - {_fmt_hhmm(c['event_ts'])} {c['event_title'][:60]}"
            )
            # 对前 3 个候选加简短 thesis
            if cands.index(c) < 3 and c.get("thesis"):
                lines.append(f"  - _{c['thesis'][:150]}_")
        lines.append("")
    else:
        lines.append(f"## 🔥 可交易候选")
        lines.append(f"(窗口内无 conf≥{min_conf} 的候选)")
        lines.append("")

    # 传导链(按 event_id 去重, 同事件只展示一次)
    seen_events: set = set()
    chains_to_show = []
    for sc in agg["supply_chains"]:
        if sc["event_id"] in seen_events:
            continue
        seen_events.add(sc["event_id"])
        chains_to_show.append(sc)
    if chains_to_show:
        lines.append("## 🔗 关键传导链")
        lines.append("")
        for sc in chains_to_show[:3]:
            lines.append(
                f"- **{_fmt_hhmm(sc['event_ts'])} {sc['title'][:40]}**"
            )
            lines.append(f"  - {sc['chain'][:200]}")
        lines.append("")

    # 风险提示(按 event_id 去重, 避免同事件多候选重复)
    seen_risk_events: set = set()
    risks = []
    for c in cands[:10]:
        if c["event_id"] in seen_risk_events:
            continue
        if c.get("risk_flags"):
            seen_risk_events.add(c["event_id"])
            title = c["event_title"][:30]
            risks.append(
                f"- **{_fmt_hhmm(c['event_ts'])} {title}**: {c['risk_flags'][:120]}"
            )
    if risks:
        lines.append("## ⚠️ 风险标签")
        lines.append("")
        lines.extend(risks[:3])
        lines.append("")

    lines.append("---")
    lines.append("_由 radar_worker 自动分析, 结果仅供参考, 不构成投资建议._")
    return "\n".join(lines)


# ==================== 飞书推送 ====================
def send_lark(markdown: str, user_id: str, dry_run: bool = False) -> bool:
    if dry_run:
        print(markdown)
        return True
    cmd = ["lark-cli", "im", "+messages-send",
           "--as", "user", "--user-id", user_id,
           "--markdown", markdown]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except FileNotFoundError:
        logger.error("lark-cli 未安装. 先执行 `brew install lark-cli` 或参考项目 README.")
        return False
    except subprocess.TimeoutExpired:
        logger.error("lark-cli 超时 (30s)")
        return False
    if p.returncode == 0:
        logger.info(f"✓ 飞书已送达 ({len(markdown)} 字)")
        return True
    logger.error(f"❌ lark-cli rc={p.returncode}: {(p.stdout + p.stderr)[:300]}")
    return False


# ==================== main ====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=0,
                    help="时间窗口小时, 0 表示从今天 9:00 开始")
    ap.add_argument("--min-conf", type=float, default=0.5)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--limit", type=int, default=500,
                    help="从 DB 最多拉多少条事件聚合")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印 markdown, 不发飞书")
    ap.add_argument("--to", type=str,
                    default=os.environ.get("LARK_USER_OPEN_ID", DEFAULT_LARK_USER))
    ap.add_argument("--title-prefix", type=str, default="",
                    help="标题前缀, 比如 [测试] [盘中] [盘后]")
    args = ap.parse_args()

    since_ts = int(time.time()) - args.hours * 3600 if args.hours else today_open_ts()

    logger.info(
        f"简报窗口 since={datetime.fromtimestamp(since_ts):%Y-%m-%d %H:%M} "
        f"min_conf={args.min_conf} top={args.top}"
    )

    agg = collect(since_ts, limit=args.limit)
    if agg["total"] == 0:
        logger.info("窗口内无事件, 跳过推送")
        return 0

    md = render_markdown(agg, args.min_conf, args.top,
                          title_prefix=args.title_prefix)
    ok = send_lark(md, args.to, dry_run=args.dry_run)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
