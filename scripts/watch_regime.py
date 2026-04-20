"""Regime 盘中监控器 - 定时快照 + 持久化.

用法:
    # 默认每 15 分钟刷一次, 写入 output/regime_timeline.jsonl
    python3 scripts/watch_regime.py

    # 自定义间隔 (分钟)
    python3 scripts/watch_regime.py --interval 5

    # 只跑 N 次后退出
    python3 scripts/watch_regime.py --max-ticks 10

    # 同时打印简报到终端
    python3 scripts/watch_regime.py --verbose

输出:
    output/regime_timeline.jsonl       每次 regime 快照一行 (JSON)
    output/regime_summary.txt          人类可读摘要 (每次追加)

盘中建议:
    09:25 开始启动, 15:00 自动退出. 整日累积约 22-44 条快照,
    可用于事后复盘 regime 切换时点与资金流映射.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, time as dtime
from pathlib import Path

# 项目根 import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["QUANT_WEB_MODE"] = "real"

from webapp.live_regime import fetch_live_regime
from utils.logger import logger


OUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUT_DIR.mkdir(exist_ok=True)
TIMELINE_PATH = OUT_DIR / "regime_timeline.jsonl"
SUMMARY_PATH = OUT_DIR / "regime_summary.txt"


def _is_trading_session(now: datetime | None = None) -> bool:
    """是否在 A 股交易时段 (09:15-15:00, 周一到周五)."""
    now = now or datetime.now()
    if now.weekday() >= 5:
        return False
    m = now.hour * 60 + now.minute
    return (9 * 60 + 15) <= m <= (15 * 60)


def _should_auto_exit(now: datetime | None = None) -> bool:
    """15:05 后自动退出 (给收盘后几分钟缓冲)."""
    now = now or datetime.now()
    if now.weekday() >= 5:
        return False
    m = now.hour * 60 + now.minute
    return m > (15 * 60 + 5)


def tick_once(verbose: bool = True) -> dict | None:
    """单次快照 + 追加写入."""
    ts = datetime.now()
    try:
        regime = fetch_live_regime(cache=False)
    except Exception as e:
        regime = {
            "data_source": f"error: {e.__class__.__name__}",
            "error": str(e),
        }

    snapshot = {
        "ts": ts.isoformat(timespec="seconds"),
        "ts_unix": int(ts.timestamp()),
        "trading_session": _is_trading_session(ts),
        **regime,
    }

    # 追加写 jsonl
    with open(TIMELINE_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(snapshot, ensure_ascii=False, default=str) + "\n")

    # 追加人类可读摘要
    regime_name = snapshot.get("regime", "?")
    confidence = snapshot.get("confidence", 0)
    pos_mult = snapshot.get("position_mult", 0)
    trend = snapshot.get("trend_direction", "?")
    ds = snapshot.get("data_source", "?")
    line = (
        f"[{ts:%Y-%m-%d %H:%M:%S}] "
        f"{regime_name:18s} | 信心 {confidence:>4.0%} | "
        f"仓位 {pos_mult:.2f} | trend={trend:5s} | src={ds}\n"
    )
    with open(SUMMARY_PATH, "a", encoding="utf-8") as f:
        f.write(line)

    if verbose:
        print(line, end="")
        if snapshot.get("reasons"):
            for r in snapshot["reasons"][:3]:
                print(f"    - {r}")

    return snapshot


def main():
    ap = argparse.ArgumentParser(description="Regime 盘中监控器")
    ap.add_argument("--interval", type=int, default=15,
                     help="采样间隔 (分钟, 默认 15)")
    ap.add_argument("--max-ticks", type=int, default=0,
                     help="最大采样次数 (0 = 不限, 盘后自动退出)")
    ap.add_argument("--force-always", action="store_true",
                     help="强制一直跑 (忽略盘后自动退出)")
    ap.add_argument("--verbose", action="store_true", default=True,
                     help="打印每次快照到终端")
    args = ap.parse_args()

    logger.info(
        f"regime monitor 启动: interval={args.interval}min, "
        f"max_ticks={args.max_ticks or '∞'}, "
        f"output={TIMELINE_PATH}"
    )

    tick = 0
    while True:
        snapshot = tick_once(verbose=args.verbose)
        tick += 1

        if args.max_ticks and tick >= args.max_ticks:
            logger.info(f"达到 max_ticks={args.max_ticks}, 退出")
            break

        if not args.force_always and _should_auto_exit():
            logger.info("盘后 15:05+, 自动退出")
            break

        sleep_s = args.interval * 60
        time.sleep(sleep_s)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断, 已退出")
