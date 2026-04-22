"""sentiment_cycle — A 股短线情绪周期日报.

读取全市场(或 500 只近似)日 K, 聚合出:
    - 涨停数  / 跌停数
    - 连板天梯 (3连/4连/5连+/最高)
    - 炸板率  (开板数 / 涨停股数)
    - 赚钱效应 (涨幅 > 0 占比)
    - 情绪分档 (冰点/温/热/沸腾/崩溃)

用 factors/alpha_limit.py 的 LU_FLAG/BOOM_BAN_FLAG/STREAK_UP 字段,
不重新实现轮子.

触发 "情绪退潮" 警告(可用于外部 regime 判断):
    boom_rate > 0.5 且 max_streak_up < 3 且 limit_up_count < 50
    → "短线建议停手"
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

env = Path(__file__).resolve().parent.parent / ".env"
if env.exists():
    for line in env.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from factors.alpha_limit import compute_limit_alpha

ROOT = Path(__file__).resolve().parent.parent
LARK_BIN = "/opt/homebrew/bin/lark-cli"
CACHE = ROOT / "cache"
OUT_DIR = ROOT / "output" / "sentiment_cycle"


def load_market_kline() -> pd.DataFrame:
    """读最新的"全市场近似"日 K (500 只)."""
    candidates = sorted([p for p in CACHE.glob("kline_*_n500.parquet")])
    if not candidates:
        raise FileNotFoundError("未找到全市场 kline 缓存")
    latest = candidates[-1]
    print(f"  数据源: {latest.name}")
    df = pd.read_parquet(latest)
    df["date"] = pd.to_datetime(df["date"])
    return df


def compute_cycle(daily: pd.DataFrame, asof: pd.Timestamp | None = None) -> dict:
    """对全市场 daily 跑 alpha_limit, 聚合当日指标."""
    if asof is None:
        asof = daily["date"].max()

    panel = compute_limit_alpha(daily)
    if panel.empty:
        return {}

    # 取当日 + 前一日截面
    dates = panel.index.get_level_values("date").unique().sort_values()
    today = dates[dates <= asof].max()
    prev_idx = dates.get_loc(today) - 1
    prev = dates[prev_idx] if prev_idx >= 0 else None

    today_slice = panel.xs(today, level="date", drop_level=True)
    prev_slice = (panel.xs(prev, level="date", drop_level=True)
                  if prev is not None else None)

    # 聚合
    lu = int(today_slice["LU_FLAG"].sum())
    ld = int(today_slice["LD_FLAG"].sum())
    boom = int(today_slice["BOOM_BAN_FLAG"].sum())
    boom_rate = (boom / max(1, lu + boom))  # 炸板率 = 炸板 / (封+炸)

    streak = today_slice["STREAK_UP"].fillna(0)
    max_streak = int(streak.max())
    ladder = {
        "2连": int(((streak >= 2) & (streak < 3)).sum()),
        "3连": int(((streak >= 3) & (streak < 4)).sum()),
        "4连": int(((streak >= 4) & (streak < 5)).sum()),
        "5连+": int((streak >= 5).sum()),
    }

    # 昨日对比
    lu_prev = int(prev_slice["LU_FLAG"].sum()) if prev_slice is not None else 0
    dlu = lu - lu_prev
    dlu_pct = (dlu / lu_prev * 100) if lu_prev > 0 else 0

    # 赚钱效应: 当日全市场涨幅占比 (用 daily 的 pct_chg)
    today_daily = daily[daily["date"] == today]
    pct_up = float((today_daily["pct_chg"] > 0).mean()) if "pct_chg" in today_daily.columns else 0.5

    # 情绪分档
    if boom_rate > 0.5 and max_streak < 3 and lu < 50:
        regime = "🔴 退潮"
        advice = "短线建议停手, 情绪已退"
    elif boom_rate > 0.4 and max_streak < 4:
        regime = "🟠 降温"
        advice = "谨慎打板, 注意 T+1 风险"
    elif lu > 100 and max_streak >= 5 and boom_rate < 0.3:
        regime = "🟢 沸腾"
        advice = "情绪高潮, 可跟随龙头, 留意次日高开"
    elif lu > 60 and max_streak >= 4:
        regime = "🟢 高涨"
        advice = "主线清晰, 可按信号操作"
    elif lu > 30:
        regime = "⚪️ 平稳"
        advice = "常规节奏, 信号为准"
    else:
        regime = "🔵 冰点"
        advice = "涨停稀少, 避免追高"

    return {
        "date": str(today.date()),
        "prev_date": str(prev.date()) if prev is not None else None,
        "limit_up_count": lu,
        "limit_up_prev": lu_prev,
        "limit_up_delta": dlu,
        "limit_up_delta_pct": dlu_pct,
        "limit_down_count": ld,
        "boom_ban_count": boom,
        "boom_rate": boom_rate,
        "max_streak_up": max_streak,
        "ladder": ladder,
        "pct_up": pct_up,
        "universe_size": int(today_slice.shape[0]),
        "regime": regime,
        "advice": advice,
    }


def build_report(m: dict) -> str:
    lines = [
        f"📊 **A股情绪周期 — {m['date']}**",
        "",
        f"样本: {m['universe_size']} 只 (500 近似全市场)",
        "",
        f"**涨停**: {m['limit_up_count']} "
        f"(昨日 {m['limit_up_prev']}, {m['limit_up_delta']:+d}, "
        f"{m['limit_up_delta_pct']:+.0f}%)",
        f"**跌停**: {m['limit_down_count']}",
        f"**炸板**: {m['boom_ban_count']} 只   "
        f"炸板率: **{m['boom_rate']:.0%}**",
        "",
        f"**连板天梯** (最高 **{m['max_streak_up']}连**)",
        f"  2连: {m['ladder']['2连']} 只",
        f"  3连: {m['ladder']['3连']} 只",
        f"  4连: {m['ladder']['4连']} 只",
        f"  5连+: {m['ladder']['5连+']} 只",
        "",
        f"**赚钱效应**: 涨幅>0 占比 {m['pct_up']:.0%}",
        "",
        f"**情绪状态**: {m['regime']}",
        f"**策略建议**: {m['advice']}",
        "",
        "⚠️ 500 只 ≠ 真实全市场 5000 只, 指标为近似值. "
        "真短线请用东财涨停池接口校准.",
    ]
    return "\n".join(lines)


def send_lark(md: str) -> bool:
    user_id = os.environ.get("LARK_USER_OPEN_ID",
                              "ou_5be0f87dc7cec796b7ea97d0a9b5302f")
    cmd = [str(LARK_BIN), "im", "+messages-send",
           "--as", "user", "--user-id", user_id, "--markdown", md]
    try:
        env = {**os.environ, "PATH": f"/opt/homebrew/bin:{os.environ.get('PATH', '/usr/local/bin:/usr/bin:/bin')}"}
        rc = subprocess.run(cmd, capture_output=True, text=True, timeout=30, env=env)
        if rc.returncode == 0:
            print("✓ 飞书已送达")
            return True
        print(f"❌ 飞书失败: {rc.stderr[:200]}")
    except Exception as e:
        print(f"❌ 飞书异常: {e}")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run-lark", action="store_true")
    ap.add_argument("--asof", default=None, help="YYYY-MM-DD")
    args = ap.parse_args()

    print(f"\n{'='*64}\n  📊 Sentiment Cycle {datetime.now():%Y-%m-%d %H:%M}\n{'='*64}")

    daily = load_market_kline()
    print(f"  总行数: {len(daily)}, 股票数 {daily['code'].nunique()}, "
          f"日期范围 {daily['date'].min().date()}~{daily['date'].max().date()}")

    asof = pd.Timestamp(args.asof) if args.asof else None
    metrics = compute_cycle(daily, asof)
    if not metrics:
        print("❌ 计算失败"); return 1

    report = build_report(metrics)
    print(f"\n{'='*64}\n飞书消息:\n{'='*64}")
    print(report)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    import json
    out_json = OUT_DIR / f"{metrics['date']}.json"
    out_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False,
                                    default=str), encoding="utf-8")
    print(f"\n  metrics: {out_json}")

    if not args.dry_run_lark:
        send_lark(report)


if __name__ == "__main__":
    main()
