"""每日数据增量更新 - 累积样本.

每个交易日收盘后跑一次:
    1. kline: 新增今日一行 × 500 只
    2. 龙虎榜: 拉今日上榜数据
    3. insider: 今日高管增减持
    4. fundflow: 今日主力资金流 (akshare 有近 120 日, 每日加 1)

策略: 读现有 parquet 最大日期, 只拉新增部分, concat + 去重 + 写回.

用法:
    python3 scripts/daily_data_updater.py                 # 增量到今天
    python3 scripts/daily_data_updater.py --only kline    # 只更新某类
    python3 scripts/daily_data_updater.py --force-full    # 全量重拉 (慎用)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from data_adapter.em_direct import bulk_fetch_daily
from data_adapter.lhb import fetch_lhb_range
from data_adapter.insider import fetch_insider_range
from data_adapter.fundflow import bulk_fetch_fund_flow

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"


def _fmt_date(d: pd.Timestamp, sep: str = "") -> str:
    if sep == "":
        return d.strftime("%Y%m%d")
    return d.strftime(f"%Y{sep}%m{sep}%d")


def _next_business_day(d: pd.Timestamp) -> pd.Timestamp:
    return (d + pd.Timedelta(days=1)).normalize()


# ---------- kline 增量 ----------
def update_kline(today: pd.Timestamp, force_full: bool = False):
    pattern = "kline_20230101_*_n500.parquet"
    files = sorted(CACHE.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not files or force_full:
        print("  [kline] 无缓存 / 全量模式, 跳过 (请跑 run_real_research_v5 初始化)")
        return

    latest = files[-1]
    df = pd.read_parquet(latest)
    df["date"] = pd.to_datetime(df["date"])
    cur_max = df["date"].max()
    print(f"  [kline] 当前覆盖到 {cur_max.date()}, 池子 {df['code'].nunique()} 只")

    if cur_max >= today:
        print(f"  [kline] 已是最新, 跳过")
        return

    start_d = _next_business_day(cur_max)
    start_s = _fmt_date(start_d)
    end_s = _fmt_date(today)
    codes = df["code"].unique().tolist()
    print(f"  [kline] 增量拉 {start_s} → {end_s} ({len(codes)} 只)...")
    new = bulk_fetch_daily(codes, start_s, end_s, sleep_ms=80, progress=False)
    if new.empty:
        print(f"  [kline] 无新数据")
        return
    new["date"] = pd.to_datetime(new["date"])
    print(f"  [kline] 新增 {len(new)} 行")

    merged = pd.concat([df, new], ignore_index=True).drop_duplicates(
        subset=["code", "date"], keep="last"
    ).sort_values(["code", "date"])

    # 新文件名: kline_20230101_{today}_n500.parquet
    new_path = CACHE / f"kline_20230101_{_fmt_date(today)}_n500.parquet"
    merged.to_parquet(new_path)
    print(f"  [kline] ✓ 写入 {new_path.name} ({len(merged)} 行)")


# ---------- lhb 增量 ----------
def update_lhb(today: pd.Timestamp):
    files = sorted([p for p in CACHE.glob("lhb_2*.parquet")
                     if "taxonomy" not in p.name],
                    key=lambda p: p.stat().st_mtime)
    if not files:
        print("  [lhb] 无缓存, 跳过"); return

    latest = files[-1]
    df = pd.read_parquet(latest)
    df["TRADE_DATE"] = pd.to_datetime(df["TRADE_DATE"])
    cur_max = df["TRADE_DATE"].max()
    print(f"  [lhb] 当前覆盖到 {cur_max.date()}, {len(df)} 条历史")

    if cur_max >= today:
        print(f"  [lhb] 已是最新")
        return

    # 拉最近区间, lhb API 按日期 filter 不会重复太多
    start_s = _fmt_date(_next_business_day(cur_max))
    end_s = _fmt_date(today)
    print(f"  [lhb] 增量 {start_s} → {end_s}...")
    new = fetch_lhb_range(start_s, end_s)
    if new.empty:
        print(f"  [lhb] 无新数据"); return

    # ⚠️ 客户端过滤 (API filter 偶尔失效)
    new["TRADE_DATE"] = pd.to_datetime(new["TRADE_DATE"])
    new = new[new["TRADE_DATE"] > cur_max]
    if new.empty:
        print(f"  [lhb] 过滤后无真·新数据"); return

    merged = pd.concat([df, new], ignore_index=True).drop_duplicates(
        subset=["TRADE_DATE", "code", "EXPLAIN", "BUY_SEAT_NEW"], keep="last"
    )
    new_path = CACHE / f"lhb_20230101_{_fmt_date(today)}.parquet"
    merged.to_parquet(new_path)
    print(f"  [lhb] ✓ 真·新增 {len(new)} 条, 总 {len(merged)}, 写入 {new_path.name}")


# ---------- insider 增量 ----------
def update_insider(today: pd.Timestamp):
    files = sorted(CACHE.glob("insider_*.parquet"),
                    key=lambda p: p.stat().st_mtime)
    if not files:
        print("  [insider] 无缓存"); return

    latest = files[-1]
    df = pd.read_parquet(latest)
    df["CHANGE_DATE"] = pd.to_datetime(df["CHANGE_DATE"])
    cur_max = df["CHANGE_DATE"].max()
    print(f"  [insider] 当前覆盖到 {cur_max.date()}, {len(df)} 条历史")

    if cur_max >= today:
        print(f"  [insider] 已是最新"); return

    start_s = _fmt_date(_next_business_day(cur_max))
    end_s = _fmt_date(today)
    print(f"  [insider] 增量 {start_s} → {end_s}...")
    new = fetch_insider_range(start_s, end_s)
    if new.empty:
        print(f"  [insider] 无新数据"); return

    # ⚠️ 东财 datacenter 的 CHANGE_DATE filter 有时不生效 → 客户端 double check
    new["CHANGE_DATE"] = pd.to_datetime(new["CHANGE_DATE"])
    new = new[new["CHANGE_DATE"] > cur_max]
    if new.empty:
        print(f"  [insider] 过滤后无真·新数据 (API filter 可能失效)"); return

    merged = pd.concat([df, new], ignore_index=True).drop_duplicates(
        subset=["CHANGE_DATE", "code", "PERSON_NAME", "CHANGE_AMOUNT"],
        keep="last",
    )
    new_path = CACHE / f"insider_20230101_{_fmt_date(today)}.parquet"
    merged.to_parquet(new_path)
    print(f"  [insider] ✓ 真·新增 {len(new)} 条, 总 {len(merged)}")


# ---------- fundflow 增量 ----------
def update_fundflow(today: pd.Timestamp):
    """资金流 akshare 只有最近 120 日, 每次重拉覆盖."""
    files = list(CACHE.glob("fundflow_*.parquet"))
    if not files:
        print("  [fundflow] 无缓存, 跳过 (请先跑过一次完整拉取)"); return

    latest = sorted(files, key=lambda p: p.stat().st_mtime)[-1]
    df = pd.read_parquet(latest)
    df["date"] = pd.to_datetime(df["date"])
    cur_max = df["date"].max()
    codes = df["code"].unique().tolist()
    print(f"  [fundflow] 当前覆盖到 {cur_max.date()}, {len(codes)} 只")

    if cur_max >= today:
        print(f"  [fundflow] 已是最新"); return

    # akshare 每只都要重拉 (返回全 120 日), 用最小代价方案:
    # 只重拉前 200 只最活跃的, 作为日常更新
    # 完全覆盖需要 full batch
    print(f"  [fundflow] 重拉 {len(codes)} 只 (akshare 限速 ~5 min)...")
    # 用新文件名,覆盖 latest
    new_path = CACHE / f"fundflow_{len(codes)}_{_fmt_date(today)}.parquet"
    new_df = bulk_fetch_fund_flow(codes, sleep_ms=300, cache_path=new_path)
    if not new_df.empty:
        print(f"  [fundflow] ✓ 写入 {new_path.name} ({len(new_df)} 行)")


# ---------- 主 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["kline", "lhb", "insider", "fundflow"])
    ap.add_argument("--asof", default=None)
    ap.add_argument("--force-full", action="store_true")
    args = ap.parse_args()

    today = pd.Timestamp(args.asof) if args.asof else pd.Timestamp.today().normalize()
    print(f"\n{'='*64}\n  📥 Daily Data Updater — asof {today.date()}\n{'='*64}\n")

    sources = {
        "kline": lambda: update_kline(today, args.force_full),
        "lhb": lambda: update_lhb(today),
        "insider": lambda: update_insider(today),
        "fundflow": lambda: update_fundflow(today),
    }

    to_run = [args.only] if args.only else list(sources.keys())
    for name in to_run:
        print(f"\n── {name.upper()} ──")
        try:
            sources[name]()
        except Exception as e:
            print(f"  [{name}] ❌ 失败: {e}")

    print(f"\n{'='*64}\n  ✅ 增量更新完成\n{'='*64}")


if __name__ == "__main__":
    main()
