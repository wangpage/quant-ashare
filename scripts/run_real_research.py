"""用真实东财数据跑一次完整 ResearchPipeline."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from data_adapter.em_direct import (
    bulk_fetch_daily, fetch_index_daily, fetch_hot_stocks,
)
from pipeline import (
    ResearchPipeline, build_research_report,
    build_research_report_html,
)
from utils.logger import logger


def main(n_stocks: int = 20, start: str = "20230101", end: str = "20260420"):
    print(f"\n{'='*64}")
    print(f"  真实数据研究 Pipeline - {time.strftime('%Y-%m-%d %H:%M')}")
    print(f"  股票池: top {n_stocks} 活跃股")
    print(f"  时段: {start} → {end}")
    print('='*64)

    # 1. 股票池 - 硬编码活跃大盘+行业龙头
    DEFAULT_POOL = [
        "300750",  # 宁德时代
        "600519",  # 贵州茅台
        "000858",  # 五粮液
        "601318",  # 中国平安
        "600036",  # 招商银行
        "601398",  # 工商银行
        "002594",  # 比亚迪
        "000333",  # 美的集团
        "600900",  # 长江电力
        "601888",  # 中国中免
        "601012",  # 隆基绿能
        "002415",  # 海康威视
        "600030",  # 中信证券
        "000001",  # 平安银行
        "600276",  # 恒瑞医药
        "601166",  # 兴业银行
        "002475",  # 立讯精密
        "600309",  # 万华化学
        "002352",  # 顺丰控股
        "000651",  # 格力电器
        "601899",  # 紫金矿业
        "600028",  # 中国石化
        "600050",  # 中国联通
        "601988",  # 中国银行
        "688981",  # 中芯国际
    ]
    print("\n[1/4] 选定股票池 (硬编码活跃股)...")
    codes = DEFAULT_POOL[:n_stocks]
    print(f"  {len(codes)} 只: {codes}")

    # 2. 拉日线
    print(f"\n[2/4] 拉日线数据 ({start} - {end})...")
    t0 = time.time()
    daily_df = bulk_fetch_daily(codes, start, end, sleep_ms=80,
                                  include_factor=True)
    print(f"  耗时 {time.time()-t0:.1f}s, 总记录 {len(daily_df)} 行")

    if daily_df.empty:
        print("❌ 没拉到数据, 中止")
        return

    # 3. 拉指数 (用于 Barra 市场收益)
    print("\n[3/4] 拉沪深300 指数...")
    idx_df = fetch_index_daily("000300", start, end)
    print(f"  沪深300 {len(idx_df)} 行")

    # 4. 造 research pipeline 需要的输入
    # 近似市值: 近期 amount 均值 × 常数 (粗略)
    last_close = daily_df.groupby("code")["close"].last()
    # 硬编码简化: 假设都是大盘股, 用真实市值也可以后续替换
    market_cap = pd.Series({c: 3e10 for c in codes})

    # 市场收益 = 沪深300 pct_chg
    idx_df["date"] = pd.to_datetime(idx_df["date"])
    market_return = idx_df.set_index("date")["pct_chg"] / 100

    # 5. 跑 ResearchPipeline
    # 先按 code+date 排序 (pipeline 的 audit 需要)
    daily_df = daily_df.sort_values(["code", "date"]).reset_index(drop=True)
    print("\n[4/4] 运行 ResearchPipeline...")
    pipe = ResearchPipeline(skip_audit=False)
    result = pipe.run(
        daily_df=daily_df,
        market_cap=market_cap,
        market_return=market_return,
    )

    report = build_research_report(result)
    print("\n" + report)

    # 保存 Markdown + HTML 两种格式
    ts = time.strftime("%Y%m%d_%H%M")
    out_dir = Path(__file__).resolve().parent.parent / "output"
    out_dir.mkdir(exist_ok=True)
    md_path = out_dir / f"research_{ts}.md"
    html_path = out_dir / f"research_{ts}.html"
    md_path.write_text(report, encoding="utf-8")
    build_research_report_html(result, out_path=html_path)
    print(f"\n📝 Markdown: {md_path}")
    print(f"🌐 HTML 交互报告: {html_path}")
    print(f"   浏览器打开: open {html_path}")

    # 如有 qlib 可继续接入 LightGBM, 这里先用自有的 research pipeline
    return result


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="股票数量")
    ap.add_argument("--start", default="20230101")
    ap.add_argument("--end", default="20260420")
    args = ap.parse_args()
    main(n_stocks=args.n, start=args.start, end=args.end)
