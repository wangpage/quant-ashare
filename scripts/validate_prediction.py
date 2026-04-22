"""验证 2026-04-21 watchlist 信号对 0422 涨跌的命中率.

调用方式:
    python3 scripts/validate_prediction.py             # 读 akshare 实时快照
    python3 scripts/validate_prediction.py --mode close  # 收盘后用日 K (更准)

输出:
    /Users/page/Desktop/股票/验证_20260422.csv
    - 每只股: 0421 预测方向 vs 0422 实际涨跌
    - 分档命中率统计 (🟩/🟢 做多命中率, 🟡/🟥 做空命中率)
    - IC (alpha_z vs 实际收益率的秩相关)
    - 因子调整建议
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
LARK_BIN = "/opt/homebrew/bin/lark-cli"

env = ROOT / ".env"
if env.exists():
    for line in env.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

SIG_CSV = ROOT / "output" / "paper_trade" / "watchlist_signals" / "2026-04-21.csv"
OUT_CSV = Path("/Users/page/Desktop/股票/验证_20260422.csv")
SUMMARY_MD = Path("/Users/page/Desktop/股票/验证摘要_20260422.md")


def send_lark(markdown: str) -> bool:
    """推送 markdown 消息到飞书."""
    user_id = os.environ.get("LARK_USER_OPEN_ID",
                              "ou_5be0f87dc7cec796b7ea97d0a9b5302f")
    cmd = [str(LARK_BIN), "im", "+messages-send",
           "--as", "user", "--user-id", user_id,
           "--markdown", markdown]
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


def fetch_spot_em() -> pd.DataFrame:
    """akshare 东财全市场实时快照 (盘中有效). 带 3 次重试."""
    import akshare as ak
    import time
    last_err = None
    for attempt in range(3):
        try:
            df = ak.stock_zh_a_spot_em()
            df = df.rename(columns={
                "代码": "code",
                "名称": "name_em",
                "最新价": "price_now",
                "涨跌幅": "pct_chg",
                "涨跌额": "chg_amt",
                "今开": "open_today",
                "昨收": "prev_close",
            })
            return df[["code", "name_em", "price_now", "pct_chg", "open_today", "prev_close"]]
        except Exception as e:
            last_err = e
            print(f"  [重试 {attempt+1}/3] {e}")
            time.sleep(5)
    raise RuntimeError(f"akshare spot 拉取失败 3 次: {last_err}")


def fetch_daily_close(codes: list[str], date: str) -> pd.DataFrame:
    """收盘后用 bulk_fetch_daily 拉 0422 日 K."""
    from data_adapter.em_direct import bulk_fetch_daily
    df = bulk_fetch_daily(codes, date, date, sleep_ms=80, progress=False)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="spot", choices=["spot", "close"],
                    help="spot=盘中快照, close=收盘日 K")
    ap.add_argument("--date", default="20260422", help="验证日 (仅 close 模式用)")
    ap.add_argument("--dry-run-lark", action="store_true",
                    help="不推送飞书 (默认推送)")
    args = ap.parse_args()

    print(f"\n{'='*64}\n  验证模式: {args.mode}  时间: {datetime.now():%Y-%m-%d %H:%M}\n{'='*64}\n")

    # 1. 加载 0421 预测
    pred = pd.read_csv(SIG_CSV)
    pred["code"] = pred["code"].astype(str).str.zfill(6)
    print(f"  加载预测: {len(pred)} 只 (0421 alpha_z)")

    # 2. 拉 0422 最新数据
    if args.mode == "spot":
        print("  拉取实时快照 (akshare spot)...")
        spot = fetch_spot_em()
        spot["code"] = spot["code"].astype(str).str.zfill(6)
        merged = pred.merge(spot, on="code", how="left")
        merged["实际涨跌%"] = merged["pct_chg"].astype(float)
        merged["最新价"] = merged["price_now"]
    else:
        print(f"  拉取收盘日 K {args.date}...")
        daily = fetch_daily_close(pred["code"].tolist(), args.date)
        if daily.empty:
            print("❌ 无数据")
            return 1
        daily["code"] = daily["code"].astype(str).str.zfill(6)
        daily_last = daily.sort_values("date").groupby("code").tail(1)
        daily_last["实际涨跌%"] = (daily_last["close"] / pred.set_index("code").loc[daily_last["code"]]["latest_close"].values - 1) * 100
        merged = pred.merge(daily_last[["code", "close", "实际涨跌%"]], on="code", how="left")
        merged["最新价"] = merged["close"]

    # 3. 打分 — 预测方向正确吗?
    def verdict(row):
        z = row["alpha_z"]
        pct = row["实际涨跌%"]
        if pd.isna(pct):
            return "无数据"
        if z >= 0.75 and pct > 0:
            return "✅ 看多对"
        if z >= 0.75 and pct <= 0:
            return "❌ 看多错"
        if z <= -0.75 and pct < 0:
            return "✅ 看空对"
        if z <= -0.75 and pct >= 0:
            return "❌ 看空错"
        return "⚪️ 中性"

    merged["验证结果"] = merged.apply(verdict, axis=1)

    # 4. 统计命中率
    valid = merged.dropna(subset=["实际涨跌%"])
    long_signals = valid[valid["alpha_z"] >= 0.75]
    short_signals = valid[valid["alpha_z"] <= -0.75]

    long_hit = (long_signals["实际涨跌%"] > 0).sum() / max(1, len(long_signals))
    short_hit = (short_signals["实际涨跌%"] < 0).sum() / max(1, len(short_signals))

    # IC: alpha_z 与实际收益的秩相关
    ic, pval = spearmanr(valid["alpha_z"], valid["实际涨跌%"])
    # 多空价差: 前 N 只 - 后 N 只
    top_n = min(10, len(valid) // 3)
    ranked = valid.sort_values("alpha_z", ascending=False)
    ls_spread = ranked.head(top_n)["实际涨跌%"].mean() - ranked.tail(top_n)["实际涨跌%"].mean()

    # 5. 保存 CSV
    out_cols = ["排名", "name", "code", "latest_close", "alpha_z", "信号档",
                "明日(0422)预测", "操作建议", "主导因子", "最新价", "实际涨跌%", "验证结果"]
    pred_full = pred.copy()
    if "排名" not in pred_full.columns:
        pred_full["排名"] = range(1, len(pred_full) + 1)
    # 从预测报告里取完整列
    full_pred_csv = Path("/Users/page/Desktop/股票/预测_20260421_for_0422.csv")
    if full_pred_csv.exists():
        full = pd.read_csv(full_pred_csv)
        full["code"] = full["股票代码"].astype(str).str.zfill(6)
        merged_final = full.merge(
            merged[["code", "最新价", "实际涨跌%", "验证结果"]],
            on="code", how="left"
        )
    else:
        merged_final = merged

    merged_final.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    # 6. Summary
    summary = [
        f"# 量化信号验证报告 {datetime.now():%Y-%m-%d %H:%M}",
        "",
        f"- 预测基准: 2026-04-21 收盘后",
        f"- 验证标的: 2026-04-22 ({args.mode}: {'盘中实时' if args.mode=='spot' else '收盘日K'})",
        f"- 样本数: {len(valid)} 只",
        "",
        "## 🎯 核心指标",
        "",
        f"| 指标 | 数值 | 说明 |",
        f"|---|---|---|",
        f"| **看多命中率** | {long_hit:.1%} ({(long_signals['实际涨跌%']>0).sum()}/{len(long_signals)}) | 🟢+🟩 档股票中,实际上涨的比例 |",
        f"| **看空命中率** | {short_hit:.1%} ({(short_signals['实际涨跌%']<0).sum()}/{len(short_signals)}) | 🟡+🟥 档股票中,实际下跌的比例 |",
        f"| **IC (Spearman)** | {ic:+.3f} (p={pval:.3f}) | alpha_z 与实际涨跌的秩相关,>0.05 算有效 |",
        f"| **多空价差** | {ls_spread:+.2f}% | top{top_n} 平均涨跌 - bottom{top_n} 平均涨跌 |",
        f"| **全体平均** | {valid['实际涨跌%'].mean():+.2f}% | 大盘同期对比 |",
        "",
        "## 📊 Top 5 alpha 实际表现",
        "",
    ]
    for _, r in ranked.head(5).iterrows():
        summary.append(f"- {r['name']} ({r['code']}): z={r['alpha_z']:+.2f}, 实际 {r['实际涨跌%']:+.2f}% {r['验证结果']}")

    summary.append("")
    summary.append("## 📊 Bottom 5 alpha 实际表现")
    summary.append("")
    for _, r in ranked.tail(5).iterrows():
        summary.append(f"- {r['name']} ({r['code']}): z={r['alpha_z']:+.2f}, 实际 {r['实际涨跌%']:+.2f}% {r['验证结果']}")

    summary.append("")
    summary.append("## 🔧 模型调整建议")
    summary.append("")
    if ic > 0.1:
        summary.append(f"- ✅ IC={ic:+.3f} 显著为正,因子权重方向正确,保持")
    elif ic > 0.03:
        summary.append(f"- ⚠️ IC={ic:+.3f} 微弱为正,信号不强,考虑加 Barra 中性化")
    elif ic > -0.03:
        summary.append(f"- ⚠️ IC={ic:+.3f} 接近 0,信号 ≈ 随机,单日 OOS 样本小不必过度反应")
    else:
        summary.append(f"- ❌ IC={ic:+.3f} 显著为负! 若多日持续,因子极性需要翻转 (短期反转 → 短期动量)")

    if long_hit < 0.4:
        summary.append(f"- ❌ 看多命中率 {long_hit:.1%} 低于 40%,检查 REV_5/REV_20 短反转是否过度依赖")
    if short_hit < 0.4:
        summary.append(f"- ❌ 看空命中率 {short_hit:.1%} 低于 40%,MAX_RET_5 单日暴涨负向因子可能已失效")

    summary.append("")
    summary.append(f"> 单日样本仅 {len(valid)} 点,任何结论都需≥20 个交易日样本才可靠. 本报告仅做即时参考.")

    SUMMARY_MD.write_text("\n".join(summary), encoding="utf-8")

    # 7. 打印
    print(f"\n  ✅ 看多档命中率: {long_hit:.1%} ({(long_signals['实际涨跌%']>0).sum()}/{len(long_signals)})")
    print(f"  ✅ 看空档命中率: {short_hit:.1%} ({(short_signals['实际涨跌%']<0).sum()}/{len(short_signals)})")
    print(f"  📊 IC(Spearman): {ic:+.3f}  p={pval:.3f}")
    print(f"  📊 多空价差: {ls_spread:+.2f}% (top{top_n} - bottom{top_n})")
    print(f"  📊 全体平均: {valid['实际涨跌%'].mean():+.2f}%")
    print(f"\n  📄 CSV: {OUT_CSV}")
    print(f"  📄 摘要: {SUMMARY_MD}")

    # 8. 推送飞书
    if not args.dry_run_lark:
        lark_msg = "\n".join(summary)
        send_lark(lark_msg)


if __name__ == "__main__":
    main()
