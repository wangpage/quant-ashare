"""对 2026-04-21 信号生成"明日涨跌预测 + 加减仓建议"CSV + 飞书推送.

逻辑:
- alpha_z 基于 7 个反转/低波因子合成, 属于日频短期反转信号.
- z >= 0.75 → 预测明日偏涨, 建议加仓/持有
- z <= -0.75 → 预测明日偏跌, 建议减仓/清仓
- 中间区 → 方向不明, 持有观察
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

ROOT = Path(__file__).resolve().parent.parent
LARK_BIN = "/opt/homebrew/bin/lark-cli"

env = ROOT / ".env"
if env.exists():
    for line in env.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

SIG_CSV = ROOT / "output" / "paper_trade" / "watchlist_signals" / "2026-04-21.csv"
OUT_CSV = Path("/Users/page/Desktop/股票/预测_20260421_for_0422.csv")


def send_lark(markdown: str) -> bool:
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


def tier_label(z: float) -> tuple[str, str, str]:
    if z >= 1.5:
        return "🟩 强买入", "偏涨 (高置信)", "加仓 4-5%"
    if z >= 1.0:
        return "🟢 买入", "偏涨", "加仓 3%"
    if z >= 0.75:
        return "🟢 买入", "偏涨", "加仓 2%"
    if z >= -0.75:
        return "⚪️ 持有", "方向不明", "不操作"
    if z >= -1.5:
        return "🟡 减仓", "偏跌", "减仓 1/3"
    return "🟥 清仓", "偏跌 (高置信)", "清仓或大幅减仓"


def build_markdown(df: pd.DataFrame) -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    buy = df[df["alpha_z"] >= 0.75].sort_values("alpha_z", ascending=False)
    sell = df[df["alpha_z"] <= -0.75].sort_values("alpha_z")
    lines = [
        f"📋 **明日逐只预测 — {today}**",
        "",
        f"基础: 72 只自选池 7 因子截面 alpha",
        f"明日交易日: 2026-04-22",
        "",
    ]
    if len(buy):
        lines.append(f"**🟢 加仓/买入清单 ({len(buy)} 只)**")
        for _, r in buy.iterrows():
            lines.append(f"  • {r['股票名称']} {r['股票代码']} "
                         f"¥{r['收盘价(0421)']:.2f}  z={r['alpha_z']:+.2f}  "
                         f"[{r['主导因子']}] → {r['操作建议']}")
        lines.append("")
    if len(sell):
        lines.append(f"**🟡🟥 减仓/清仓清单 ({len(sell)} 只,若持有)**")
        for _, r in sell.iterrows():
            lines.append(f"  • {r['股票名称']} {r['股票代码']} "
                         f"¥{r['收盘价(0421)']:.2f}  z={r['alpha_z']:+.2f}  "
                         f"[{r['主导因子']}] → {r['操作建议']}")
        lines.append("")
    lines.append(f"**⚪️ 持有观察 ({len(df)-len(buy)-len(sell)} 只)**: 方向不明,不操作")
    lines.append("")
    lines.append("⚠️ 明早 9:30 开盘后自动拉实时数据验证此预测并推送命中率.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run-lark", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(SIG_CSV)
    df = df.sort_values("alpha_z", ascending=False).reset_index(drop=True)

    out = []
    for rank, r in enumerate(df.itertuples(index=False), 1):
        tier, pred, action = tier_label(r.alpha_z)
        out.append({
            "排名": rank,
            "股票名称": r.name,
            "股票代码": str(r.code).zfill(6),
            "收盘价(0421)": round(r.latest_close, 2),
            "alpha_z": round(r.alpha_z, 3),
            "信号档": tier,
            "明日(0422)预测": pred,
            "操作建议": action,
            "主导因子": f"{r.driver_sign}{r.top_driver}",
            "信心等级": round(abs(r.alpha_z), 2),
        })

    out_df = pd.DataFrame(out)
    out_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"✓ 已保存预测报告: {OUT_CSV}")
    print(f"  共 {len(out_df)} 只")
    print(f"  🟢+ 买入档: {(out_df['信号档'].str.contains('买入')).sum()} 只")
    print(f"  ⚪️  持有档: {(out_df['信号档'].str.contains('持有')).sum()} 只")
    print(f"  🟡+ 减仓档: {(out_df['信号档'].str.contains('减仓')).sum()} 只")
    print(f"  🟥  清仓档: {(out_df['信号档'].str.contains('清仓')).sum()} 只")

    if not args.dry_run_lark:
        md = build_markdown(out_df)
        send_lark(md)


if __name__ == "__main__":
    main()
