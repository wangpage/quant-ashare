"""生成人类可读的日报 / 研究报告."""
from __future__ import annotations

from datetime import date
from typing import Any

from tabulate import tabulate


def build_daily_report(decision, out_path: str | None = None) -> str:
    """生成每日决策报告."""
    lines = []
    lines.append("=" * 60)
    lines.append(f"  每日决策报告 - {decision.date}")
    lines.append("=" * 60)
    lines.append(f"\n【Market Regime】 {decision.regime}")
    lines.append(f"仓位乘数: {decision.position_mult:.2f}")
    for n in decision.notes[:3]:
        lines.append(f"  - {n}")

    if decision.candidates:
        lines.append("\n【候选股票 Top 10】")
        rows = [[c.get("rank"), c.get("code"),
                 f"{c.get('score', 0):.4f}",
                 c.get("agent_action", "N/A"),
                 f"{c.get('agent_conviction', 0):.2f}" if c.get("agent_conviction") else "N/A",
                 f"{c.get('alloc_cny', 0):.0f}",
                 "❌" if c.get("risk_filtered") else "✓"]
                for c in decision.candidates[:10]]
        lines.append(tabulate(
            rows, headers=["排名", "代码", "模型分", "Agent", "信心", "配资", "通过"],
        ))

    if decision.orders:
        lines.append("\n【下单计划】")
        rows = [[o["code"], o["shares"], f"{o['ref_price']:.2f}",
                 f"{o['cost_bps']:.1f} bps", o["n_slices"]]
                for o in decision.orders]
        lines.append(tabulate(
            rows, headers=["代码", "股数", "参考价", "冲击成本", "拆片"],
        ))

        total_cost = sum(o["cost_bps"] * o["shares"] * o["ref_price"] / 1e4
                         for o in decision.orders)
        total_notional = sum(o["shares"] * o["ref_price"] for o in decision.orders)
        lines.append(f"\n组合总 notional: ¥{total_notional:,.0f}")
        lines.append(f"组合总冲击成本: ¥{total_cost:,.0f} "
                     f"({total_cost/total_notional*1e4:.1f} bps)"
                     if total_notional else "")

    if decision.reflections_saved > 0:
        lines.append(f"\n【记忆】已存入 {decision.reflections_saved} 条交易反思")

    report = "\n".join(lines)

    if out_path:
        from pathlib import Path
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(report, encoding="utf-8")

    return report


def build_research_report(result, out_path: str | None = None) -> str:
    """生成研究报告."""
    lines = [result.summary()]

    # 各阶段 artifact 信息
    lines.append("\n" + "=" * 60)
    lines.append("  各阶段输出")
    lines.append("=" * 60)
    for stage, info in result.stage_results.items():
        if stage.startswith("_"):
            continue
        lines.append(f"\n【{stage}】")
        if isinstance(info, dict):
            for k, v in info.items():
                if isinstance(v, (list, dict)):
                    lines.append(f"  {k}: {str(v)[:150]}")
                else:
                    lines.append(f"  {k}: {v}")

    # 最终判定
    lines.append("\n" + "=" * 60)
    if result.errors:
        lines.append("  ❌ 研究失败 (有 critical 错误)")
    elif result.warnings:
        lines.append("  ⚠️  研究完成, 但有警告")
    else:
        lines.append("  ✅ 研究通过, 可考虑进入样本外测试")
    lines.append("=" * 60)

    report = "\n".join(lines)
    if out_path:
        from pathlib import Path
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(report, encoding="utf-8")
    return report
