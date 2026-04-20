"""两份研究报告对比 HTML - 左右并排 KPI + 叠加净值曲线.

典型用法: baseline (3 玩具因子) vs Alpha158-lite + Barra 残差化.
一眼看到 "换策略带来多少夏普改善".
"""
from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path

try:
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False


def _stat(result, key: str, default=0.0):
    return result.backtest_stats.get(key, default)


def _ic(result, key: str, default=0.0):
    return result.ic_stats.get(key, default)


def _delta_color(diff: float, higher_better: bool = True) -> str:
    if diff == 0:
        return "#95a5a6"
    win = (diff > 0) if higher_better else (diff < 0)
    return "#27ae60" if win else "#e74c3c"


def _arrow(diff: float, higher_better: bool = True) -> str:
    if abs(diff) < 1e-9:
        return "—"
    win = (diff > 0) if higher_better else (diff < 0)
    return "▲" if diff > 0 else "▼"


def _nav_overlay(result_a, result_b, name_a: str, name_b: str) -> str:
    if not _HAS_PLOTLY:
        return "<div>plotly 不可用</div>"
    nav_a = result_a.stage_results.get("_bt_nav", [])
    nav_b = result_b.stage_results.get("_bt_nav", [])
    fig = go.Figure()
    if nav_a:
        fig.add_trace(go.Scatter(
            x=list(range(len(nav_a))), y=nav_a, mode="lines",
            name=name_a, line=dict(color="#95a5a6", width=2),
        ))
    if nav_b:
        fig.add_trace(go.Scatter(
            x=list(range(len(nav_b))), y=nav_b, mode="lines",
            name=name_b, line=dict(color="#e74c3c", width=2.5),
        ))
    fig.add_hline(y=1.0, line_dash="dot", line_color="#bdc3c7")
    fig.update_layout(
        title=dict(text="净值曲线对比 (起点 = 1.0)", x=0.5),
        height=420, hovermode="x unified",
        paper_bgcolor="white", plot_bgcolor="#fafafa",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(t=60, b=40, l=50, r=20),
    )
    fig.update_xaxes(title_text="调仓期索引")
    fig.update_yaxes(title_text="净值")
    return fig.to_html(include_plotlyjs=False, full_html=False,
                        div_id="nav-overlay")


def _kpi_compare_row(
    label: str, a_val: float, b_val: float,
    fmt: str = "{:.4f}", higher_better: bool = True,
    unit: str = "",
) -> str:
    diff = b_val - a_val
    color = _delta_color(diff, higher_better)
    arrow = _arrow(diff, higher_better)
    pct = (b_val / a_val - 1) * 100 if abs(a_val) > 1e-9 else 0.0
    return f"""
    <tr>
      <td class="metric-label">{html.escape(label)}</td>
      <td class="metric-a">{fmt.format(a_val)}{unit}</td>
      <td class="metric-b">{fmt.format(b_val)}{unit}</td>
      <td class="metric-delta" style="color:{color};">
        {arrow} {fmt.format(abs(diff))}{unit}
        <span class="metric-pct">({pct:+.1f}%)</span>
      </td>
    </tr>
    """


def build_compare_report(
    result_a, result_b,
    name_a: str = "Baseline",
    name_b: str = "Alpha158 + Barra 残差",
    out_path: str | Path | None = None,
) -> str:
    """生成对比 HTML 报告."""
    rows = []
    rows.append(_kpi_compare_row(
        "夏普比率", float(_stat(result_a, "sharpe")),
        float(_stat(result_b, "sharpe")),
        fmt="{:.3f}",
    ))
    rows.append(_kpi_compare_row(
        "年化收益", float(_stat(result_a, "annual_return")),
        float(_stat(result_b, "annual_return")),
        fmt="{:.2%}", unit="",
    ))
    rows.append(_kpi_compare_row(
        "最大回撤", float(_stat(result_a, "max_drawdown")),
        float(_stat(result_b, "max_drawdown")),
        fmt="{:.2%}", higher_better=True,  # 回撤越小 (less negative) 越好
    ))
    rows.append(_kpi_compare_row(
        "胜率", float(_stat(result_a, "win_rate")),
        float(_stat(result_b, "win_rate")),
        fmt="{:.1%}",
    ))
    rows.append(_kpi_compare_row(
        "最终净值", float(_stat(result_a, "final_nav", 1.0)),
        float(_stat(result_b, "final_nav", 1.0)),
        fmt="{:.3f}",
    ))
    rows.append(_kpi_compare_row(
        "换手率 (每期)", float(_stat(result_a, "avg_turnover")),
        float(_stat(result_b, "avg_turnover")),
        fmt="{:.2%}", higher_better=False,   # 越低越好 (省手续费)
    ))
    rows.append(_kpi_compare_row(
        "IC 均值", float(_ic(result_a, "ic_mean")),
        float(_ic(result_b, "ic_mean")),
        fmt="{:.4f}",
    ))
    rows.append(_kpi_compare_row(
        "IC t-stat", float(_ic(result_a, "ic_t_stat")),
        float(_ic(result_b, "ic_t_stat")),
        fmt="{:.2f}",
    ))
    rows.append(_kpi_compare_row(
        "ICIR", float(_ic(result_a, "icir")),
        float(_ic(result_b, "icir")),
        fmt="{:.3f}",
    ))
    rows.append(_kpi_compare_row(
        "Gate 拒单次数",
        float(_stat(result_a, "gate_hard_rejects")),
        float(_stat(result_b, "gate_hard_rejects")),
        fmt="{:.0f}", higher_better=True,  # 拒单多说明风控更严
    ))
    kpi_table = "\n".join(rows)

    # 判定
    sharpe_a = float(_stat(result_a, "sharpe"))
    sharpe_b = float(_stat(result_b, "sharpe"))
    diff = sharpe_b - sharpe_a
    verdict_text, verdict_color = (
        (f"✅ 升级成功, 夏普 +{diff:.2f}", "#27ae60") if diff > 0.1
        else (f"⚠️ 小幅改善, 夏普 +{diff:.2f}", "#f39c12") if diff > 0
        else (f"❌ 未改善, 夏普 {diff:+.2f}", "#e74c3c")
    )

    nav_overlay = _nav_overlay(result_a, result_b, name_a, name_b)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    plotly_cdn = "https://cdn.plot.ly/plotly-2.27.0.min.js" if _HAS_PLOTLY else ""

    doc = f"""<!DOCTYPE html>
<html lang="zh-CN"><head>
<meta charset="UTF-8">
<title>策略对比报告</title>
<script src="{plotly_cdn}"></script>
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; font-family: -apple-system, "PingFang SC", sans-serif;
         background: #ecf0f1; color: #2c3e50; }}
  .container {{ max-width: 1100px; margin: 0 auto; padding: 24px; }}
  .header {{ background: linear-gradient(135deg,#2c3e50,#34495e);
             color: white; padding: 24px; border-radius: 8px;
             margin-bottom: 20px; }}
  .header h1 {{ margin: 0 0 4px; }}
  .header .sub {{ opacity: 0.85; font-size: 14px; }}
  .verdict {{ background: {verdict_color}; color: white; padding: 16px;
              border-radius: 8px; text-align: center; font-size: 18px;
              font-weight: bold; margin-bottom: 20px; }}
  .compare-card {{ background: white; padding: 20px; border-radius: 8px;
                     box-shadow: 0 1px 3px rgba(0,0,0,0.08);
                     margin-bottom: 20px; }}
  .strategy-names {{ display: grid; grid-template-columns: 1fr 1fr 1fr 1fr;
                       gap: 12px; margin-bottom: 12px; font-weight: 600; }}
  .strategy-names .sa {{ color: #95a5a6; }}
  .strategy-names .sb {{ color: #e74c3c; }}
  .compare-table {{ width: 100%; border-collapse: collapse; }}
  .compare-table th {{ text-align: left; padding: 10px 12px;
                        background: #ecf0f1; font-size: 12px;
                        color: #7f8c8d; text-transform: uppercase;
                        letter-spacing: 0.5px; border-bottom: 2px solid #bdc3c7; }}
  .compare-table td {{ padding: 10px 12px; border-bottom: 1px solid #ecf0f1;
                        font-family: "SF Mono", Consolas, monospace; font-size: 14px; }}
  .metric-label {{ font-family: inherit !important; font-weight: 600;
                     color: #34495e; }}
  .metric-a {{ color: #95a5a6; }}
  .metric-b {{ color: #2c3e50; font-weight: 600; }}
  .metric-delta {{ font-weight: 600; }}
  .metric-pct {{ font-size: 11px; opacity: 0.7; }}
</style></head><body>
<div class="container">
  <div class="header">
    <h1>🆚 策略对比报告</h1>
    <div class="sub">A: <strong>{html.escape(name_a)}</strong> &nbsp;·&nbsp;
                     B: <strong>{html.escape(name_b)}</strong></div>
    <div class="sub">生成: {ts}</div>
  </div>

  <div class="verdict">{verdict_text}</div>

  <div class="compare-card">
    <table class="compare-table">
      <thead>
        <tr>
          <th>指标</th>
          <th>A: {html.escape(name_a)}</th>
          <th>B: {html.escape(name_b)}</th>
          <th>差值 (B - A)</th>
        </tr>
      </thead>
      <tbody>{kpi_table}</tbody>
    </table>
  </div>

  <div class="compare-card">
    {nav_overlay}
  </div>

  <div style="text-align:center; color:#95a5a6; font-size:12px; padding:20px;">
    pipeline.reporting_compare · {ts}
  </div>
</div></body></html>"""

    if out_path:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(doc, encoding="utf-8")
    return doc
