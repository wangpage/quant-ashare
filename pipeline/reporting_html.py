"""研究报告 HTML 版本 - plotly 交互图表 + 结构化面板.

布局 (单页):
    [头部]       项目名 / 时间 / 总体 verdict
    [KPI 卡片]   夏普 / 年化 / 回撤 / 胜率 (彩色大数字)
    [图表区]
        Row1: 净值曲线 + 回撤曲线 (dual-axis)
        Row2: 每期收益分布 + 换手率时序
        Row3: IC 统计 + 拒单饼图
    [诊断段]
        Barra 诊断表
        Lookahead 报告表
        Gate 拒单原因 Top-10
    [原始 stage 输出]  折叠式 <details>

依赖 plotly (requirements.txt 已有).
"""
from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False


# ============ KPI 卡片阈值 ============
_SHARPE_BANDS = [(2.0, "#2ecc71"), (1.5, "#27ae60"), (1.0, "#f39c12"),
                  (0.5, "#e67e22"), (-1e9, "#e74c3c")]
_DD_BANDS = [(-0.05, "#2ecc71"), (-0.10, "#f39c12"), (-0.20, "#e67e22"),
              (-1e9, "#e74c3c")]


def _color_for(value: float, bands: list) -> str:
    for threshold, color in bands:
        if value >= threshold:
            return color
    return bands[-1][1]


def _kpi_card(label: str, value: str, color: str, hint: str = "") -> str:
    return f"""
    <div class="kpi-card" style="border-left: 5px solid {color};">
      <div class="kpi-label">{html.escape(label)}</div>
      <div class="kpi-value" style="color: {color};">{html.escape(str(value))}</div>
      <div class="kpi-hint">{html.escape(hint)}</div>
    </div>
    """


def _nav_drawdown_figure(nav: list, drawdown: list) -> str:
    if not _HAS_PLOTLY or not nav:
        return "<div class='plot-skip'>净值曲线不可用</div>"
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.7, 0.3], vertical_spacing=0.05,
        subplot_titles=("累计净值曲线", "回撤曲线"),
    )
    x = list(range(len(nav)))
    fig.add_trace(
        go.Scatter(x=x, y=nav, mode="lines", name="NAV",
                    line=dict(color="#2980b9", width=2)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=x, y=[d * 100 for d in drawdown], mode="lines",
                    name="Drawdown %", fill="tozeroy",
                    line=dict(color="#e74c3c", width=1)),
        row=2, col=1,
    )
    fig.update_layout(
        height=500, showlegend=False, margin=dict(t=40, b=30, l=50, r=20),
        paper_bgcolor="white", plot_bgcolor="#fafafa",
    )
    fig.update_yaxes(title_text="净值", row=1, col=1)
    fig.update_yaxes(title_text="回撤 %", row=2, col=1)
    fig.update_xaxes(title_text="调仓期索引", row=2, col=1)
    return fig.to_html(include_plotlyjs=False, full_html=False,
                        div_id="nav-dd-chart")


def _returns_turnover_figure(returns: list, turnovers: list) -> str:
    if not _HAS_PLOTLY or not returns:
        return "<div class='plot-skip'>收益/换手不可用</div>"
    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.5, 0.5],
        subplot_titles=("每期收益分布", "换手率时序"),
    )
    fig.add_trace(
        go.Histogram(x=[r * 100 for r in returns], nbinsx=30,
                      marker_color="#3498db", name="Returns %"),
        row=1, col=1,
    )
    if turnovers:
        fig.add_trace(
            go.Scatter(x=list(range(len(turnovers))),
                        y=[t * 100 for t in turnovers],
                        mode="lines+markers", line=dict(color="#9b59b6"),
                        name="Turnover %"),
            row=1, col=2,
        )
    fig.update_layout(
        height=320, showlegend=False,
        margin=dict(t=40, b=30, l=50, r=20),
        paper_bgcolor="white", plot_bgcolor="#fafafa",
    )
    fig.update_xaxes(title_text="单期收益 %", row=1, col=1)
    fig.update_xaxes(title_text="调仓期", row=1, col=2)
    return fig.to_html(include_plotlyjs=False, full_html=False,
                        div_id="returns-turnover-chart")


def _reject_pie(reject_reasons: list) -> str:
    if not _HAS_PLOTLY or not reject_reasons:
        return "<div class='plot-skip'>无拒单</div>"
    labels = [r[0][:30] for r in reject_reasons[:8]]
    values = [r[1] for r in reject_reasons[:8]]
    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values, hole=0.4,
        marker=dict(colors=["#e74c3c", "#e67e22", "#f39c12", "#c0392b",
                             "#d35400", "#8e44ad", "#16a085", "#2c3e50"]),
    )])
    fig.update_layout(
        title=dict(text="Gate 拒单原因 Top-8", x=0.5),
        height=320, margin=dict(t=50, b=20, l=20, r=20),
        paper_bgcolor="white",
    )
    return fig.to_html(include_plotlyjs=False, full_html=False,
                        div_id="reject-pie")


def _ic_bar(ic_stats: dict) -> str:
    if not _HAS_PLOTLY or not ic_stats:
        return "<div class='plot-skip'>无 IC 数据</div>"
    metrics = ["ic_mean", "ic_std", "icir", "ic_t_stat"]
    values = [float(ic_stats.get(m, 0)) for m in metrics]
    colors = ["#27ae60" if abs(v) > 0.02 else "#95a5a6" for v in values]
    fig = go.Figure(data=[go.Bar(
        x=metrics, y=values, marker_color=colors,
        text=[f"{v:.4f}" for v in values], textposition="outside",
    )])
    fig.update_layout(
        title=dict(text="IC 统计", x=0.5),
        height=320, margin=dict(t=50, b=30, l=50, r=20),
        paper_bgcolor="white", plot_bgcolor="#fafafa",
        yaxis=dict(zeroline=True, zerolinecolor="#bdc3c7"),
    )
    return fig.to_html(include_plotlyjs=False, full_html=False,
                        div_id="ic-bar")


def _table_from_dict(d: dict, title: str = "") -> str:
    if not d:
        return f"<p class='empty'>无 {html.escape(title)} 数据</p>"
    rows = []
    for k, v in d.items():
        if isinstance(v, float):
            v_str = f"{v:.4f}"
        elif isinstance(v, (list, tuple)):
            v_str = ", ".join(str(x)[:30] for x in v[:5])
        elif isinstance(v, dict):
            v_str = json.dumps(v, ensure_ascii=False)[:100]
        else:
            v_str = str(v)[:120]
        rows.append(
            f"<tr><td>{html.escape(str(k))}</td>"
            f"<td>{html.escape(v_str)}</td></tr>"
        )
    return (
        f"<table class='diag-table'>"
        f"<thead><tr><th>字段</th><th>值</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _verdict_banner(result) -> tuple[str, str]:
    """返回 (text, color). 优先级: errors > warnings > pass."""
    if result.errors:
        return ("❌ 研究失败 (有 CRITICAL 错误)", "#e74c3c")
    if result.lookahead_report.get("verdict") == "FAIL":
        return ("❌ 前视偏差 CRITICAL, 信号不可信", "#e74c3c")
    if result.warnings:
        return ("⚠️ 研究通过, 但有警告 (见下方)", "#f39c12")
    return ("✅ 研究通过, 可进入样本外测试", "#27ae60")


def build_research_report_html(
    result, out_path: str | Path | None = None,
    project_name: str = "quant-ashare",
) -> str:
    """生成 HTML 研究报告. 返回 HTML 字符串; out_path 时同时写文件."""
    bt = result.backtest_stats
    ic = result.ic_stats
    sharpe = float(bt.get("sharpe", 0))
    annual = float(bt.get("annual_return", 0))
    max_dd = float(bt.get("max_drawdown", 0))
    win_rate = float(bt.get("win_rate", 0))
    final_nav = float(bt.get("final_nav", 1.0))

    verdict_text, verdict_color = _verdict_banner(result)

    kpi_html = "".join([
        _kpi_card(
            "夏普比率", f"{sharpe:.2f}",
            _color_for(sharpe, _SHARPE_BANDS),
            "≥1.5 头部; ≥2.0 机构级",
        ),
        _kpi_card(
            "年化收益", f"{annual*100:.2f}%",
            _color_for(annual, [(0.2, "#27ae60"), (0.1, "#f39c12"),
                                 (0, "#e67e22"), (-1e9, "#e74c3c")]),
            f"最终净值 {final_nav:.3f}",
        ),
        _kpi_card(
            "最大回撤", f"{max_dd*100:.2f}%",
            _color_for(max_dd, _DD_BANDS),
            "< -15% 建议熔断",
        ),
        _kpi_card(
            "胜率", f"{win_rate*100:.1f}%",
            _color_for(win_rate, [(0.55, "#27ae60"), (0.5, "#f39c12"),
                                    (-1e9, "#e74c3c")]),
            f"调仓 {bt.get('n_rebalances', 0)} 次",
        ),
        _kpi_card(
            "换手率", f"{bt.get('avg_turnover', 0)*100:.2f}%",
            "#3498db", "每期平均",
        ),
        _kpi_card(
            "IC 均值", f"{float(ic.get('ic_mean', 0)):.4f}",
            _color_for(abs(float(ic.get("ic_mean", 0))),
                        [(0.05, "#27ae60"), (0.02, "#f39c12"),
                         (-1e9, "#95a5a6")]),
            f"t = {float(ic.get('ic_t_stat', 0)):.2f}",
        ),
    ])

    nav_fig = _nav_drawdown_figure(
        result.stage_results.get("_bt_nav", []),
        result.stage_results.get("_bt_drawdown", []),
    )
    rt_fig = _returns_turnover_figure(
        result.stage_results.get("_bt_returns", []),
        result.stage_results.get("_bt_turnovers", []),
    )
    reject_fig = _reject_pie(result.gate_stats.get("top_reject", []))
    ic_fig = _ic_bar(ic)

    # 诊断段
    barra_diag = _table_from_dict(result.neutralize_diagnostics, "Barra 诊断")
    lookahead_diag = _table_from_dict(
        {k: v for k, v in result.lookahead_report.items()
         if k != "suspicious_features"},
        "Lookahead",
    )
    gate_diag = _table_from_dict(result.gate_stats, "Gate 统计")

    warnings_html = ""
    if result.warnings:
        items = "".join(f"<li>{html.escape(w)}</li>" for w in result.warnings)
        warnings_html = f"<ul class='warn-list'>{items}</ul>"

    errors_html = ""
    if result.errors:
        items = "".join(f"<li>{html.escape(e)}</li>" for e in result.errors)
        errors_html = f"<ul class='error-list'>{items}</ul>"

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    plotly_cdn = (
        "https://cdn.plot.ly/plotly-2.27.0.min.js"
        if _HAS_PLOTLY else ""
    )

    html_doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>{html.escape(project_name)} Research Report</title>
<script src="{plotly_cdn}"></script>
<style>
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; font-family: -apple-system, "Helvetica Neue", "PingFang SC",
                             Arial, sans-serif;
    background: #ecf0f1; color: #2c3e50; line-height: 1.5;
  }}
  .container {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}
  .header {{
    background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
    color: white; padding: 24px; border-radius: 8px;
    margin-bottom: 20px;
  }}
  .header h1 {{ margin: 0 0 8px; font-size: 24px; }}
  .header .subtitle {{ opacity: 0.8; font-size: 14px; }}
  .verdict {{
    padding: 16px; border-radius: 8px; color: white;
    font-size: 18px; font-weight: bold; text-align: center;
    background: {verdict_color}; margin-bottom: 20px;
  }}
  .kpi-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    gap: 12px; margin-bottom: 24px;
  }}
  .kpi-card {{
    background: white; padding: 16px; border-radius: 6px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
  }}
  .kpi-label {{ font-size: 12px; color: #7f8c8d;
                 text-transform: uppercase; letter-spacing: 0.5px; }}
  .kpi-value {{ font-size: 28px; font-weight: bold; margin: 6px 0; }}
  .kpi-hint {{ font-size: 11px; color: #95a5a6; }}
  .chart-section {{
    background: white; padding: 16px; border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08); margin-bottom: 20px;
  }}
  .chart-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  @media (max-width: 800px) {{ .chart-row {{ grid-template-columns: 1fr; }} }}
  .diag-section {{
    background: white; padding: 20px; border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08); margin-bottom: 20px;
  }}
  .diag-section h2 {{
    margin: 0 0 12px; font-size: 16px; color: #34495e;
    border-bottom: 2px solid #3498db; padding-bottom: 6px;
  }}
  .diag-table {{
    width: 100%; border-collapse: collapse; font-size: 13px;
  }}
  .diag-table th {{
    text-align: left; padding: 8px 10px; background: #ecf0f1;
    color: #34495e; border-bottom: 1px solid #bdc3c7;
  }}
  .diag-table td {{
    padding: 6px 10px; border-bottom: 1px solid #ecf0f1;
    font-family: "SF Mono", Consolas, monospace;
  }}
  .diag-table td:first-child {{ color: #7f8c8d; }}
  .warn-list li {{ color: #e67e22; margin: 4px 0; }}
  .error-list li {{ color: #e74c3c; font-weight: bold; }}
  .empty {{ color: #95a5a6; font-style: italic; }}
  .plot-skip {{ padding: 40px; text-align: center; color: #95a5a6; }}
  details summary {{
    cursor: pointer; padding: 8px 0; color: #7f8c8d;
    font-size: 14px;
  }}
  pre {{
    background: #2c3e50; color: #ecf0f1; padding: 12px;
    border-radius: 4px; overflow-x: auto; font-size: 12px;
  }}
</style>
</head>
<body>
<div class="container">

  <div class="header">
    <h1>📊 {html.escape(project_name)} - 研究报告</h1>
    <div class="subtitle">生成时间: {ts}</div>
  </div>

  <div class="verdict">{verdict_text}</div>

  <div class="kpi-grid">{kpi_html}</div>

  <div class="chart-section">
    <h2 style="margin:0 0 12px; font-size:16px; color:#34495e;
                border-bottom:2px solid #3498db; padding-bottom:6px;">
      📈 回测净值 & 回撤
    </h2>
    {nav_fig}
  </div>

  <div class="chart-section">
    <div class="chart-row">
      <div>{rt_fig}</div>
    </div>
  </div>

  <div class="chart-section">
    <div class="chart-row">
      <div>{ic_fig}</div>
      <div>{reject_fig}</div>
    </div>
  </div>

  <div class="diag-section">
    <h2>🧬 Barra 中性化诊断</h2>
    {barra_diag}
  </div>

  <div class="diag-section">
    <h2>🔍 Lookahead 前视偏差扫描</h2>
    {lookahead_diag}
  </div>

  <div class="diag-section">
    <h2>🛡️ PreTradeGate 风控统计</h2>
    {gate_diag}
  </div>

  {"<div class='diag-section'><h2>⚠️ 警告</h2>" + warnings_html + "</div>"
    if warnings_html else ""}
  {"<div class='diag-section'><h2>❌ 错误</h2>" + errors_html + "</div>"
    if errors_html else ""}

  <details class="diag-section">
    <summary>原始 stage 输出 (调试用)</summary>
    <pre>{html.escape(_format_stages(result.stage_results))}</pre>
  </details>

  <div style="text-align:center; color:#95a5a6; font-size:12px;
               padding: 20px;">
    Generated by pipeline.reporting_html · plotly {_HAS_PLOTLY}
  </div>
</div>
</body></html>"""

    if out_path:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(html_doc, encoding="utf-8")
    return html_doc


def _format_stages(stage_results: dict) -> str:
    """折叠内部键 (_ 开头), 只显示公开信息."""
    lines = []
    for k, v in stage_results.items():
        if k.startswith("_"):
            continue
        lines.append(f"[{k}]")
        if isinstance(v, dict):
            for kk, vv in v.items():
                lines.append(f"  {kk}: {str(vv)[:120]}")
        else:
            lines.append(f"  {str(v)[:200]}")
        lines.append("")
    return "\n".join(lines)
