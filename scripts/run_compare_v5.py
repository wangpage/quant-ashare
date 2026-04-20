"""v5 A/B \u5bf9\u6bd4: v3 baseline vs v5 (\u542f\u7528\u5168\u90e8\u65b0\u56e0\u5b50).

\u4e24\u6b21\u8fd0\u884c run_real_research_v5.main:
    baseline: \u4ec5 v3 \u539f\u6709\u6280\u672f+\u53cd\u8f6c+\u65e7 LHB \u56e0\u5b50
    enhanced: \u52a0 microstructure + seat_network + intraday (proxy)

\u5904\u7406:
    - \u53ef\u9009 --with_ann \u540c\u65f6\u542f\u7528\u516c\u544a\u56e0\u5b50
    - \u8f93\u51fa HTML \u5e76\u6392\u62a5\u544a
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scripts.run_real_research_v5 import main as run_v5

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"


def _cum_nav(daily_returns: list[float]) -> list[float]:
    """\u7b80\u6613\u7d2f\u79ef NAV."""
    nav, cur = [1.0], 1.0
    for r in daily_returns:
        cur *= (1 + r)
        nav.append(cur)
    return nav


def _summarize(stats: dict) -> dict:
    keys = ["sharpe", "annual_return", "annual_vol", "max_drawdown",
            "info_ratio", "excess_return", "excess_max_dd",
            "avg_turnover", "n_rebalances"]
    return {k: stats.get(k, 0) for k in keys}


def _delta_cell(a, b, is_pct=False, is_higher_better=True, prec=2):
    d = b - a
    color = "#4caf50" if (d > 0) == is_higher_better else "#e53935"
    if abs(d) < 1e-6:
        color = "#999"
    arrow = "\u2191" if d > 0 else "\u2193"
    if is_pct:
        return f'<span style="color:{color}">{arrow} {d*100:+.{prec}f}%</span>'
    return f'<span style="color:{color}">{arrow} {d:+.{prec}f}</span>'


def _build_html(res_a: dict, res_b: dict, name_a: str, name_b: str,
                out_path: Path):
    """\u5bf9\u6bd4\u62a5\u544a."""
    sa, sb = res_a["stats"], res_b["stats"]

    # KPI \u8868
    rows = [
        ("Sharpe",          sa["sharpe"],          sb["sharpe"],          False, True, 2),
        ("\u5e74\u5316\u6536\u76ca",       sa["annual_return"],   sb["annual_return"],   True,  True, 2),
        ("\u5e74\u5316\u6ce2\u52a8",       sa["annual_vol"],      sb["annual_vol"],      True,  False, 2),
        ("\u6700\u5927\u56de\u64a4",       sa["max_drawdown"],    sb["max_drawdown"],    True,  True, 2),
        ("\u8d85\u989d\u6536\u76ca",       sa["excess_return"],   sb["excess_return"],   True,  True, 2),
        ("\u4fe1\u606f\u6bd4\u7387",       sa["info_ratio"],      sb["info_ratio"],      False, True, 2),
        ("\u8d85\u989d\u56de\u64a4",       sa["excess_max_dd"],   sb["excess_max_dd"],   True,  True, 4),
        ("\u6362\u624b\u7387",           sa["avg_turnover"],    sb["avg_turnover"],    True,  False, 2),
    ]
    kpi_html = "<tr><th style='padding:8px'>\u6307\u6807</th>" \
               f"<th>{name_a}</th><th>{name_b}</th><th>\u5dee\u5f02</th></tr>"
    for label, va, vb, is_pct, hib, prec in rows:
        fa = f"{va*100:.{prec}f}%" if is_pct else f"{va:.{prec}f}"
        fb = f"{vb*100:.{prec}f}%" if is_pct else f"{vb:.{prec}f}"
        delta = _delta_cell(va, vb, is_pct, hib, prec)
        kpi_html += (f"<tr><td style='padding:6px'>{label}</td>"
                     f"<td style='text-align:right'>{fa}</td>"
                     f"<td style='text-align:right'>{fb}</td>"
                     f"<td style='text-align:right'>{delta}</td></tr>")

    # IC \u5bf9\u6bd4\u56fe
    ic_a, ic_b = res_a["ic"], res_b["ic"]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("\u7d2f\u79ef IC \u66f2\u7ebf",
                                        "\u6ed1\u52a8 IC (20d)"))
    fig.add_trace(go.Scatter(x=ic_a.index, y=ic_a.cumsum(),
                             name=f"{name_a} cum-IC", line=dict(color="#2196f3")),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=ic_b.index, y=ic_b.cumsum(),
                             name=f"{name_b} cum-IC", line=dict(color="#e53935")),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=ic_a.index, y=ic_a.rolling(20).mean(),
                             name=f"{name_a} rolling-IC", line=dict(color="#2196f3", dash="dash")),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=ic_b.index, y=ic_b.rolling(20).mean(),
                             name=f"{name_b} rolling-IC", line=dict(color="#e53935", dash="dash")),
                  row=2, col=1)
    fig.update_layout(height=600, hovermode="x unified",
                      title=f"A/B \u5bf9\u6bd4: {name_a} vs {name_b}")
    fig_html = fig.to_html(include_plotlyjs="cdn", full_html=False)

    # \u56e0\u5b50\u6570\u5bf9\u6bd4
    n_a = len(res_a.get("feat_cols", []))
    n_b = len(res_b.get("feat_cols", []))
    n_new = len(res_b.get("new_factor_cols", []))

    # \u5168\u9875
    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8">
<title>v5 A/B \u5bf9\u6bd4 - {time.strftime('%Y-%m-%d %H:%M')}</title>
<style>
body {{ font-family: 'Helvetica Neue', Arial, sans-serif; max-width: 1200px;
       margin: 20px auto; padding: 0 20px; color: #222; }}
h1 {{ color: #1a237e; border-bottom: 3px solid #1a237e; padding-bottom: 8px; }}
h2 {{ color: #283593; margin-top: 28px; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
th {{ background: #e8eaf6; padding: 10px; text-align: left; }}
td {{ border-bottom: 1px solid #eee; padding: 8px; }}
tr:nth-child(even) {{ background: #fafafa; }}
.summary {{ background: #f0f4ff; padding: 16px 20px; border-radius: 8px; margin: 16px 0; }}
.summary strong {{ color: #1a237e; }}
</style>
</head><body>
<h1>v5 A/B \u5bf9\u6bd4 - 8 \u5927\u95ee\u9898\u96c6\u6210\u7248</h1>
<div class="summary">
<p>\u53d1\u751f\u65f6\u95f4: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
<p>\u6bd4\u8f83\u5bf9\u8c61: <strong>{name_a}</strong> ({n_a} \u56e0\u5b50) vs
<strong>{name_b}</strong> ({n_b} \u56e0\u5b50\uff0c\u5176\u4e2d\u65b0 {n_new})</p>
<p>\u65b0\u56e0\u5b50\u5bb6\u65cf:
microstructure (\u8dcc\u505c/\u6c89\u5bc2/\u6307\u6570\u63a9\u62a4) + seat_network (\u5e2d\u4f4d\u7f51\u7edc) +
intraday proxy (\u65e5\u5185\u72b9\u8c6b\u5ea6)</p>
</div>

<h2>KPI \u5bf9\u6bd4</h2>
<table>{kpi_html}</table>

<h2>IC \u66f2\u7ebf</h2>
{fig_html}

<h2>\u539f\u59cb\u6570\u636e</h2>
<details><summary>\u70b9\u51fb\u5c55\u5f00\u5177\u4f53\u7edf\u8ba1</summary>
<h3>{name_a}</h3>
<ul>
<li>IC mean: {ic_a.mean():.4f}</li>
<li>IC std: {ic_a.std():.4f}</li>
<li>IC&gt;0 \u5360\u6bd4: {(ic_a > 0).mean():.2%}</li>
<li>Sharpe: {sa['sharpe']:.2f}</li>
</ul>
<h3>{name_b}</h3>
<ul>
<li>IC mean: {ic_b.mean():.4f}</li>
<li>IC std: {ic_b.std():.4f}</li>
<li>IC&gt;0 \u5360\u6bd4: {(ic_b > 0).mean():.2%}</li>
<li>Sharpe: {sb['sharpe']:.2f}</li>
</ul>
</details>

</body></html>"""
    # 清理可能的孤立 surrogate 再写
    html_clean = html.encode("utf-8", errors="replace").decode("utf-8")
    out_path.write_text(html_clean, encoding="utf-8")


def main(pool: int = 200, start: str = "20230101", end: str = "20260420",
         top: float = 0.1, reb: int = 20, with_ann: bool = False):
    t0 = time.time()
    print("\n" + "="*70)
    print(" A/B: v3 baseline vs v5+8\u5927\u95ee\u9898 (\u8fd0\u884c 2 \u6b21)")
    print("="*70)

    print(f"\n\u2194\ufe0f  \u5b9e\u9a8c A: v3 baseline (\u65e0\u65b0\u56e0\u5b50)...")
    res_a = run_v5(n_pool=pool, start=start, end=end,
                   top_ratio=top, rebalance_days=reb,
                   with_microstructure=False, with_seat=False,
                   with_intraday=False, with_ann=False)

    print(f"\n\u2194\ufe0f  \u5b9e\u9a8c B: v5 (\u542f\u7528\u6240\u6709\u65b0\u56e0\u5b50, with_ann={with_ann})...")
    res_b = run_v5(n_pool=pool, start=start, end=end,
                   top_ratio=top, rebalance_days=reb,
                   with_microstructure=True, with_seat=True,
                   with_intraday=True, with_ann=with_ann)

    if res_a is None or res_b is None:
        print("\u274c \u6709\u4e00\u7aef\u8fd0\u884c\u5931\u8d25")
        return

    ts = time.strftime("%Y%m%d_%H%M")
    OUT.mkdir(exist_ok=True)
    out_path = OUT / f"compare_v5_{ts}.html"
    name_a = "v3 baseline"
    name_b = "v5 + 8\u5927\u95ee\u9898"
    _build_html(res_a, res_b, name_a, name_b, out_path)
    print(f"\n\u2705 HTML \u62a5\u544a: {out_path}")
    print(f"\u603b\u8017\u65f6: {time.time()-t0:.0f}s")

    # \u6458\u8981
    sa, sb = res_a["stats"], res_b["stats"]
    print("\n" + "="*70)
    print(" A/B \u6458\u8981")
    print("="*70)
    hdr_delta = "\u5dee\u5f02"
    print(f"  {'':25s}{'A(baseline)':>15s}{'B(v5)':>15s}{hdr_delta:>15s}")
    for k in ["sharpe", "annual_return", "max_drawdown",
              "excess_return", "info_ratio", "avg_turnover"]:
        a, b = sa.get(k, 0), sb.get(k, 0)
        d = b - a
        if k in ("annual_return", "max_drawdown", "excess_return", "avg_turnover"):
            print(f"  {k:25s}{a*100:>14.2f}% {b*100:>14.2f}% {d*100:>+14.2f}%")
        else:
            print(f"  {k:25s}{a:>15.3f}{b:>15.3f}{d:>+15.3f}")
    print(f"  {'IC mean':25s}{res_a['ic'].mean():>15.4f}{res_b['ic'].mean():>15.4f}"
          f"{res_b['ic'].mean()-res_a['ic'].mean():>+15.4f}")
    lbl_factor = "\u56e0\u5b50\u6570"
    print(f"  {lbl_factor:25s}{len(res_a['feat_cols']):>15d}{len(res_b['feat_cols']):>15d}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=200)
    ap.add_argument("--start", default="20230101")
    ap.add_argument("--end", default="20260420")
    ap.add_argument("--top", type=float, default=0.1)
    ap.add_argument("--reb", type=int, default=20)
    ap.add_argument("--with_ann", action="store_true")
    a = ap.parse_args()
    main(pool=a.pool, start=a.start, end=a.end,
         top=a.top, reb=a.reb, with_ann=a.with_ann)
