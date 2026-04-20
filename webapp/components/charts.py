"""复用图表组件 - 基于 plotly."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def kline_chart(df: pd.DataFrame, title: str = "", height: int = 500):
    """K 线 + 成交量 叠加图."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.03,
    )

    fig.add_trace(
        go.Candlestick(
            x=df["date"], open=df["open"], high=df["high"],
            low=df["low"], close=df["close"],
            increasing_line_color="#d94a38",    # 红涨
            decreasing_line_color="#33a453",    # 绿跌
            name="K线",
        ),
        row=1, col=1,
    )

    # 均线
    for w, color in [(5, "#ff9800"), (20, "#2196f3"), (60, "#9c27b0")]:
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["close"].rolling(w).mean(),
                mode="lines",
                line=dict(color=color, width=1),
                name=f"MA{w}",
            ),
            row=1, col=1,
        )

    colors = np.where(df["close"] >= df["open"], "#d94a38", "#33a453")
    fig.add_trace(
        go.Bar(
            x=df["date"], y=df["volume"],
            marker_color=colors, name="成交量",
        ),
        row=2, col=1,
    )

    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        height=height,
        showlegend=True,
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="成交量", row=2, col=1)
    return fig


def nav_curve(df: pd.DataFrame, cols: list[str] | None = None,
              height: int = 420):
    """净值曲线 (可叠加多条)."""
    cols = cols or ["strategy", "benchmark"]
    fig = go.Figure()
    palette = {"strategy": "#d94a38", "benchmark": "#888888",
               "excess": "#2196f3"}
    labels = {"strategy": "策略净值", "benchmark": "基准 (沪深300)",
              "excess": "超额"}
    for c in cols:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df[c], mode="lines",
            line=dict(color=palette.get(c, "#666"), width=2),
            name=labels.get(c, c),
        ))
    fig.update_layout(
        height=height,
        yaxis_title="净值",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=30, b=20),
    )
    return fig


def ic_curve(df: pd.DataFrame, height: int = 320):
    """IC 日度 + 20 日滚动."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["date"], y=df["ic_daily"],
        marker_color="#cfd8dc", name="日度 IC",
    ))
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["ic_20d"],
        mode="lines", line=dict(color="#d94a38", width=2),
        name="IC 20日滚动",
    ))
    fig.add_hline(y=0, line=dict(color="#999", dash="dot"))
    fig.update_layout(
        height=height, yaxis_title="IC",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig


def theme_radar(theme_scores: pd.DataFrame, top_n: int = 6):
    """主题评分雷达图."""
    df = theme_scores.head(top_n)
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=df["总分"].tolist() + [df["总分"].iloc[0]],
        theta=df["主题"].tolist() + [df["主题"].iloc[0]],
        fill="toself",
        line=dict(color="#d94a38", width=2),
        fillcolor="rgba(217, 74, 56, 0.18)",
        name="主题热度",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        showlegend=False,
        height=380,
        margin=dict(l=40, r=40, t=20, b=20),
    )
    return fig


def portfolio_donut(alloc: pd.DataFrame, height: int = 320):
    """资金分配饼图 (donut)."""
    fig = go.Figure(data=[go.Pie(
        labels=alloc["主题"],
        values=alloc["分配(元)"],
        hole=0.55,
        textinfo="label+percent",
        marker=dict(colors=["#d94a38", "#2196f3", "#ff9800", "#4caf50",
                            "#9c27b0", "#607d8b"]),
    )])
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False,
    )
    return fig


def drawdown_curve(nav: pd.Series, height: int = 240):
    """回撤曲线."""
    running_max = nav.cummax()
    dd = (nav - running_max) / running_max
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=nav.index if hasattr(nav, "index") else list(range(len(nav))),
        y=dd, mode="lines", fill="tozeroy",
        line=dict(color="#d94a38", width=1.5),
        fillcolor="rgba(217, 74, 56, 0.15)",
        name="回撤",
    ))
    fig.update_layout(
        height=height, yaxis_tickformat=".1%",
        yaxis_title="回撤",
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig
