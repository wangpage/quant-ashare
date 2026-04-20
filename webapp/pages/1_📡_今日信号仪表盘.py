"""页面 1: 今日信号仪表盘."""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from webapp import data_providers as dp
from webapp.components import charts


st.set_page_config(page_title="今日信号仪表盘", page_icon="📡", layout="wide")


# ==================== 顶部: 日期 + 大盘 Regime ====================
st.title("📡 今日信号仪表盘")
st.caption(f"更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

regime = dp.get_market_regime()

regime_color = {
    "bull_trending":  "🟢 上升趋势",
    "bull_quiet":      "🟢 温和上行",
    "bear_trending":  "🔴 下降趋势",
    "bear_quiet":      "🟠 震荡偏弱",
    "choppy":          "🟡 横盘震荡",
    "crash":           "⛔ 崩盘",
    "euphoria":        "⚠️ 狂热",
    "unknown":         "❓ 未知",
}.get(regime["regime"], regime["regime"])

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("市场状态", regime_color)
col2.metric("仓位乘数", f"{regime['position_mult']:.2f}")
col3.metric("赚钱效应", regime["money_effect"])
col4.metric("涨停家数", regime["limit_up_count"])
col5.metric("流动性", regime["liquidity_level"])

with st.expander("📋 Regime 详细判据", expanded=False):
    for r in regime["reasons"]:
        st.markdown(f"- {r}")
    st.caption(f"判断信心: {regime['confidence']:.0%}  |  "
                f"趋势: {regime['trend_direction']} ({regime['trend_strength']:.2f})  |  "
                f"波动: {regime['vol_level']}")

st.divider()


# ==================== 主题评分 + 资金分配 ====================
st.subheader("🎯 主题评分 (综合政策 + 巨头 + 地缘信号)")

themes = dp.get_theme_scores()
alloc = dp.get_portfolio_allocation()

left, right = st.columns([1.6, 1])
with left:
    st.dataframe(
        themes, hide_index=True, use_container_width=True,
        column_config={
            "总分":   st.column_config.ProgressColumn(
                "综合分", min_value=0, max_value=10, format="%.2f"),
            "政策":   st.column_config.NumberColumn("政策", format="%.1f"),
            "巨头":   st.column_config.NumberColumn("巨头", format="%.1f"),
            "地缘":   st.column_config.NumberColumn("地缘", format="%.1f"),
        },
    )

with right:
    st.plotly_chart(charts.theme_radar(themes), use_container_width=True)

st.markdown("### 💰 资金分配建议 (100 万基础仓位)")
c1, c2 = st.columns([1, 1])
with c1:
    st.dataframe(
        alloc, hide_index=True, use_container_width=True,
        column_config={
            "分配(元)": st.column_config.NumberColumn(format="%d"),
            "占比":     st.column_config.ProgressColumn(
                min_value=0, max_value=1, format="%.1f%%"),
        },
    )
with c2:
    st.plotly_chart(charts.portfolio_donut(alloc), use_container_width=True)

st.divider()


# ==================== 今日信号池 ====================
st.subheader("🎯 今日 Top 信号 (量化模型 + 主题加权)")

top_k = st.slider("展示股票数", min_value=5, max_value=30, value=15, step=5)
signals = dp.get_today_signals(top_k=top_k)

# 按动作着色
def _color_action(val):
    colors = {"buy": "#e8f5e9", "sell": "#ffebee", "hold": "#fff9c4"}
    return f"background-color: {colors.get(val, '')}"

styled = signals.style.map(_color_action, subset=["动作"])

st.dataframe(
    styled, hide_index=True, use_container_width=True,
    column_config={
        "模型分": st.column_config.ProgressColumn(
            min_value=0, max_value=1, format="%.2f"),
        "仓位":   st.column_config.NumberColumn(format="%.1f%%"),
        "参考价": st.column_config.NumberColumn(format="%.2f"),
    },
    height=min(42 * (len(signals) + 1), 680),
)

st.caption(
    "信号说明: **模型分** = LightGBM 预测的 forward return 分位; "
    "**动作** 由 regime + 风控过滤得出; **仓位** 为凯利公式建议."
)

with st.expander("📖 如何解读这些信号"):
    st.markdown("""
    ### 信号质量分层
    - **模型分 > 0.7**: 一线信号, 可优先配置
    - **模型分 0.5-0.7**: 二线信号, 需交叉验证
    - **模型分 < 0.5**: 观察信号, 不建议建仓

    ### 动作字段
    - `buy`:  明确买入, 按仓位建仓
    - `hold`: 可观望或减仓
    - `sell`: 退出信号

    ### 风险提示
    - 涨停/停牌股票已自动屏蔽
    - T+1 规则已考虑
    - 实际执行建议分批 (TWAP/VWAP)
    """)
