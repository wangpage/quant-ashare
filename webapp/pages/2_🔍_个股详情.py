"""页面 2: 个股详情."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from webapp import data_providers as dp
from webapp.components import charts


st.set_page_config(page_title="个股详情", page_icon="🔍", layout="wide")

st.title("🔍 个股详情")

# ==================== 股票选择 ====================
STOCK_POOL = [
    "300750", "600519", "002156", "600584", "688256",
    "600760", "002155", "600549", "601100", "603728", "603011",
]

col1, col2 = st.columns([1, 2])
with col1:
    code = st.selectbox(
        "选择股票代码", options=STOCK_POOL, index=0,
        help="演示模式只支持预置股票池",
    )
    custom = st.text_input("或输入 6 位代码", value="")
    if custom and len(custom) == 6 and custom.isdigit():
        code = custom

with col2:
    info = dp.get_stock_info(code)
    st.markdown(f"### {info['name']} ({info['code']})")
    m1, m2, m3 = st.columns(3)
    m1.metric("行业", info.get("industry", "-"))
    m2.metric("市值", info.get("market_cap", "-"))
    themes = dp.get_stock_themes(code)
    m3.metric("所属主题", ", ".join(themes) if themes else "-")

st.divider()


# ==================== K 线图 ====================
st.subheader("📈 K 线 (近 120 日)")
days = st.select_slider("时间范围", options=[30, 60, 120, 250], value=120)
kline = dp.get_stock_kline(code, days=days)

st.plotly_chart(
    charts.kline_chart(kline, title=f"{info['name']} ({code})"),
    use_container_width=True,
)

# 简单统计
lastest_close = kline["close"].iloc[-1]
pct_30d = (lastest_close / kline["close"].iloc[-min(30, len(kline))] - 1) * 100
pct_5d = (lastest_close / kline["close"].iloc[-min(5, len(kline))] - 1) * 100
vol_20d = kline["close"].pct_change().tail(20).std() * (252 ** 0.5) * 100

c1, c2, c3, c4 = st.columns(4)
c1.metric("最新价", f"{lastest_close:.2f}")
c2.metric("5日涨跌", f"{pct_5d:+.2f}%")
c3.metric("30日涨跌", f"{pct_30d:+.2f}%")
c4.metric("年化波动率", f"{vol_20d:.1f}%")

st.divider()


# ==================== 因子得分 ====================
st.subheader("🧬 因子画像")

factors = dp.get_stock_factors(code)

fcol1, fcol2 = st.columns([1.5, 1])
with fcol1:
    st.dataframe(
        factors, hide_index=True, use_container_width=True,
        column_config={
            "数值": st.column_config.NumberColumn(format="%.3f"),
        },
        height=420,
    )
with fcol2:
    st.markdown("#### 因子说明")
    st.caption("""
    - **REV5 / 反转**: 短期超跌反弹信号
    - **TURN_RATIO20**: 换手激增 = 资金关注上升
    - **VOL20**: 历史波动率, 越高风险越大
    - **BOLL_POS**: 布林带位置, >0.8 超买, <-0.8 超卖
    - **MA_DIFF20**: 距 20 日均线, >0 强势
    - **GAP1**: 昨日收盘到今开的跳空
    - **LIMIT_UP20**: 近 20 日涨停次数
    - **TREND10**: 趋势一致性, 连续上涨天数
    """)

st.divider()


# ==================== 主题归属 + 历史交易 ====================
st.subheader("🎯 主题标签与关联事件")

if themes:
    for t in themes:
        st.markdown(f"- **{t}** (综合热度待接入实时评分)")
else:
    st.info("该股票不属于当前 8 大主线.")

with st.expander("🧠 历史交易记忆 (从 Memory Curator 查询)"):
    st.info("接入 memory.MemoryStore 后, 此处显示该股票的交易反思记录")
    st.code("""
    示例:
    2026-03-15 [bull_trending, 持有 4 日, +6.2%]
    反思: [连板 3 天后次日低开 3%+] -> [不追入, 等放量确认]
    质量分: 8/10
    """, language="text")

with st.expander("🌐 相关巨头事件 (海外联动)"):
    st.info("接入 expert_signals.supply_chain_intel 后, 显示该股票关联的 NVIDIA / TSMC / Tesla 等事件")
