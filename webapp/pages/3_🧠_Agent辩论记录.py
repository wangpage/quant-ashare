"""页面 3: Agent 辩论记录."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from webapp import data_providers as dp


st.set_page_config(page_title="Agent 辩论记录", page_icon="🧠", layout="wide")

st.title("🧠 多智能体辩论记录")
st.caption("Hermes XML 结构化 + Bull vs Bear 多轮辩论 + 交易员 + 风控")

# ==================== 决策列表 ====================
decisions = dp.get_recent_debates()

st.markdown("### 最近决策")
dec_options = [f"{d['datetime']}  {d['code']} {d['name']}  →  {d['final_action'].upper()} (信心 {d['final_conviction']:.0%})"
               for d in decisions]
sel_idx = st.selectbox("选择决策查看完整辩论", options=range(len(dec_options)),
                        format_func=lambda i: dec_options[i])
selected = decisions[sel_idx]

st.divider()


# ==================== 决策概览 ====================
detail = dp.get_debate_detail(selected["id"])

st.subheader(f"📊 决策概览: {detail['name']} ({detail['code']})")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("最终动作", selected["final_action"].upper())
c2.metric("综合评分", f"{selected['final_score']:+.2f}")
c3.metric("信心", f"{selected['final_conviction']:.0%}")
c4.metric("风控裁决", selected["risk_decision"])
c5.metric("辩论轮数", len(detail.get("debate_rounds", [])))

st.divider()


# ==================== 三分析师并行输出 ====================
st.subheader("1️⃣ 三分析师并行分析")

a1, a2, a3 = st.columns(3)
analysts = detail["analysts"]

for col, (name, key, icon) in zip(
    [a1, a2, a3],
    [("基本面分析师", "fundamental", "💼"),
     ("技术面分析师", "technical",   "📈"),
     ("情绪面分析师", "sentiment",   "🌊")],
):
    with col:
        data = analysts[key]
        view_emoji = {"bullish": "🟢 看多", "bearish": "🔴 看空",
                      "neutral": "⚪ 中性"}.get(data["view"], data["view"])
        st.markdown(f"#### {icon} {name}")
        st.markdown(f"**{view_emoji}** · 分值 {data['score']:+.2f}")
        st.markdown(f"**💭 THINKING**\n{data['thinking']}")
        st.markdown(f"**🔗 REASONING**\n{data['reasoning']}")
        st.markdown(f"**⚠️ RISK**\n{data['risk']}")

st.divider()


# ==================== 多轮辩论 ====================
st.subheader("2️⃣ Bull vs Bear 多轮辩论")

for round_data in detail.get("debate_rounds", []):
    st.markdown(f"#### Round {round_data['round']}")
    bull_col, bear_col = st.columns(2)
    with bull_col:
        st.success(f"🐂 **多头 (Bull)**\n\n{round_data['bull']}")
    with bear_col:
        st.error(f"🐻 **空头 (Bear)**\n\n{round_data['bear']}")

st.divider()


# ==================== Judge 裁决 ====================
st.subheader("3️⃣ Researcher Judge 最终裁决")

judge = detail["judge"]
j_view = {"bullish": "🟢 看多", "bearish": "🔴 看空",
          "neutral": "⚪ 中性"}.get(judge["solution"], judge["solution"])

jc1, jc2, jc3 = st.columns([1, 1, 2])
with jc1:
    st.metric("结论", j_view)
with jc2:
    st.metric("综合分 / 信心", f"{judge['score']:+.2f} / {judge['conviction']:.0%}")
with jc3:
    st.info(f"**REASONING**: {judge['reasoning']}")
st.success(f"**EXPLANATION**: {judge['explanation']}")

st.divider()


# ==================== 交易员下单 ====================
st.subheader("4️⃣ 交易员执行方案")

trader = detail["trader"]
t1, t2, t3, t4 = st.columns(4)
t1.metric("动作", trader["action"].upper())
t2.metric("仓位", f"{trader['size_pct']:.1%}")
t3.metric("入场 / 止损 / 止盈",
           f"{trader['entry_price']:.2f} / {trader['stop_loss_price']:.2f} / {trader['take_profit_price']:.2f}")
t4.metric("持仓天数", f"{trader['holding_days']} 天")

st.info(f"📝 {trader['reasoning']}")

st.divider()


# ==================== 风控审核 ====================
st.subheader("5️⃣ 风控经理审核")

risk = detail["risk_review"]
r_color = {"approve": "success", "modify": "warning",
           "reject": "error"}.get(risk["action"], "info")

if r_color == "success":
    st.success(f"✅ **{risk['action'].upper()}** — {risk['reasoning']}")
elif r_color == "warning":
    st.warning(f"⚠️ **{risk['action'].upper()}** — {risk['reasoning']}")
else:
    st.error(f"⛔ **{risk['action'].upper()}** — {risk['reasoning']}")

with st.expander("🔬 看看完整的原始 XML 输出 (Hermes 格式)"):
    st.code("""
<THINKING>宁德时代估值已充分反应国内新能源放缓...</THINKING>

<SCRATCHPAD>
关键指标:
- PE / PB / PEG =  22 / 4.5 / 1.1
- ROE 趋势 = 18% → 19%
- 自由现金流 = 400 亿+
</SCRATCHPAD>

<NON_CONSENSUS>
海外收入占比从 25% 向 40% 切换, 储能 2027 超车动力电池
</NON_CONSENSUS>

<SECOND_ORDER>
AI 吃电 → 储能需求 → 大储/工商业储能
</SECOND_ORDER>

<CROWDING_CHECK>
公募持仓 8% (已减仓), 卖方 40 家, 一致推荐 62%.
结论: neutral (可进可退)
</CROWDING_CHECK>

<CONTRARIAN>
若大基金减仓 + 卖方下调 3 家 + 跌破 430, 则 oversold reverse
</CONTRARIAN>

<REASONING>
1) 海外订单已锁定 2026-2027
2) 450 估值对应 2026 业绩 22 倍 PE
3) 北向 5 日净买 32 亿
</REASONING>

<SCORE>0.42</SCORE>
<CONVICTION>0.72</CONVICTION>
<SOLUTION>bullish, action=buy, size_pct=0.12</SOLUTION>
<EXPLANATION>反补贴落地前仓位适度, 落地后看条款再决定</EXPLANATION>
""", language="xml")
