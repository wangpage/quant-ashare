"""页面 5: 我的持仓 + 实时推荐.

功能:
    1. 左侧: Top 10 推荐 (真实东财实时价, 含止损/止盈/动作建议)
    2. 右侧: 输入持仓 → 系统诊断 → 何时卖
    3. 底部: 明确告知"涨跌概率"这东西的局限性
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import streamlit as st

from webapp import data_providers as dp
from webapp import positions_engine as pe
from webapp.realtime_quote import realtime_quote
from webapp.components import charts


st.set_page_config(page_title="我的持仓", page_icon="💼", layout="wide")

st.title("💼 我的持仓 + Top 10 实时推荐")


# ==================== 关键免责声明 (顶部) ====================
with st.container(border=True):
    st.error(
        "⚠️ **关于涨跌概率**: 任何系统说 '涨 73%, 跌 27%' 都是骗你. "
        "真实量化 IC 0.04-0.06 = 胜率 55-58%, 接近抛硬币. "
        "**本页给你的是**: 真实价格 + 规则化的止损止盈位 + 技术指标一致性评分 (不是预测!)."
    )


# ==================== 数据抓取 ====================
@st.cache_data(ttl=60)
def fetch_all_data(codes: tuple[str, ...]):
    """缓存 60 秒, 避免每次交互都拉数据."""
    kline = {c: dp.get_stock_kline(c, days=120) for c in codes}
    quotes = {}
    for c in codes:
        q = realtime_quote(c)
        if q and q.get("price"):
            quotes[c] = q
        time.sleep(0.05)
    return kline, quotes


# 候选池 (基于信号 top 15)
signals_df = dp.get_today_signals(top_k=15)
candidates = [
    {
        "code": row["代码"], "name": row["名称"],
        "model_score": row["模型分"],
        "theme": row["主题"],
    }
    for _, row in signals_df.iterrows()
]

with st.spinner("抓取实时行情 (东财)..."):
    codes_tuple = tuple(c["code"] for c in candidates)
    kline_data, quotes = fetch_all_data(codes_tuple)


# ==================== Top 10 推荐 ====================
st.subheader("📊 Top 10 实时推荐 (模型分 × 主题 × 技术面)")

if not quotes:
    st.warning("⚠️ 实时行情抓取失败 (可能被限频或网络), 使用 K线最后价作为参考价.")

recommendations = pe.build_recommendations(
    candidates, kline_data, quotes, top_n=10,
)

if not recommendations:
    st.error("无推荐数据, 请检查数据源.")
else:
    rec_rows = []
    bias_emoji = {
        "strong_bull": "🟢🟢", "lean_bull": "🟢",
        "neutral":     "⚪",
        "lean_bear":   "🔴", "strong_bear": "🔴🔴",
    }
    for r in recommendations:
        rec_rows.append({
            "#": r.rank,
            "代码": r.code,
            "名称": r.name,
            "现价": r.current_price,
            "建议买入价": r.suggested_entry,
            "止损位": r.stop_loss,
            "止盈位": r.take_profit,
            "止损%": round((r.stop_loss / r.current_price - 1) * 100, 1),
            "止盈%": round((r.take_profit / r.current_price - 1) * 100, 1),
            "技术面": f"{bias_emoji.get(r.directional_bias, '⚪')} {r.directional_bias}",
            "一致性": f"{r.bias_confidence:.0%}",
            "主题": r.theme[:18],
            "最大持有": f"{r.max_hold_days}天",
        })
    rec_df = pd.DataFrame(rec_rows)

    st.dataframe(
        rec_df, hide_index=True, use_container_width=True,
        column_config={
            "现价":    st.column_config.NumberColumn(format="%.2f"),
            "建议买入价": st.column_config.NumberColumn(format="%.2f"),
            "止损位":  st.column_config.NumberColumn(format="%.2f"),
            "止盈位":  st.column_config.NumberColumn(format="%.2f"),
            "止损%":  st.column_config.NumberColumn(format="%.1f%%"),
            "止盈%":  st.column_config.NumberColumn(format="%.1f%%"),
        },
        height=420,
    )

    with st.expander("📖 每只股票的完整推荐逻辑", expanded=False):
        for r in recommendations:
            st.markdown(
                f"**{r.rank}. {r.code} {r.name}** - 现价 {r.current_price:.2f}, "
                f"建议 {r.suggested_entry:.2f} 买入 / {r.stop_loss:.2f} 止损 / "
                f"{r.take_profit:.2f} 止盈"
            )
            for reason in r.reasoning:
                st.caption(f"  • {reason}")
            st.divider()


# ==================== 持仓诊断 ====================
st.subheader("💼 输入你的持仓, 系统诊断")

st.caption(
    "添加你已持有的股票. 系统会给出: 当前盈亏 + 建议动作 (继续持有/减仓/止损/止盈) + 触发规则."
)

# 初始化 session_state
if "positions" not in st.session_state:
    st.session_state["positions"] = []

# 输入表单
with st.form("add_position", clear_on_submit=True):
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        in_code = st.text_input("股票代码", placeholder="例如 300750")
    with c2:
        in_cost = st.number_input("成本价", min_value=0.0, value=0.0, step=0.01, format="%.2f")
    with c3:
        in_shares = st.number_input("持仓数量 (股)", min_value=0, value=0, step=100)
    with c4:
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("➕ 添加持仓", use_container_width=True)

    if submitted and in_code and in_cost > 0 and in_shares > 0:
        st.session_state["positions"].append({
            "code": in_code.zfill(6),
            "cost": in_cost,
            "shares": int(in_shares),
        })
        st.success(f"已添加 {in_code}")

# 清空持仓
if st.session_state["positions"]:
    if st.button("🗑️ 清空持仓"):
        st.session_state["positions"] = []
        st.rerun()

# 诊断
if st.session_state["positions"]:
    st.markdown("### 📋 持仓诊断结果")

    # 批量拉 K 线 + 实时价
    codes = tuple(p["code"] for p in st.session_state["positions"])
    pos_kline, pos_quotes = fetch_all_data(codes)

    action_emoji = {
        "strong_hold": "🟢 强势持有",
        "hold":        "🟡 继续持有",
        "reduce":      "🟠 建议减仓",
        "sell":        "🔴 止盈卖出",
        "stop_loss":   "⛔ 止损卖出",
        "add":         "🟢 可补仓",
    }

    diag_rows = []
    for p in st.session_state["positions"]:
        code = p["code"]
        name = dp.get_stock_info(code).get("name", code)
        df = pos_kline.get(code, pd.DataFrame())

        # 把实时价写回 df 最后一行 (用于诊断用最新价)
        if code in pos_quotes and not df.empty:
            df = df.copy()
            df.loc[df.index[-1], "close"] = pos_quotes[code]["price"]

        diag = pe.evaluate_position(
            code, name, p["cost"], p["shares"], df,
        )
        diag_rows.append({
            "代码": code,
            "名称": name,
            "成本": p["cost"],
            "现价": diag.current_price,
            "盈亏%": round(diag.pnl_pct * 100, 2),
            "盈亏元": round(diag.pnl_abs, 0),
            "止损位": diag.stop_loss_price,
            "止盈位": diag.stop_profit_price,
            "建议动作": action_emoji.get(diag.action, diag.action),
            "技术信心": f"{diag.bias_confidence:.0%}",
            "触发规则": "; ".join(diag.triggered_rules) if diag.triggered_rules else "-",
        })

    pos_df = pd.DataFrame(diag_rows)

    st.dataframe(
        pos_df, hide_index=True, use_container_width=True,
        column_config={
            "成本":    st.column_config.NumberColumn(format="%.2f"),
            "现价":    st.column_config.NumberColumn(format="%.2f"),
            "盈亏%":  st.column_config.NumberColumn(format="%.2f%%"),
            "盈亏元":  st.column_config.NumberColumn(format="%.0f"),
            "止损位":  st.column_config.NumberColumn(format="%.2f"),
            "止盈位":  st.column_config.NumberColumn(format="%.2f"),
        },
    )

    # 汇总
    total_pnl = sum(r["盈亏元"] for r in diag_rows)
    total_cost = sum(p["cost"] * p["shares"] for p in st.session_state["positions"])
    total_mkt = sum(d["现价"] * p["shares"]
                    for d, p in zip(diag_rows, st.session_state["positions"]))

    c1, c2, c3 = st.columns(3)
    c1.metric("组合成本", f"¥{total_cost:,.0f}")
    c2.metric("组合市值", f"¥{total_mkt:,.0f}")
    c3.metric(
        "组合盈亏",
        f"¥{total_pnl:+,.0f}",
        f"{total_pnl / max(total_cost, 1) * 100:+.2f}%",
    )

    # 详细 per-stock 分析
    with st.expander("🔬 逐只股票完整诊断", expanded=False):
        for p, row in zip(st.session_state["positions"], diag_rows):
            code = p["code"]
            name = row["名称"]
            df = pos_kline.get(code, pd.DataFrame())
            if df.empty:
                continue
            if code in pos_quotes:
                df = df.copy()
                df.loc[df.index[-1], "close"] = pos_quotes[code]["price"]
            diag = pe.evaluate_position(code, name, p["cost"], p["shares"], df)

            st.markdown(f"#### {code} {name}")

            col_l, col_r = st.columns([1.5, 1])
            with col_l:
                st.plotly_chart(
                    charts.kline_chart(df.tail(60), height=350),
                    use_container_width=True,
                )
            with col_r:
                st.markdown(f"**建议动作:** {action_emoji.get(diag.action, diag.action)}")
                for reason in diag.action_reason:
                    st.markdown(f"- {reason}")
                st.divider()
                st.markdown("**技术指标:**")
                if diag.tech:
                    st.markdown(f"- RSI14: {diag.tech.rsi14:.1f}")
                    st.markdown(f"- ATR14: {diag.tech.atr14:.2f}")
                    st.markdown(f"- 布林位置: {diag.tech.boll_pos:+.2f}")
                    st.markdown(f"- MACD: {diag.tech.macd_signal}")
                    st.markdown("**信号详情:**")
                    for d in diag.tech.signal_detail:
                        st.caption(f"  {d}")
            st.divider()

else:
    st.info("👆 在上面的表单添加你的持仓, 开始诊断.")


# ==================== 教育区: 关于 "涨跌概率" ====================
st.divider()

with st.expander("🎓 为什么本页不告诉你 '涨 XX%' 概率?"):
    st.markdown("""
    ### 残酷的真相

    任何声称能给出精确涨跌概率的系统都在**欺骗你**. 原因:

    **1. 数学上的限制**
    - 顶级量化模型 IC ≈ 0.05, 意味着预测和真实涨跌的相关性只有 5%
    - 这对应胜率约 55-58%, 离 "涨 73%" 的精度差几个数量级
    - 西蒙斯的大奖章基金长期胜率 55%, 已经是神级水平

    **2. 市场是反身性的**
    - 任何模型一旦公开, alpha 就衰减 (Goodhart's Law)
    - 黑天鹅 (战争/监管/爆雷) 从不会在概率里

    **3. 为什么行业标准是 "directional bias" 而不是 "probability"**
    - 多头信号 4/6 = "lean_bull" 表示**多数指标倾向看多**
    - 这不是 "涨 66% 概率", 只是 "**多个独立视角的一致性**"

    ### 专业量化给的是什么?

    | ❌ 不给 | ✅ 给 |
    |---|---|
    | "涨 73% 概率" | 技术指标一致性 (4/6 看多) |
    | "保证赚钱" | 基于 ATR 的止损位, 控制最大亏损 |
    | "xxx 股票必涨" | 多维评分 (模型 + 主题 + 技术面), 候选池 |
    | "抓到起涨点" | 明确的入场/止损/止盈位 |

    ### 你的真正优势

    系统做的事:
    - 过滤掉涨停/停牌/ST 等不可交易股
    - 避免买在拥挤度顶点
    - 强制止损 (阻止你"再等等")
    - 多维度交叉验证, 减少单一信号失真

    你要做的事:
    - 决定风险承受能力 (单票不超 15%)
    - 严守止损 (触发不犹豫)
    - 记录每笔交易的教训
    - 控制情绪 (这才是最难的)
    """)
