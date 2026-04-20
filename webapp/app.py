"""A股量化系统 - 主入口 (Streamlit).

运行:
    cd quant_ashare
    streamlit run webapp/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# 让 webapp.* 能导入 (用户从项目根启动时亦可)
_PARENT = Path(__file__).resolve().parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

import streamlit as st

from webapp import data_providers as dp


st.set_page_config(
    page_title="A股量化系统",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==================== 侧边栏 ====================
def render_sidebar():
    with st.sidebar:
        st.title("📊 Quant A股")
        st.caption("qlib + akshare + LLM 多智能体")

        st.divider()

        mode = "Mock 演示" if dp.is_mock() else "Real 实盘数据"
        color = "🟡" if dp.is_mock() else "🟢"
        st.markdown(f"**数据模式:** {color} {mode}")

        if dp.is_mock():
            st.caption(
                "当前使用占位数据. 要接真实数据, 设置环境变量:\n"
                "`export QUANT_WEB_MODE=real`"
            )

        st.divider()

        st.markdown("### 📖 页面导航")
        st.markdown(
            "- 📡 **今日信号仪表盘** - 大盘状态 + 主题 + 信号池\n"
            "- 🔍 **个股详情** - K 线 / 因子 / 主题归属\n"
            "- 🧠 **Agent 辩论记录** - 多智能体决策链\n"
            "- 📈 **回测与策略** - 净值曲线 + IC 分析"
        )

        st.divider()

        st.markdown("### 🔗 项目链接")
        st.markdown(
            "- [GitHub](https://github.com/wangpage/quant-ashare)\n"
            "- [15 圈内 tricks](../ADVANCED_TRICKS.md)\n"
            "- [Level2 接入](../LEVEL2_LIVE_TEST_PLAN.md)"
        )


# ==================== 主页 ====================
def render_home():
    st.title("📈 A 股量化交易系统")
    st.caption("基于 qlib + akshare + LLM 多智能体的机构级量化研究+决策平台")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("覆盖股票", "100+")
    with col2:
        st.metric("因子数量", "190+")
    with col3:
        st.metric("测试用例", "219")
    with col4:
        st.metric("核心模块", "16")

    st.divider()

    st.markdown("## 🎯 快速入口")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.info("📡 **今日信号仪表盘**\n\n看大盘状态、top 主题、当日推荐股票池")
    with c2:
        st.info("🔍 **个股详情**\n\n输入代码看 K 线 / 因子 / 主题 / 回测")
    with c3:
        st.info("🧠 **Agent 辩论记录**\n\n三分析师 + Bull vs Bear 多轮辩论 + 交易员 + 风控")
    with c4:
        st.info("📈 **回测与策略**\n\n策略净值 vs 基准 / IC 分析 / 交易指标")

    st.markdown("👈 使用左侧导航进入各功能模块")

    st.divider()

    st.markdown("## 🏗️ 架构速览")

    st.code("""
┌─────────────────────────────────────────────────────────┐
│  LLM 决策层  Hermes XML + Bull vs Bear 多轮辩论            │
│  [基本面] [技术] [情绪] → [Bull vs Bear] → [交易] → [风控]  │
├─────────────────────────────────────────────────────────┤
│  qlib 核心 + Alpha158 + 35 A股特化因子                    │
├─────────────────────────────────────────────────────────┤
│  【暗门模块】                                             │
│  标签工程 / 微结构 / 因子衰减 / 事件屏蔽 /                  │
│  Barra 中性化 / 组合优化 / 数据清洗 / 执行层               │
├─────────────────────────────────────────────────────────┤
│  数据: akshare + 东财/新浪直连 + Level2 NATS              │
├─────────────────────────────────────────────────────────┤
│  风控: 涨跌停 / T+1 / 凯利 / 回撤熔断 / Regime 乘数         │
└─────────────────────────────────────────────────────────┘
    """, language="text")

    st.divider()

    with st.expander("⚠️ 免责声明 (请务必阅读)"):
        st.warning("""
        - 本项目仅供**学习与研究**使用, 不构成投资建议
        - 量化投资存在**实质性风险**
        - 任何 "95% 胜率" / "夏普 >5" 的宣传都是**过拟合或话术**
        - 实盘前必须: 6 个月样本外回测 + 3 个月模拟盘 + 从小资金起步

        **真正让你赚钱的不是代码, 是纪律 / 耐心 / 仓位管理 / 持续学习**
        """)


# ==================== 入口 ====================
def main():
    render_sidebar()
    render_home()


if __name__ == "__main__":
    main()
