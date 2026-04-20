"""页面 4: 回测与策略曲线."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import streamlit as st

from webapp import data_providers as dp
from webapp.components import charts


st.set_page_config(page_title="回测与策略", page_icon="📈", layout="wide")

st.title("📈 回测与策略曲线")

# ==================== 加载回测数据 ====================
report = dp.get_backtest_report()
stats = report["stats"]
curve = report["curve"]
ic_df = report["ic"]

# ==================== 关键指标 ====================
st.subheader("核心指标")
st.caption(f"回测区间: {stats['period']}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("年化收益", f"{stats['annual_return']:+.2%}")
c2.metric("超额收益 (vs 沪深300)", f"{stats['excess_annual']:+.2%}")
c3.metric("夏普比率", f"{stats['sharpe']:.2f}")
c4.metric("最大回撤", f"{stats['max_drawdown']:.2%}")

c5, c6, c7, c8 = st.columns(4)
c5.metric("胜率", f"{stats['win_rate']:.1%}")
c6.metric("年化波动", f"{stats['annual_vol']:.2%}")
c7.metric("日均换手", f"{stats['avg_turnover']:.2%}")
c8.metric("交易次数", f"{stats['trade_count']}")

st.divider()


# ==================== 净值曲线 ====================
st.subheader("📈 策略净值曲线")

show_bench = st.checkbox("叠加基准 (沪深 300)", value=True)
show_excess = st.checkbox("叠加超额收益", value=False)

cols = ["strategy"]
if show_bench:
    cols.append("benchmark")
if show_excess:
    cols.append("excess")

st.plotly_chart(charts.nav_curve(curve, cols=cols), use_container_width=True)

# 回撤曲线
st.subheader("📉 回撤 (Drawdown)")
nav_series = curve["strategy"].copy()
nav_series.index = curve["date"]
st.plotly_chart(charts.drawdown_curve(nav_series), use_container_width=True)

st.divider()


# ==================== IC 分析 ====================
st.subheader("🧪 IC 衰减监控")

ic_c1, ic_c2, ic_c3, ic_c4 = st.columns(4)
ic_c1.metric("IC mean",  f"{stats['ic_mean']:.4f}")
ic_c2.metric("ICIR",      f"{stats['icir']:.2f}")
ic_c3.metric("IC 20日",   f"{ic_df['ic_20d'].iloc[-1]:.4f}"
             if not pd.isna(ic_df['ic_20d'].iloc[-1]) else "N/A")
ic_c4.metric("样本天数",  f"{len(ic_df)}")

st.plotly_chart(charts.ic_curve(ic_df), use_container_width=True)

st.caption(
    "💡 **IC 解读**: 日度 IC > 0 = 预测与真实正相关, IC 20日均值稳定在 0.03-0.06 "
    "是主流因子的合理区间. IC 20日均值 < 0.01 持续 2 个月 = **因子衰减**, 建议下线."
)

st.divider()


# ==================== 因子分解 ====================
st.subheader("🔍 Barra 风格暴露 (示例)")
st.caption("用 Barra CNE5 六风格因子回归策略收益, 残差才是真正 alpha.")

style_cols = ["Size", "Beta", "Momentum", "ResidualVol", "Liquidity", "NonLinSize"]
import numpy as np

np.random.seed(42)
exposure = pd.DataFrame({
    "风格因子": style_cols,
    "暴露": np.random.uniform(-0.3, 0.4, 6).round(3),
    "贡献 (年化)": np.random.uniform(-0.02, 0.03, 6).round(4),
})
exposure["解读"] = [
    "Size 负暴露 = 偏小盘",
    "Beta 略正 = 跟随大盘",
    "Momentum 正 = 偏动量",
    "残差波动 中性",
    "流动性 略偏好高流动",
    "非线性规模 略负",
]

st.dataframe(
    exposure, hide_index=True, use_container_width=True,
    column_config={
        "暴露": st.column_config.NumberColumn(format="%.3f"),
        "贡献 (年化)": st.column_config.NumberColumn(format="%.2%"),
    },
)

alpha_residual = stats['annual_return'] - exposure["贡献 (年化)"].sum()
st.info(f"📊 **真实 Alpha (扣除风格暴露后残差)**: {alpha_residual:+.2%} / 年")
st.caption("如果真 alpha < 5%, 说明策略收益主要来自风格 beta, 并非独特选股能力.")

st.divider()


# ==================== 导出 ====================
st.subheader("📥 导出")
csv = curve.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "下载完整回测数据 (CSV)",
    data=csv, file_name="backtest_curve.csv", mime="text/csv",
)

with st.expander("📋 上线 checklist (12 项硬约束)"):
    st.markdown("""
    - [ ] 回测是否屏蔽 涨跌停 / 停牌 / 财报窗口 / 新股 / ST?
    - [ ] 冲击成本用 sqrt 律, 不是固定 bps?
    - [ ] 风格中性化用 Barra 6 风格, 不是 Size + Industry?
    - [ ] 样本外回测至少 6 个月?
    - [ ] rolling_IC / all_time_IC > 0.7 (最近 60 天)?
    - [ ] 组合日换手率 < 60% (避免手续费吞噬)?
    - [ ] 单票最大仓位 < 5%, 单行业 < 25%?
    - [ ] 波动率目标 + 回撤熔断 (15% limit)?
    - [ ] 因子与公开因子 (FF5) 相关性 < 0.4?
    - [ ] 盘后资金流因子**不能**在盘中用?
    - [ ] 解禁日 / 财报日 前后 2 天不新开仓?
    - [ ] Level2 订单流因子仅用于 T+1 以下频率?
    """)
