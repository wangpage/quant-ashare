"""研究报告浏览页 - 列出 output/ 目录下所有 HTML 报告, 嵌入展示.

功能:
    - 自动扫描 output/*.html, 按修改时间倒序
    - 区分: compare (对比) / baseline / alpha158_barra / research (单次)
    - 支持:
        · 一键运行对比实验 (run_compare.py)
        · 一键运行单次研究 (run_real_research.py)
    - iframe 嵌入 HTML, 保持原交互 (plotly 可缩放)
"""
from __future__ import annotations

import base64
import subprocess
import sys
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="研究报告", page_icon="📊", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"

st.title("📊 研究报告中心")
st.caption(
    "集中浏览所有研究/回测/对比 HTML 报告. 支持 plotly 交互."
)

# ---------- 扫描所有报告 ----------
if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

html_files = sorted(
    OUTPUT_DIR.glob("*.html"),
    key=lambda p: p.stat().st_mtime, reverse=True,
)


def _label_for(p: Path) -> tuple[str, str]:
    """返回 (emoji, 类型标签)."""
    name = p.name.lower()
    if name.startswith("compare"):
        return "🆚", "对比报告"
    if "alpha158" in name:
        return "🔬", "Alpha158 实验"
    if "baseline" in name:
        return "📉", "Baseline 实验"
    if name.startswith("research"):
        return "📝", "研究报告"
    return "📄", "其他"


# ---------- 顶部: 运行新实验 ----------
with st.expander("▶ 运行新实验", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        exp_type = st.selectbox(
            "实验类型",
            options=["对比实验 (baseline vs Alpha158)",
                      "单次研究 (默认)"],
            index=0,
        )
    with col2:
        n_stocks = st.number_input("股票数", min_value=5, max_value=25,
                                     value=15, step=5)
    with col3:
        start_date = st.text_input("开始日期", value="20240101")

    if st.button("🚀 开始运行 (几分钟, 拉真实数据)"):
        script = ("scripts/run_compare.py"
                  if "对比" in exp_type
                  else "scripts/run_real_research.py")
        cmd = [sys.executable, str(PROJECT_ROOT / script),
               "--n", str(int(n_stocks)), "--start", start_date]
        with st.spinner("运行中, 拉日线 + 跑 pipeline..."):
            try:
                result = subprocess.run(
                    cmd, cwd=str(PROJECT_ROOT),
                    capture_output=True, text=True, timeout=600,
                )
                if result.returncode == 0:
                    st.success("✅ 完成! 下方报告列表已刷新")
                    st.code(result.stdout[-1500:], language="text")
                    st.rerun()
                else:
                    st.error(f"运行失败 (returncode={result.returncode})")
                    st.code(result.stderr[-2000:], language="text")
            except subprocess.TimeoutExpired:
                st.error("超时 (> 10 分钟)")

# ---------- 报告列表 + 预览 ----------
if not html_files:
    st.info(
        "📝 还没有报告. 点击上方 ▶ 运行新实验, "
        "或命令行跑: `python scripts/run_compare.py`"
    )
    st.stop()

st.markdown("### 📂 所有报告")

# 侧边栏: 报告选择
with st.sidebar:
    st.markdown("### 📂 报告列表")
    report_labels = []
    for p in html_files:
        emoji, kind = _label_for(p)
        mtime = p.stat().st_mtime
        import datetime as _dt
        time_str = _dt.datetime.fromtimestamp(mtime).strftime("%m-%d %H:%M")
        report_labels.append(f"{emoji} {p.stem} · {time_str}")

    selected_idx = st.radio(
        "选择报告", options=list(range(len(html_files))),
        format_func=lambda i: report_labels[i],
        label_visibility="collapsed",
    )
    st.caption(f"共 {len(html_files)} 份报告")

selected = html_files[selected_idx]
emoji, kind = _label_for(selected)

col_left, col_right = st.columns([3, 1])
with col_left:
    st.markdown(f"#### {emoji} {selected.name}")
    st.caption(f"类型: {kind}  ·  大小: {selected.stat().st_size // 1024} KB")
with col_right:
    with open(selected, "rb") as f:
        data = f.read()
    st.download_button(
        "⬇️ 下载 HTML", data=data,
        file_name=selected.name, mime="text/html",
    )

# iframe 嵌入
html_content = selected.read_text(encoding="utf-8")
# 用 srcdoc 方式嵌入: 更好的同源策略兼容
import streamlit.components.v1 as components
components.html(html_content, height=1400, scrolling=True)

st.caption(
    "💡 图表由 plotly 渲染, 可拖拽、缩放、悬停查看数值. "
    "完整新标签打开: 点右上 ⬇️ 下载后双击"
)
