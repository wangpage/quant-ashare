#!/usr/bin/env bash
# 启动 Streamlit Web UI
set -e
cd "$(dirname "$0")/.."

# 默认 Mock 模式, 要接真实数据改为 real:
#   QUANT_WEB_MODE=real bash scripts/run_webapp.sh
: "${QUANT_WEB_MODE:=mock}"
export QUANT_WEB_MODE

streamlit run webapp/app.py \
    --server.port=8501 \
    --server.headless=false \
    --browser.gatherUsageStats=false \
    --theme.base=light \
    --theme.primaryColor="#d94a38"
