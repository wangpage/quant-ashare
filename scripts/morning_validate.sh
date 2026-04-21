#!/bin/bash
# 明早 10:25 运行 — 拉 0422 实时快照验证 0421 预测 + 推飞书
# 用法: bash scripts/morning_validate.sh
#      或 crontab: 25 10 22 4 * cd /Users/page/Desktop/股票/quant_ashare && bash scripts/morning_validate.sh

set -e
cd "$(dirname "$0")/.."

LOG="logs/validate_$(date +%Y%m%d_%H%M).log"
mkdir -p logs

echo "[$(date)] 开始验证" | tee -a "$LOG"
python3 scripts/validate_prediction.py --mode spot 2>&1 | tee -a "$LOG"
echo "[$(date)] 完成" | tee -a "$LOG"
