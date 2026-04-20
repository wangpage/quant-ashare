#!/usr/bin/env bash
# 端到端跑通 pipeline (首次运行 ~1-2小时, 主要是拉数据)
set -e
cd "$(dirname "$0")/.."

echo "[1/5] 拉取沪深300日K线..."
python scripts/01_download_data.py

echo "[2/5] 转换为 qlib bin 格式..."
python scripts/02_convert_to_qlib.py

echo "[3/5] 训练 LightGBM 模型 + 回测..."
python scripts/03_train_model.py

echo "[4/5] 生成回测报告..."
python scripts/04_backtest_report.py

echo "[5/5] 生成今日信号..."
python scripts/05_daily_signal.py --top-k 20

echo "✅ pipeline 完成. 结果保存在 output/, 回测记录在 mlruns/"
