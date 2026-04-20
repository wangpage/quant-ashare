#!/usr/bin/env bash
# 一键跑 4 个 Level2 测试, 生成汇总报告.
set -e
cd "$(dirname "$0")/.."

REPORT_DIR="output/level2_tests_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$REPORT_DIR"

echo "========================================"
echo "  Level2 测试套件"
echo "  报告目录: $REPORT_DIR"
echo "========================================"

echo ""
echo "[T2] 数据解析正确性 (离线, mock 数据)"
python tests/test_parser.py 2>&1 | tee "$REPORT_DIR/T2_parser.log" || true

echo ""
echo "[T1] 行情接入联通性 (需要真实 NATS)"
python tests/test_connectivity.py 2>&1 | tee "$REPORT_DIR/T1_connectivity.log" || true

echo ""
echo "[T3] 延迟与吞吐压测 (需要真实 NATS, 1 分钟)"
python tests/test_latency.py 2>&1 | tee "$REPORT_DIR/T3_latency.log" || true

echo ""
echo "[T4] 集成到 quant_ashare (需要真实 NATS, 1 分钟)"
python tests/test_integration.py 2>&1 | tee "$REPORT_DIR/T4_integration.log" || true

echo ""
echo "✅ 所有测试完成, 报告见 $REPORT_DIR/"
ls -la "$REPORT_DIR/"
