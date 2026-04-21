#!/bin/bash
# 一键安装 crontab: A股 paper trade 每日自动化
#
# 安装内容:
#   • 每交易日 15:30  → 数据更新 + paper trade + 飞书通知 + git push
#   • 每周日 20:00    → leak detector 自检
#
# 用法:
#   bash scripts/install_cron.sh           # 交互安装
#   bash scripts/install_cron.sh --uninstall
#   bash scripts/install_cron.sh --dry-run # 只预览不安装

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="$(which python3)"
MARKER="# quant-ashare cron"

usage() {
  cat <<EOF
Usage: $0 [--uninstall | --dry-run]

Actions:
  (default)     安装 crontab (幂等, 会先去掉旧行)
  --uninstall   卸载 crontab
  --dry-run     只预览要写入的内容, 不修改 crontab
EOF
}

gen_cron() {
  cat <<EOF
$MARKER-start
PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin
# 每交易日 15:30 收盘后: 数据增量 + paper trade + 飞书通知 + push
30 15 * * 1-5 cd "$PROJECT_DIR" && $PYTHON_BIN scripts/cron_daily.py >> logs/cron_last.log 2>&1
# 每周日 20:00: leak detector 回归
0 20 * * 0 cd "$PROJECT_DIR" && $PYTHON_BIN scripts/cron_daily.py --leak-check >> logs/cron_last.log 2>&1
$MARKER-end
EOF
}

remove_old() {
  local cur
  cur="$(crontab -l 2>/dev/null || true)"
  # 删掉 marker 之间的行 (幂等)
  echo "$cur" | sed "/$MARKER-start/,/$MARKER-end/d"
}

install() {
  mkdir -p "$PROJECT_DIR/logs"
  local new
  new="$(remove_old)
$(gen_cron)"
  # 去掉开头空行
  new="$(echo "$new" | sed '/./,$!d')"
  echo "$new" | crontab -
  echo "✅ crontab 已安装"
  echo ""
  echo "当前 crontab:"
  crontab -l | grep -A 10 "$MARKER-start" | head -20
  echo ""
  echo "日志位置: $PROJECT_DIR/logs/cron_last.log"
  echo "立即测试: python3 scripts/cron_daily.py --test-notify"
}

uninstall() {
  local new
  new="$(remove_old)"
  echo "$new" | crontab -
  echo "✅ crontab quant-ashare 行已清除"
}

case "${1:-}" in
  --uninstall) uninstall ;;
  --dry-run)
    echo "即将写入 crontab 的内容 (不修改):"
    echo ""
    gen_cron
    ;;
  -h|--help) usage ;;
  "")
    echo "即将安装:"
    echo ""
    gen_cron
    echo ""
    read -p "确认? (y/N) " ans
    [[ "$ans" != "y" && "$ans" != "Y" ]] && echo "取消" && exit 0
    install
    ;;
  *) usage; exit 1 ;;
esac
