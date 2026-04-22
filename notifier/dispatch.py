"""分析师推送 CLI 入口.

退出码约定 (被 cron_daily.py 识别):
  0  推送成功
  10 auth 过期, 主 cron 应打标但不失败
  1  其他异常

MVP 只走 IM 通道, Doc/Base 在 phase 2 加。
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# 允许从项目根目录 python -m 调用
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# 加载 .env (复用 predict_tomorrow.py 的轻量模式)
_env = _ROOT / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

from analyst.brief_builder import build
from analyst.im_formatter import format_im
from notifier import feishu_client
from utils.logger import logger

_DEFAULT_USER = "ou_5be0f87dc7cec796b7ea97d0a9b5302f"


def main() -> int:
    ap = argparse.ArgumentParser(description="A股分析师简报推送")
    ap.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"))
    ap.add_argument("--top-n", type=int, default=3,
                    help="明日关注 Top N, 默认 3")
    ap.add_argument("--user-id", default=os.environ.get("LARK_USER_OPEN_ID", _DEFAULT_USER),
                    help="飞书接收者 open_id")
    ap.add_argument("--dry-run", action="store_true",
                    help="不发送飞书, 只打印 Markdown 到 stdout")
    args = ap.parse_args()

    # 1. Preflight
    if not args.dry_run and not feishu_client.auth_preflight():
        logger.error("auth preflight 失败, 请运行 `lark-cli auth login`")
        return 10

    # 2. Build brief
    try:
        brief = build(args.date, top_n=args.top_n)
    except Exception as e:
        logger.exception(f"brief build 失败: {e}")
        return 1

    # 3. Format IM
    try:
        markdown = format_im(brief)
    except Exception as e:
        logger.exception(f"format_im 失败: {e}")
        return 1

    if args.dry_run:
        print(markdown)
        return 0

    # 4. Send IM
    ok = feishu_client.send_im(args.user_id, markdown)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
