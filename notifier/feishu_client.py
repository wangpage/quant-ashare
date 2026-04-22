"""lark-cli subprocess 封装: auth preflight + 超时 + 错误归一.

不装 lark_oapi SDK, 完全复用 Claude Code 已有的 lark-cli 1.0.8 认证上下文。
cron 调用时 $HOME 能读到 ~/.lark/profile.json。
"""
from __future__ import annotations

import json
import os
import subprocess
from typing import Optional

from utils.logger import logger

_LARK_CLI = "lark-cli"
_DEFAULT_TIMEOUT = 30


def auth_status() -> dict:
    """拿 lark-cli 的登录状态 JSON. 失败返 {}."""
    try:
        p = subprocess.run(
            [_LARK_CLI, "auth", "status"],
            capture_output=True, text=True, timeout=10,
        )
        if p.returncode != 0:
            logger.warning(f"lark-cli auth status rc={p.returncode}: {p.stderr[:200]}")
            return {}
        return json.loads(p.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError) as e:
        logger.warning(f"lark-cli auth status 异常: {e}")
        return {}


def auth_preflight() -> bool:
    """True = 可推送; False = 跳过本次(不抛异常)."""
    status = auth_status()
    if not status:
        logger.warning("preflight: auth status 空, 不确定是否登录, 尝试推送")
        return True
    token_status = status.get("tokenStatus", "")
    if token_status == "expired":
        logger.error(f"preflight: token 已过期, 需要 lark-cli auth login")
        return False
    if token_status == "needs_refresh":
        logger.info(f"preflight: token 需刷新但仍可用 (expiresAt={status.get('expiresAt')})")
    return True


def send_im(user_id: str, markdown: str, timeout: int = _DEFAULT_TIMEOUT,
             as_user: bool = False) -> bool:
    """发 IM 消息(markdown).

    默认 as_user=False (bot 身份) —— 飞书不允许 user 身份给自己的 open_id 发 P2P,
    必须用 bot 身份才能让消息出现在"机器人推送"会话里。

    Args:
        user_id: 飞书 open_id (ou_ 开头)
        markdown: Markdown 文本
        timeout: 子进程超时秒
        as_user: True 用当前登录 user 身份; False (默认) 用 app 的 bot 身份
    Returns:
        True 表示 lark-cli 退出 0; False 其他.
    """
    if not markdown.strip():
        logger.warning("send_im: 空消息,跳过")
        return False
    as_flag = "user" if as_user else "bot"
    cmd = [
        _LARK_CLI, "im", "+messages-send",
        "--as", as_flag,
        "--user-id", user_id,
        "--markdown", markdown,
    ]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if p.returncode == 0:
            logger.info(f"send_im ✓ to {user_id[:12]}…")
            return True
        logger.error(f"send_im ✗ rc={p.returncode}: {p.stderr[:300] or p.stdout[:300]}")
        return False
    except subprocess.TimeoutExpired:
        logger.error(f"send_im 超时 {timeout}s")
        return False
    except FileNotFoundError:
        logger.error("lark-cli 未找到, 请 brew install / npm install lark-cli")
        return False


# Phase 2 占位, MVP 不实现
def create_doc(*args, **kwargs):
    raise NotImplementedError("Phase 2: 飞书文档通道待实现")


def bitable_add_row(*args, **kwargs):
    raise NotImplementedError("Phase 2: 多维表格通道待实现")


if __name__ == "__main__":
    import sys
    s = auth_status()
    print(json.dumps(s, indent=2, ensure_ascii=False))
    print(f"preflight: {auth_preflight()}")
    if len(sys.argv) > 1 and sys.argv[1] == "--test-send":
        uid = os.environ.get("LARK_USER_OPEN_ID", s.get("userOpenId", ""))
        if uid:
            send_im(uid, "🧪 feishu_client preflight test")
