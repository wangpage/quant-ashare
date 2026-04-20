"""Level2 配置校验 - 启动时 fail-fast, 不允许模糊占位符绕过.

设计原则:
    1. **显式比隐式好**: 占位符用 sentinel 常量, 不靠 "TODO" 字符串撞猜
    2. **早失败比晚失败好**: 配置 load 时就校验, 不要等盘中 NATS 连不上才发现
    3. **无效 URL 必须抛错**: 宁可启动失败, 也别带着假配置进生产
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlparse


# 明确的占位符 sentinel - 配置里出现这些值 = 未填
UNSET_SENTINELS = {
    "", "REPLACE_ME", "REPLACE_WITH_REAL", "TODO", "FIXME",
    "YOUR_KEY", "YOUR_PASSWORD", "CHANGEME", "XXX",
}


class ConfigurationError(RuntimeError):
    """显式配置错误, 区别于运行时偶发的 ConnectionError."""


@dataclass
class ValidatedServer:
    active_name: str         # 'shanghai' / 'guangzhou'
    primary_url: str
    backup_url: str | None

    @property
    def urls(self) -> list[str]:
        out = [self.primary_url]
        if self.backup_url:
            out.append(self.backup_url)
        return out


_PLACEHOLDER_TOKEN_RE = re.compile(
    r"(^|[\W_/:])(TODO|FIXME|REPLACE(_ME|_WITH_REAL)?|XXX|CHANGEME)"
    r"([\W_/:]|$)",
    re.IGNORECASE,
)


def _is_placeholder(val) -> bool:
    """判断一个值是否是占位符 (而不是真实凭证/URL).

    识别策略 (按严格 → 宽松):
        1. 完全匹配 UNSET_SENTINELS 空字符串/已知值
        2. 未展开的 ${VAR} (但 ${VAR:-default} 合法)
        3. 正则匹配 TODO/FIXME/REPLACE_ME/XXX/CHANGEME 作为 token
           (token 边界: 行首/行尾/非字母数字字符)
           覆盖 'nats://TODO_IP:8888', 'my-FIXME-host' 等嵌入式占位符
    """
    if val is None:
        return True
    if not isinstance(val, str):
        return False
    s = val.strip()
    if s in UNSET_SENTINELS:
        return True
    # 未展开的 ${VAR} 或 ${VAR:-default}
    if s.startswith("${") and s.endswith("}") and ":-" not in s:
        return True
    # 嵌入式占位符: nats://TODO:8888 / host-FIXME-01 等
    if _PLACEHOLDER_TOKEN_RE.search(s):
        return True
    return False


def validate_nats_url(url: str, *, label: str = "url") -> str:
    """校验一个 NATS URL 结构完整.

    合法示例:
        nats://host:4222
        nats://host:4222,nats://host2:4222
        tls://host:4222
    非法示例:
        nats://TODO:4222
        nats://:4222        (无 host)
        host:4222           (无 scheme)
    """
    if _is_placeholder(url):
        raise ConfigurationError(
            f"{label} 仍是占位符/未设置: {url!r}. 请修改 config/level2.yaml "
            f"或设置对应环境变量."
        )

    parts = urlparse(url if "://" in url else f"nats://{url}")
    if parts.scheme not in {"nats", "tls", "natss"}:
        raise ConfigurationError(
            f"{label} 协议非法: {url!r} (scheme={parts.scheme}). 期望 nats:// / tls:// / natss://"
        )
    if not parts.hostname:
        raise ConfigurationError(f"{label} 无 hostname: {url!r}")
    # urlparse.port 会在越界时抛 ValueError, 手动处理
    try:
        port = parts.port
    except ValueError:
        raise ConfigurationError(f"{label} 端口格式非法 (越界): {url!r}")
    if port is None:
        raise ConfigurationError(f"{label} 无端口号: {url!r}")
    if not (1 <= port <= 65535):
        raise ConfigurationError(f"{label} 端口越界: {port}")
    # IP 或 DNS 名简单形状校验
    host = parts.hostname
    if not (re.match(r"^[a-zA-Z0-9.\-]+$", host) or ":" in host):
        raise ConfigurationError(f"{label} hostname 形状可疑: {host}")
    return url


def validate_level2_config(cfg: dict) -> ValidatedServer:
    """完整校验 level2.yaml. 任何问题抛 ConfigurationError.

    Returns:
        ValidatedServer: 已通过校验的主/备 URL, 可直接交给 nats.connect().
    """
    if not isinstance(cfg, dict):
        raise ConfigurationError(f"配置根必须是 dict, 得到 {type(cfg)}")

    conn = cfg.get("connection")
    if not conn:
        raise ConfigurationError("缺 connection 段")

    active = conn.get("active")
    if not active:
        raise ConfigurationError("缺 connection.active")

    servers = conn.get("servers", {})
    if active not in servers:
        raise ConfigurationError(
            f"connection.servers 中没有 {active!r}; "
            f"可选: {list(servers.keys())}"
        )

    server = servers[active]
    primary = server.get("host")
    if not primary:
        raise ConfigurationError(
            f"servers.{active}.host 未设置 (必需)"
        )
    primary = validate_nats_url(primary, label=f"servers.{active}.host")

    backup_raw = server.get("backup")
    backup = None
    if backup_raw and not _is_placeholder(backup_raw):
        # backup 允许缺失, 但如果填了就必须合法
        backup = validate_nats_url(backup_raw, label=f"servers.{active}.backup")

    # 凭证校验
    auth = conn.get("auth", {})
    user = auth.get("user")
    password = auth.get("password")
    if _is_placeholder(user) or _is_placeholder(password):
        raise ConfigurationError(
            "Level2 凭证未设置 (或仍是占位符). "
            "设置环境变量 LEVEL2_USER / LEVEL2_PASSWORD, "
            "或使用测试账号 level2_test / level2@test."
        )

    return ValidatedServer(
        active_name=active,
        primary_url=primary,
        backup_url=backup,
    )
