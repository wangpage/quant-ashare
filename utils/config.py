"""配置加载器.

支持 ${ENV_VAR:-default} 语法, 在 yaml 加载时自动展开环境变量.
示例: password: "${LEVEL2_PASSWORD:-level2@test}"
     -> 运行时若 LEVEL2_PASSWORD 未设置, 回落到 "level2@test" (公开测试账号)
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import yaml


_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"

# 匹配 ${VAR} 或 ${VAR:-default}
_ENV_RE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)(?::-([^}]*))?\}")


def _expand_env(value):
    if isinstance(value, str):
        def repl(m):
            name, default = m.group(1), m.group(2)
            return os.environ.get(name, default if default is not None else "")
        return _ENV_RE.sub(repl, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def load_config(path: Path | str | None = None) -> dict:
    path = Path(path) if path else _CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _expand_env(raw)


CONFIG = load_config()
PROJECT_ROOT = Path(__file__).resolve().parent.parent
