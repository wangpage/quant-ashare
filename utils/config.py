from __future__ import annotations

from pathlib import Path
import yaml


_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"


def load_config(path: Path | str | None = None) -> dict:
    path = Path(path) if path else _CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


CONFIG = load_config()
PROJECT_ROOT = Path(__file__).resolve().parent.parent
