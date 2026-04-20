"""Logger wrapper: loguru 如果没装, fallback 到 stdlib logging."""
from __future__ import annotations

import sys
from pathlib import Path

from .config import PROJECT_ROOT

_LOG_DIR = PROJECT_ROOT / "logs"
_LOG_DIR.mkdir(exist_ok=True)

try:
    from loguru import logger  # type: ignore
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}:{line}</cyan> | {message}",
        level="INFO",
    )
    logger.add(
        _LOG_DIR / "quant_{time:YYYY-MM-DD}.log",
        rotation="00:00",
        retention="30 days",
        level="DEBUG",
        encoding="utf-8",
    )
except ImportError:
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("quant_ashare")

__all__ = ["logger"]
