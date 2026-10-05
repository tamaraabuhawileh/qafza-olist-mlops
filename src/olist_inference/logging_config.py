from __future__ import annotations

import logging
from typing import Any

from .config import PROJECT_ROOT, load_config


def setup_logging(
    config: dict[str, Any] | None = None,
) -> logging.Logger:
    """Configure console and file logging."""

    if config is None:
        config = load_config()

    log_config = config["logging"]

    log_dir = PROJECT_ROOT / log_config["directory"]
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / log_config["filename"]

    logger = logging.getLogger("olist_inference")
    logger.setLevel(
        getattr(logging, log_config["level"].upper())
    )

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        log_config["format"]
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(
        log_file,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger