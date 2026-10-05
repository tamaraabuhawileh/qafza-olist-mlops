from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def _resolve_path(value: str) -> Path:
    """Convert a configured relative path to an absolute project path."""
    path = Path(value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def load_config() -> dict[str, Any]:
    """Load project configuration."""
    load_dotenv(PROJECT_ROOT / ".env")

    config_path = Path(
        os.getenv(
            "OLIST_CONFIG_PATH",
            str(DEFAULT_CONFIG_PATH),
        )
    )

    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise TypeError(
        "Configuration file must contain a YAML mapping."
    )

    return config


def get_artifact_paths(
    config: dict[str, Any],
) -> dict[str, Path]:
    """Resolve artifact paths from configuration."""

    artifact_config = config["artifacts"]

    return {
        name: _resolve_path(path)
        for name, path in artifact_config.items()
    }


def get_log_directory(
    config: dict[str, Any],
) -> Path:
    """Return the configured logging directory."""

    return _resolve_path(
        config["logging"]["directory"]
    )
def validate_artifact_paths(
    config: dict[str, Any],
) -> None:
    """Check that required inference artifacts exist."""

    paths = get_artifact_paths(config)

    required = [
        "preprocessor_path",
        "feature_list_path",
        "input_features_path",
        "model_path",
    ]

    missing = [
        str(paths[name])
        for name in required
        if not paths[name].exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Required inference artifacts are missing:\n"
            + "\n".join(missing)
        )