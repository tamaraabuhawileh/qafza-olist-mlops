"""Load inference settings and resolve project-relative paths."""

import json
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(value: str | Path) -> Path:
    """Resolve relative paths against the project root."""
    path = Path(value).expanduser()

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load configuration without loading or fitting model objects."""
    selected_path = (
        config_path
        if config_path is not None
        else os.getenv("OLIST_CONFIG", "config/inference.json")
    )
    path = resolve_path(selected_path)

    with path.open(encoding="utf-8") as file:
        config = json.load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration must be a JSON object.")

    required_sections = ("service", "artifacts", "prediction", "logging")

    for section in required_sections:
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing or invalid configuration section: {section}")

    required_artifacts = (
        "preprocessor_path",
        "input_features_path",
        "transformed_features_path",
        "model_path",
        "model_config_path",
    )

    for key in required_artifacts:
        value = config["artifacts"].get(key)

        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Missing or invalid artifact path: {key}")

        config["artifacts"][key] = resolve_path(value)

    log_directory = config["logging"].get("directory")

    if not isinstance(log_directory, str) or not log_directory.strip():
        raise ValueError("Missing or invalid logging directory.")

    config["logging"]["directory"] = resolve_path(log_directory)

    return config