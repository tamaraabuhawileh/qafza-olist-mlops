from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from .config import (
    get_artifact_paths,
    load_config,
    validate_artifact_paths,
)


def load_json(path: Path) -> dict | list:
    """Load a JSON artifact."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_artifacts(
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load fitted inference artifacts produced by Task 2."""

    if config is None:
        config = load_config()

    validate_artifact_paths(config)

    paths = get_artifact_paths(config)

    preprocessor = joblib.load(
        paths["preprocessor_path"]
    )

    model = joblib.load(
        paths["model_path"]
    )

    feature_list = load_json(
        paths["feature_list_path"]
    )

    input_features = load_json(
        paths["input_features_path"]
    )

    model_config = {}

    model_config_path = paths.get("model_config_path")

    if (
        model_config_path is not None
        and model_config_path.exists()
    ):
        model_config = load_json(model_config_path)

    threshold = model_config.get(
        "threshold",
        config["prediction"]["default_threshold"],
    )

    return {
        "preprocessor": preprocessor,
        "model": model,
        "feature_list": feature_list,
        "input_features": input_features,
        "model_config": model_config,
        "threshold": float(threshold),
    }