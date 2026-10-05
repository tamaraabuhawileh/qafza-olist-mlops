from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from .config import PROJECT_ROOT, load_config


def log_prediction(
    input_data: dict[str, Any],
    result: dict[str, Any],
) -> None:
    """Store one prediction event as JSONL."""

    config = load_config()

    log_path = (
        PROJECT_ROOT
        / config["monitoring"]["prediction_log_path"]
    )

    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input": input_data,
        "prediction": result["prediction"],
        "is_late": result["is_late"],
        "probability": result["probability"],
        "threshold": result["threshold"],
        "model_version": result["model_version"],
        "latency_ms": result["latency_ms"],
    }

    with log_path.open(
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            json.dumps(
                event,
                default=str,
            )
            + "\n"
        )