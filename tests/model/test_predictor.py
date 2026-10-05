from pathlib import Path

import pandas as pd

from src.olist_inference.predictor import OlistPredictor

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_predictor_output():
    path = (
        PROJECT_ROOT
        / "artifacts"
        / "03_splits"
        / "train.parquet"
    )

    df = pd.read_parquet(path).head(1)

    predictor = OlistPredictor()

    result = predictor.predict(df)

    assert result["prediction"] in {
        "Late",
        "On Time",
    }

    assert result["is_late"] in {0, 1}

    assert 0 <= result["probability"] <= 1

    assert result["threshold"] == 0.4

    assert result["latency_ms"] >= 0