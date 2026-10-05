from pathlib import Path

import pandas as pd
import pytest

from src.olist_inference.artifacts import load_artifacts
from src.olist_inference.features import (
    build_features,
    select_model_features,
)
from src.olist_inference.predictor import OlistPredictor

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_notebook_inference_parity():
    path = (
        PROJECT_ROOT
        / "artifacts"
        / "03_splits"
        / "train.parquet"
    )

    raw = pd.read_parquet(path).head(1)

    artifacts = load_artifacts()

    # Task 2 / notebook-style inference
    features = build_features(raw)

    X = select_model_features(
        features,
        artifacts["input_features"],
    )

    transformed = artifacts[
        "preprocessor"
    ].transform(X)

    notebook_probability = float(
        artifacts["model"].predict_proba(
            transformed
        )[0, 1]
    )

    # Task 3 production inference
    predictor = OlistPredictor()

    service_result = predictor.predict(raw)

    assert service_result[
        "probability"
    ] == pytest.approx(
        notebook_probability,
        abs=1e-12,
    )