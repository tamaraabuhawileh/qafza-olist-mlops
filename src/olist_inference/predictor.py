from __future__ import annotations

import time
from typing import Any

import pandas as pd

from .artifacts import load_artifacts
from .features import build_features, select_model_features
from .logging_config import setup_logging
from .prediction_logger import log_prediction
from .validation import validate_input, validate_raw_input

logger = setup_logging()


class OlistPredictor:
    """Inference pipeline for late-delivery prediction."""

    def __init__(self) -> None:
        artifacts = load_artifacts()

        self.preprocessor = artifacts["preprocessor"]
        self.model = artifacts["model"]
        self.input_features = artifacts["input_features"]
        self.feature_list = artifacts["feature_list"]
        self.model_config = artifacts["model_config"]
        self.threshold = artifacts["threshold"]

        self.model_version = str(
            self.model_config.get("version", "1.0.0")
        )

        logger.info(
            "Predictor initialized | model=%s | "
            "version=%s | threshold=%.2f",
            type(self.model).__name__,
            self.model_version,
            self.threshold,
        )

    def predict(
        self,
        data: dict[str, Any] | pd.DataFrame,
    ) -> dict[str, Any]:
        """Predict whether one order will be late."""

        start_time = time.perf_counter()

        logger.info(
            "Prediction request received | rows=%d",
            1 if isinstance(data, dict) else len(data),
        )

        try:
            if isinstance(data, dict):
                df = pd.DataFrame([data])
            else:
                df = data.copy()

            if len(df) != 1:
                raise ValueError(
                    "predict() expects exactly one order."
                )

            # 1. Validate raw input
            validate_raw_input(df)

            # 2. Reproduce Task 2 feature engineering
            df = build_features(df)

            # 3. Select exact training features
            X = select_model_features(
                df,
                self.input_features,
            )

            # 4. Validate model-ready input
            validate_input(
                X,
                self.input_features,
            )

            # 5. Apply fitted Task 2 preprocessor
            X_transformed = self.preprocessor.transform(X)

            # 6. Verify transformed feature count
            if X_transformed.shape[1] != len(
                self.feature_list
            ):
                raise ValueError(
                    "Transformed feature count does not "
                    "match the saved Task 2 feature list."
                )

            # 7. Calculate probability of Late
            probability = float(
                self.model.predict_proba(
                    X_transformed
                )[0, 1]
            )

            # 8. Apply saved threshold
            is_late = int(
                probability >= self.threshold
            )

            prediction = (
                "Late"
                if is_late == 1
                else "On Time"
            )

            latency_ms = (
                time.perf_counter() - start_time
            ) * 1000

            result = {
                "prediction": prediction,
                "is_late": is_late,
                "probability": probability,
                "threshold": self.threshold,
                "model_version": self.model_version,
                "latency_ms": latency_ms,
            }

            logger.info(
                "Prediction completed | "
                "prediction=%s | "
                "probability=%.6f | "
                "threshold=%.2f | "
                "model_version=%s | "
                "latency_ms=%.2f",
                prediction,
                probability,
                self.threshold,
                self.model_version,
                latency_ms,
            )
            input_data = df.iloc[0].to_dict()

            log_prediction(
            input_data=input_data,
            result=result,
            )
            return result

        except Exception:
            latency_ms = (
                time.perf_counter() - start_time
            ) * 1000

            logger.exception(
                "Prediction failed | "
                "model_version=%s | "
                "latency_ms=%.2f",
                self.model_version,
                latency_ms,
            )

            raise