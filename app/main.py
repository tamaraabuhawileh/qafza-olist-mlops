from __future__ import annotations

from fastapi import FastAPI, HTTPException

from src.olist_inference.artifacts import load_artifacts
from src.olist_inference.config import load_config
from src.olist_inference.predictor import OlistPredictor
from src.olist_inference.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelInfoResponse,
    OrderInput,
    PredictionResponse,
)

config = load_config()
artifacts = load_artifacts()
predictor = OlistPredictor()


app = FastAPI(
    title=config["service"]["name"],
    version=config["service"]["version"],
    description="Olist late-delivery inference service",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "healthy",
    }


@app.get(
    "/model",
    response_model=ModelInfoResponse,
)
def model_info() -> ModelInfoResponse:
    return ModelInfoResponse(
        service_name=config["service"]["name"],
        service_version=config["service"]["version"],
        model_type=type(artifacts["model"]).__name__,
        model_version=predictor.model_version,
        threshold=predictor.threshold,
        input_feature_count=len(
            predictor.input_features
        ),
        transformed_feature_count=len(
            predictor.feature_list
        ),
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    order: OrderInput,
) -> PredictionResponse:
    try:
        data = order.model_dump()

        result = predictor.predict(data)

        return PredictionResponse(**result)

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed.",
        ) from exc
@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
)
def predict_batch(
    request: BatchPredictionRequest,
) -> BatchPredictionResponse:
    try:
        predictions = []

        for order in request.orders:
            result = predictor.predict(
                order.model_dump()
            )

            predictions.append(
                PredictionResponse(**result)
            )

        return BatchPredictionResponse(
            predictions=predictions
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Batch prediction failed.",
        ) from exc