from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class OrderInput(BaseModel):
    order_purchase_timestamp: datetime
    order_estimated_delivery_date: datetime

    item_count: float | None = None
    unique_product_count: float | None = None
    unique_seller_count_x: float | None = None

    total_item_price: float | None = None
    mean_item_price: float | None = None
    max_item_price: float | None = None
    total_freight_value: float | None = None

    payment_count: float | None = None
    payment_type_count: float | None = None
    max_payment_installments: float | None = None
    total_payment_value: float | None = None

    unique_category_count: float | None = None
    unique_seller_state_count: float | None = None

    mean_product_weight_g: float | None = None
    max_product_weight_g: float | None = None
    mean_product_length_cm: float | None = None
    mean_product_height_cm: float | None = None
    mean_product_width_cm: float | None = None

    mean_seller_latitude: float | None = None
    mean_seller_longitude: float | None = None

    customer_state: str


class PredictionResponse(BaseModel):
    prediction: str
    is_late: int
    probability: float
    threshold: float
    model_version: str
    latency_ms: float


class ModelInfoResponse(BaseModel):
    service_name: str
    service_version: str
    model_type: str
    model_version: str
    threshold: float
    input_feature_count: int
    transformed_feature_count: int

class BatchPredictionRequest(BaseModel):
    orders: list[OrderInput]


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]