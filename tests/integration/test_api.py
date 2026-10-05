import math
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_valid_order() -> dict:
    """Load one real order from the Task 2 training data."""

    path = (
        PROJECT_ROOT
        / "artifacts"
        / "03_splits"
        / "train.parquet"
    )

    df = pd.read_parquet(path).head(1)
    row = df.iloc[0].to_dict()

    fields = {
        "order_purchase_timestamp",
        "order_estimated_delivery_date",
        "item_count",
        "unique_product_count",
        "unique_seller_count_x",
        "total_item_price",
        "mean_item_price",
        "max_item_price",
        "total_freight_value",
        "payment_count",
        "payment_type_count",
        "max_payment_installments",
        "total_payment_value",
        "unique_category_count",
        "unique_seller_state_count",
        "mean_product_weight_g",
        "max_product_weight_g",
        "mean_product_length_cm",
        "mean_product_height_cm",
        "mean_product_width_cm",
        "mean_seller_latitude",
        "mean_seller_longitude",
        "customer_state",
    }

    order = {
        key: value
        for key, value in row.items()
        if key in fields
    }

    order["order_purchase_timestamp"] = str(
        row["order_purchase_timestamp"]
    )
    order["order_estimated_delivery_date"] = str(
        row["order_estimated_delivery_date"]
    )

    # Convert pandas NaN to JSON-compatible null.
    for key, value in order.items():
        if isinstance(value, float) and math.isnan(value):
            order[key] = None

    return order


def test_single_prediction():
    order = get_valid_order()

    response = client.post(
        "/predict",
        json=order,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] in {
        "Late",
        "On Time",
    }

    assert data["is_late"] in {0, 1}
    assert 0 <= data["probability"] <= 1
    assert data["threshold"] == 0.4


def test_bad_input_is_rejected():
    order = get_valid_order()

    order["order_estimated_delivery_date"] = (
        "2010-01-01T00:00:00"
    )

    response = client.post(
        "/predict",
        json=order,
    )

    assert response.status_code == 422


def test_batch_prediction():
    order = get_valid_order()

    response = client.post(
        "/predict/batch",
        json={
            "orders": [
                order,
                order,
            ]
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data["predictions"]) == 2