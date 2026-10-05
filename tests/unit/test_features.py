import pandas as pd

from src.olist_inference.features import (
    build_features,
    select_model_features,
)


def test_build_features():
    df = pd.DataFrame(
        {
            "order_purchase_timestamp": [
                "2018-01-05 10:30:00"
            ],
            "order_estimated_delivery_date": [
                "2018-01-15 10:30:00"
            ],
        }
    )

    result = build_features(df)

    assert result["purchase_month"].iloc[0] == 1
    assert result["purchase_weekday"].iloc[0] == 4
    assert result["purchase_hour"].iloc[0] == 10
    assert result["purchase_weekend"].iloc[0] == 0
    assert result["estimated_delivery_days"].iloc[0] == 10


def test_select_model_features():
    df = pd.DataFrame(
        {
            "feature_a": [1],
            "feature_b": [2],
            "unused": [3],
        }
    )

    result = select_model_features(
        df,
        ["feature_a", "feature_b"],
    )

    assert list(result.columns) == [
        "feature_a",
        "feature_b",
    ]