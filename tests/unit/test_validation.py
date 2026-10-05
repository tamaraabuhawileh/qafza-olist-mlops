import pandas as pd
import pytest

from src.olist_inference.validation import (
    validate_raw_input,
)


def test_valid_dates():
    df = pd.DataFrame(
        {
            "order_purchase_timestamp": [
                "2018-01-01"
            ],
            "order_estimated_delivery_date": [
                "2018-01-10"
            ],
        }
    )

    validate_raw_input(df)


def test_estimated_date_before_purchase_fails():
    df = pd.DataFrame(
        {
            "order_purchase_timestamp": [
                "2018-01-10"
            ],
            "order_estimated_delivery_date": [
                "2018-01-01"
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="Estimated delivery date",
    ):
        validate_raw_input(df)


def test_missing_timestamp_fails():
    df = pd.DataFrame(
        {
            "order_purchase_timestamp": [
                "2018-01-01"
            ]
        }
    )

    with pytest.raises(
        ValueError,
        match="Missing required raw fields",
    ):
        validate_raw_input(df)