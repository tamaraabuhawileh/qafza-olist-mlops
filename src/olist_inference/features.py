from __future__ import annotations

import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create prediction-time features using the same logic
    used during Task 2 feature engineering.
    """

    df = df.copy()

    required_date_columns = [
        "order_purchase_timestamp",
        "order_estimated_delivery_date",
    ]

    missing = [
        column
        for column in required_date_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required date columns: {missing}"
        )

    purchase = pd.to_datetime(
        df["order_purchase_timestamp"],
        errors="coerce",
    )

    estimated = pd.to_datetime(
        df["order_estimated_delivery_date"],
        errors="coerce",
    )

    df["purchase_month"] = purchase.dt.month

    df["purchase_weekday"] = purchase.dt.dayofweek

    df["purchase_hour"] = purchase.dt.hour

    df["purchase_weekend"] = (
        purchase.dt.dayofweek >= 5
    ).astype(int)

    df["estimated_delivery_days"] = (
        (estimated - purchase).dt.total_seconds()
        / 86400
    )

    return df

def select_model_features(
    df: pd.DataFrame,
    input_features: list[str],
) -> pd.DataFrame:
    """
    Select model input columns in the exact order
    used during training.
    """

    missing = [
        column
        for column in input_features
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing model input features: {missing}"
        )

    return df[input_features].copy()

