from __future__ import annotations

import pandas as pd

BRAZILIAN_STATES = {
    "AC", "AL", "AM", "AP", "BA", "CE", "DF",
    "ES", "GO", "MA", "MG", "MS", "MT", "PA",
    "PB", "PE", "PI", "PR", "RJ", "RN", "RO",
    "RR", "RS", "SC", "SE", "SP", "TO",
}


def validate_input(
    df: pd.DataFrame,
    input_features: list[str],
) -> None:
    """Validate model-ready input before preprocessing."""

    if df.empty:
        raise ValueError("Input data is empty.")

    missing_columns = [
        column
        for column in input_features
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required features: {missing_columns}"
        )

    if "customer_state" in df.columns:
        invalid_states = set(
            df["customer_state"].dropna().astype(str)
        ) - BRAZILIAN_STATES

        if invalid_states:
            raise ValueError(
                f"Invalid customer_state values: "
                f"{sorted(invalid_states)}"
            )

    if "estimated_delivery_days" in df.columns:
        invalid_delivery = (
            df["estimated_delivery_days"].notna()
            & (df["estimated_delivery_days"] < 0)
        )

        if invalid_delivery.any():
            raise ValueError(
                "estimated_delivery_days cannot be negative."
            )
def validate_raw_input(df: pd.DataFrame) -> None:
    """Validate fields required before feature engineering."""

    required = [
        "order_purchase_timestamp",
        "order_estimated_delivery_date",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required raw fields: {missing}"
        )

    purchase = pd.to_datetime(
        df["order_purchase_timestamp"],
        errors="coerce",
    )

    estimated = pd.to_datetime(
        df["order_estimated_delivery_date"],
        errors="coerce",
    )

    if purchase.isna().any():
        raise ValueError(
            "Invalid order_purchase_timestamp."
        )

    if estimated.isna().any():
        raise ValueError(
            "Invalid order_estimated_delivery_date."
        )

    if (estimated < purchase).any():
        raise ValueError(
            "Estimated delivery date cannot be "
            "before purchase date."
        )