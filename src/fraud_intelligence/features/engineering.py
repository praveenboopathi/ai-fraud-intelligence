from __future__ import annotations

import pandas as pd

TARGET_COLUMN = "Class"


def add_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic transaction-level features."""
    result = df.copy()

    if "Amount" in result.columns:
        result["Amount_log"] = result["Amount"].clip(lower=0).add(1).apply(
            lambda value: __import__("math").log(value)
        )

    if "Time" in result.columns:
        result["Time_hour"] = (result["Time"] / 3600) % 24

    return result


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build model-ready features while preserving the target column."""
    result = add_transaction_features(df)

    if TARGET_COLUMN not in result.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    return result