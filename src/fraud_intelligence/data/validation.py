from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

EXPECTED_COLUMNS = [
    "Time",
    *[f"V{i}" for i in range(1, 29)],
    "Amount",
    "Class",
]

NUMERIC_FEATURES = [
    "Time",
    *[f"V{i}" for i in range(1, 29)],
    "Amount",
]


@dataclass(frozen=True)
class ValidationReport:
    """Summary of dataset validation results."""

    rows: int
    columns: int
    missing_values: int
    duplicate_rows: int


def validate_schema(df: pd.DataFrame) -> None:
    """Validate the expected dataset schema."""
    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            "Unexpected dataset schema. "
            f"Expected {EXPECTED_COLUMNS}, got {list(df.columns)}"
        )


def validate_values(df: pd.DataFrame) -> None:
    """Validate required values and numeric constraints."""
    missing_values = int(df.isna().sum().sum())

    if missing_values:
        raise ValueError(
            f"Dataset contains {missing_values} missing values."
        )

    if not df[NUMERIC_FEATURES].apply(
        lambda column: pd.api.types.is_numeric_dtype(column)
    ).all():
        raise ValueError("Expected all feature columns to be numeric.")

    if not df["Class"].isin([0, 1]).all():
        raise ValueError("Class must contain only 0 and 1.")

    if (df["Time"] < 0).any():
        raise ValueError("Time must be non-negative.")

    if (df["Amount"] < 0).any():
        raise ValueError("Amount must be non-negative.")


def validate_dataset(df: pd.DataFrame) -> ValidationReport:
    """Validate a dataframe and return a quality summary."""
    validate_schema(df)
    validate_values(df)

    return ValidationReport(
        rows=len(df),
        columns=len(df.columns),
        missing_values=int(df.isna().sum().sum()),
        duplicate_rows=int(df.duplicated().sum()),
    )