from pathlib import Path

import pandas as pd
import pytest

from fraud_intelligence.data.ingestion import load_dataset
from fraud_intelligence.data.validation import validate_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATASET = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"


def test_load_dataset() -> None:
    """Verify that the raw CSV can be loaded."""
    df = load_dataset(RAW_DATASET)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (284807, 31)


def test_validate_real_dataset() -> None:
    """Verify the real dataset passes validation."""
    df = load_dataset(RAW_DATASET)
    report = validate_dataset(df)

    assert report.rows == 284807
    assert report.columns == 31
    assert report.missing_values == 0
    assert report.duplicate_rows == 1081


def test_invalid_class_is_rejected() -> None:
    """Verify invalid target values are rejected."""
    df = pd.DataFrame(
        {
            "Time": [0.0],
            **{f"V{i}": [0.0] for i in range(1, 29)},
            "Amount": [10.0],
            "Class": [2],
        }
    )

    with pytest.raises(ValueError, match="Class must contain only 0 and 1"):
        validate_dataset(df)


def test_negative_amount_is_rejected() -> None:
    """Verify negative transaction amounts are rejected."""
    df = pd.DataFrame(
        {
            "Time": [0.0],
            **{f"V{i}": [0.0] for i in range(1, 29)},
            "Amount": [-10.0],
            "Class": [0],
        }
    )

    with pytest.raises(ValueError, match="Amount must be non-negative"):
        validate_dataset(df)