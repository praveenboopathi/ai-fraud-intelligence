from pathlib import Path

import pandas as pd
import pytest

from fraud_intelligence.data.ingestion import load_dataset
from fraud_intelligence.data.splitting import (
    remove_exact_duplicates,
    split_dataset,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATASET = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"


def test_remove_exact_duplicates() -> None:
    """Verify exact duplicate rows are removed."""
    df = pd.DataFrame(
        {
            "feature": [1, 1, 2],
            "Class": [0, 0, 1],
        }
    )

    clean = remove_exact_duplicates(df)

    assert len(clean) == 2
    assert clean.duplicated().sum() == 0


def test_split_sizes() -> None:
    """Verify the expected train/validation/test proportions."""
    df = load_dataset(RAW_DATASET)
    clean = remove_exact_duplicates(df)
    splits = split_dataset(clean)

    assert len(splits.train) == 170235
    assert len(splits.validation) == 56745
    assert len(splits.test) == 56746


def test_splits_have_no_overlap() -> None:
    """Verify that no transaction appears in multiple splits."""
    df = load_dataset(RAW_DATASET)
    clean = remove_exact_duplicates(df)
    splits = split_dataset(clean)

    train_rows = set(map(tuple, splits.train.values))
    validation_rows = set(map(tuple, splits.validation.values))
    test_rows = set(map(tuple, splits.test.values))

    assert train_rows.isdisjoint(validation_rows)
    assert train_rows.isdisjoint(test_rows)
    assert validation_rows.isdisjoint(test_rows)


def test_invalid_target_column_is_rejected() -> None:
    """Verify an invalid target column raises an error."""
    df = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4],
            "Class": [0, 0, 1, 1],
        }
    )

    with pytest.raises(ValueError, match="Target column"):
        split_dataset(df, target_column="Fraud")