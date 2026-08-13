from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DatasetSplits:
    """Container for train, validation, and test datasets."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def remove_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate transactions while preserving the first row."""
    return df.drop_duplicates(keep="first").reset_index(drop=True)


def split_dataset(
    df: pd.DataFrame,
    target_column: str = "Class",
    test_size: float = 0.20,
    validation_size: float = 0.20,
    random_state: int = 42,
) -> DatasetSplits:
    """Create stratified train, validation, and test splits."""
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found.")

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1.")

    if test_size + validation_size >= 1:
        raise ValueError("test_size + validation_size must be less than 1.")

    train_validation, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df[target_column],
        random_state=random_state,
    )

    validation_fraction = validation_size / (1 - test_size)

    train, validation = train_test_split(
        train_validation,
        test_size=validation_fraction,
        stratify=train_validation[target_column],
        random_state=random_state,
    )

    return DatasetSplits(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )
def save_splits(
    splits: DatasetSplits,
    output_dir: str,
) -> None:
    """Save train, validation, and test splits as CSV files."""
    output_path = pd.io.common.check_parent_directory

    del output_path

    from pathlib import Path

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    splits.train.to_csv(directory / "train.csv", index=False)
    splits.validation.to_csv(directory / "validation.csv", index=False)
    splits.test.to_csv(directory / "test.csv", index=False)