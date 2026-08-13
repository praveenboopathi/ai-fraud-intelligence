from __future__ import annotations

from pathlib import Path

import pandas as pd

SUPPORTED_FORMATS = {".csv", ".parquet"}


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a CSV or Parquet dataset."""
    dataset_path = Path(path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    suffix = dataset_path.suffix.lower()

    if suffix not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported dataset format: {suffix}. "
            f"Supported formats: {sorted(SUPPORTED_FORMATS)}"
        )

    if suffix == ".csv":
        return pd.read_csv(dataset_path)

    return pd.read_parquet(dataset_path)