from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.preprocessing import StandardScaler

TARGET_COLUMN = "Class"


@dataclass
class FeaturePreprocessor:
    """Fit feature scaling on training data and reuse it for other splits."""

    scaler: StandardScaler
    feature_columns: list[str]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform features using parameters learned from training data."""
        missing_columns = [
            column
            for column in self.feature_columns
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing feature columns: {missing_columns}"
            )

        transformed = self.scaler.transform(df[self.feature_columns])

        result = pd.DataFrame(
            transformed,
            columns=self.feature_columns,
            index=df.index,
        )

        result[TARGET_COLUMN] = df[TARGET_COLUMN].to_numpy()

        return result


def fit_preprocessor(train_df: pd.DataFrame) -> FeaturePreprocessor:
    """Fit a StandardScaler using training features only."""
    if TARGET_COLUMN not in train_df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found."
        )

    feature_columns = [
        column for column in train_df.columns if column != TARGET_COLUMN
    ]

    scaler = StandardScaler()
    scaler.fit(train_df[feature_columns])

    return FeaturePreprocessor(
        scaler=scaler,
        feature_columns=feature_columns,
    )