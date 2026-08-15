from __future__ import annotations

import pandas as pd
from sklearn.ensemble import IsolationForest


def fit_isolation_forest(
    features: pd.DataFrame,
) -> IsolationForest:
    model = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42,
    )
    model.fit(features)
    return model


def detect_anomalies(
    model: IsolationForest,
    features: pd.DataFrame,
) -> pd.Series:
    predictions = model.predict(features)

    return pd.Series(
        (predictions == -1).astype(int),
        index=features.index,
        name="anomaly",
    )