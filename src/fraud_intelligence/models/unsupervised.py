from __future__ import annotations

import pandas as pd
from sklearn.ensemble import IsolationForest


def fit_isolation_forest(
    features: pd.DataFrame,
    contamination: float = "auto",
    random_state: int = 42,
) -> IsolationForest:
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
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