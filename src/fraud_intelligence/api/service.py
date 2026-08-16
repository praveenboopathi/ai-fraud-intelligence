from __future__ import annotations

import pandas as pd

from fraud_intelligence.features.engineering import build_features
from fraud_intelligence.models.supervised import (
    evaluate_classifier,
    prepare_features,
    train_logistic_regression,
)
from fraud_intelligence.models.unsupervised import (
    detect_anomalies,
    fit_isolation_forest,
)


def run_supervised_pipeline(
    df: pd.DataFrame,
    random_state: int = 42,
):
    """Build features, train the supervised model, and return metrics."""
    features = build_features(df)
    x, y = prepare_features(features)

    model = train_logistic_regression(
        x,
        y,
        random_state=random_state,
    )

    metrics = evaluate_classifier(
        model,
        x,
        y,
    )

    return model, metrics


def run_unsupervised_pipeline(
    df: pd.DataFrame,
    contamination: float = "auto",
    random_state: int = 42,
):
    """Build features, fit anomaly detector, and return predictions."""
    features = build_features(df)

    model = fit_isolation_forest(
        features,
        contamination=contamination,
        random_state=random_state,
    )

    predictions = detect_anomalies(
        model,
        features,
    )

    return model, predictions