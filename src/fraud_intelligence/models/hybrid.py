from __future__ import annotations

import pandas as pd

from fraud_intelligence.models.supervised import (
    ClassificationMetrics,
    evaluate_classifier,
    prepare_features,
    train_logistic_regression,
)
from fraud_intelligence.models.unsupervised import (
    detect_anomalies,
    fit_isolation_forest,
)


def train_hybrid_models(
    train_df: pd.DataFrame,
) -> tuple[object, object]:
    """Train supervised and unsupervised fraud models."""

    x_train, y_train = prepare_features(train_df)

    supervised_model = train_logistic_regression(
        x_train,
        y_train,
    )

    unsupervised_model = fit_isolation_forest(
        x_train,
    )

    return supervised_model, unsupervised_model


def predict_hybrid(
    supervised_model: object,
    unsupervised_model: object,
    features: pd.DataFrame,
    threshold: float = 0.95,
) -> pd.DataFrame:
    """Generate combined supervised and unsupervised predictions."""

    probabilities = supervised_model.predict_proba(features)[:, 1]
    supervised_prediction = (
        probabilities >= threshold
    ).astype(int)

    anomaly_prediction = detect_anomalies(
        unsupervised_model,
        features,
    )

    result = pd.DataFrame(
        {
            "fraud_probability": probabilities,
            "supervised_prediction": supervised_prediction,
            "anomaly_prediction": anomaly_prediction,
        },
        index=features.index,
    )

    result["hybrid_prediction"] = (
        (result["supervised_prediction"] == 1)
        | (result["anomaly_prediction"] == 1)
    ).astype(int)

    return result


def evaluate_hybrid(
    supervised_model: object,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.95,
) -> ClassificationMetrics:
    """Evaluate the supervised component of the hybrid system."""

    return evaluate_classifier(
        supervised_model,
        x_test,
        y_test,
        threshold,
    )