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
    """Build transaction-level features and train the supervised model."""

    features = build_features(df)

    feature_columns = [
        "Amount",
        "Time",
        "Amount_log",
        "Time_hour",
        "Class",
    ]

    features = features[feature_columns]

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
    """Build transaction features, fit anomaly detector, and return predictions."""

    features = build_features(df)

    # Use only transaction-level features available from the dashboard.
    feature_columns = [
        "Amount",
        "Time",
        "Amount_log",
        "Time_hour",
    ]

    features = features[feature_columns]

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
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Fraud Detection API")


class TransactionRequest(BaseModel):
    Amount: float
    Time: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
def analyze_transaction(transaction: TransactionRequest):
    df = pd.DataFrame(
        [
            {
                "Amount": transaction.Amount,
                "Time": transaction.Time,
                "Class": 0,
            }
        ]
    )

    supervised_model, metrics = run_supervised_pipeline(df)

    features = build_features(df)
    feature_columns = [
        "Amount",
        "Time",
        "Amount_log",
        "Time_hour",
    ]

    anomaly_features = features[feature_columns]

    anomaly_model = fit_isolation_forest(anomaly_features)
    anomaly = int(
        detect_anomalies(anomaly_model, anomaly_features).iloc[0]
    )

    probability = float(
        supervised_model.predict_proba(
            features[
                [
                    "Amount",
                    "Time",
                    "Amount_log",
                    "Time_hour",
                ]
            ]
        )[0][1]
    )

    return {
        "fraud_probability": probability,
        "fraud_prediction": int(probability >= 0.5),
        "anomaly": anomaly,
        "metrics": {
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "roc_auc": metrics.roc_auc,
            "pr_auc": metrics.pr_auc,
        },
    }