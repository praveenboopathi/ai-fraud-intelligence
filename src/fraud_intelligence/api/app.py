from __future__ import annotations

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from fraud_intelligence.api.service import (
    run_supervised_pipeline,
    run_unsupervised_pipeline,
)

from fraud_intelligence.features.engineering import build_features


app = FastAPI(
    title="AI Fraud Intelligence API",
    version="0.1.0",
    description="Fraud detection API using supervised and unsupervised models.",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Transaction(BaseModel):
    Amount: float = Field(ge=0)
    Time: float = Field(ge=0)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict/supervised")
def predict_supervised(transaction: Transaction) -> dict[str, float | str]:
    # Load the training dataset.
    train_df = pd.read_csv("data/processed/train.csv")

    # Train the supervised model.
    model, metrics = run_supervised_pipeline(train_df)

    # Create the transaction to predict.
    prediction_df = pd.DataFrame(
        [
            {
                "Amount": transaction.Amount,
                "Time": transaction.Time,
                "Class": 0,
            }
        ]
    )

    # Apply exactly the same feature engineering used during training.
    prediction_features = build_features(prediction_df).drop(
        columns=["Class"]
    )

    # Make prediction.
    prediction = model.predict(prediction_features)[0]

    # Get fraud probability.
    fraud_probability = float(
        model.predict_proba(prediction_features)[0, 1]
    )

    # Determine risk level.
    if fraud_probability >= 0.70:
        risk_level = "HIGH"
    elif fraud_probability >= 0.30:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "prediction": int(prediction),
        "fraud_probability": fraud_probability,
        "risk_level": risk_level,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "roc_auc": metrics.roc_auc,
        "pr_auc": metrics.pr_auc,
    }

@app.post("/predict/unsupervised")
def predict_unsupervised(
    transaction: Transaction,
) -> dict[str, int]:

    # Load the real training dataset.
    train_df = pd.read_csv("data/processed/train.csv")

    # Train the Isolation Forest model.
    model, _ = run_unsupervised_pipeline(train_df)

    # ---------------------------------------------------------
    # Create a new transaction using the same structure
    # as the training dataset.
    # ---------------------------------------------------------
    prediction_df = train_df.iloc[[0]].copy()

    prediction_df["Amount"] = transaction.Amount
    prediction_df["Time"] = transaction.Time

    # Class is not used by the unsupervised model, but keeping
    # it here allows build_features() to receive the same
    # dataset structure.
    prediction_df["Class"] = 0

    # Combine training data + new transaction.
    combined_df = pd.concat(
        [
            train_df,
            prediction_df,
        ],
        ignore_index=True,
    )

    # Apply the same feature engineering.
    features = build_features(combined_df)

    # Isolation Forest uses only transaction-level features.
    feature_columns = [
        "Amount",
        "Time",
        "Amount_log",
        "Time_hour",
    ]

    features = features[feature_columns]

    # Select only the newly created transaction.
    prediction_features = features.iloc[[-1]]

    # Isolation Forest:
    # -1 = anomaly
    #  1 = normal
    prediction = model.predict(prediction_features)[0]

    anomaly = 1 if prediction == -1 else 0

    return {
        "anomaly": anomaly,
    }