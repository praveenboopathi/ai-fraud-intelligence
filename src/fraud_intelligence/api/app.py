from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from fraud_intelligence.api.service import (
    run_supervised_pipeline,
    run_unsupervised_pipeline,
)

app = FastAPI(
    title="AI Fraud Intelligence API",
    version="0.1.0",
    description="Fraud detection API using supervised and unsupervised models.",
)


class Transaction(BaseModel):
    Amount: float = Field(ge=0)
    Time: float = Field(ge=0)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict/supervised")
def predict_supervised(transaction: Transaction) -> dict[str, float]:
    import pandas as pd

    df = pd.DataFrame([transaction.model_dump()])
    model, metrics = run_supervised_pipeline(df)

    return {
        "prediction": float(model.predict(df)[0]),
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "roc_auc": metrics.roc_auc,
        "pr_auc": metrics.pr_auc,
    }


@app.post("/predict/unsupervised")
def predict_unsupervised(transaction: Transaction) -> dict[str, int]:
    import pandas as pd

    df = pd.DataFrame([transaction.model_dump()])
    _, predictions = run_unsupervised_pipeline(df)

    return {"anomaly": int(predictions.iloc[0])}