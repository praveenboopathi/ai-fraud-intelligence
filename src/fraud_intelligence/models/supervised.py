from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

TARGET_COLUMN = "Class"


@dataclass(frozen=True)
class ClassificationMetrics:
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float


def prepare_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    x = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    return x, y


def train_logistic_regression(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> LogisticRegression:
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=random_state,
    )

    model.fit(x_train, y_train)

    return model


def evaluate_classifier(
    model: LogisticRegression,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> ClassificationMetrics:
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    return ClassificationMetrics(
        precision=precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        recall=recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        f1=f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        roc_auc=roc_auc_score(
            y_test,
            probabilities,
        ),
        pr_auc=average_precision_score(
            y_test,
            probabilities,
        ),
    )