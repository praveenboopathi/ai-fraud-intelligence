import numpy as np
import pandas as pd

from fraud_intelligence.models.supervised import (
    evaluate_classifier,
    prepare_features,
    train_logistic_regression,
)


def make_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_1": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            "feature_2": [1.0, 0.8, 0.6, 0.4, 0.2, 0.0],
            "Class": [0, 0, 0, 1, 1, 1],
        }
    )


def test_prepare_features_separates_target() -> None:
    df = make_dataset()

    features, target = prepare_features(df)

    assert "Class" not in features.columns
    assert len(features) == len(target)
    assert target.tolist() == [0, 0, 0, 1, 1, 1]


def test_prepare_features_requires_target() -> None:
    df = make_dataset().drop(columns=["Class"])

    try:
        prepare_features(df)
    except ValueError as exc:
        assert "Missing target column" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_logistic_regression_trains() -> None:
    df = make_dataset()
    features, target = prepare_features(df)

    model = train_logistic_regression(features, target)

    assert model.classes_.tolist() == [0, 1]
    assert model.coef_.shape == (1, 2)


def test_classifier_evaluation_returns_metrics() -> None:
    df = make_dataset()
    features, target = prepare_features(df)

    model = train_logistic_regression(features, target)
    metrics = evaluate_classifier(model, features, target)

    values = np.array(
        [
            metrics.precision,
            metrics.recall,
            metrics.f1,
            metrics.roc_auc,
            metrics.pr_auc,
        ]
    )

    assert np.isfinite(values).all()
    assert (values >= 0).all()
    assert (values <= 1).all()