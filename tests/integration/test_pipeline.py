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


def make_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Amount": [
                10.0,
                11.0,
                12.0,
                10.5,
                11.5,
                500.0,
                450.0,
                9.5,
                13.0,
                600.0,
            ],
            "Time": [
                3600.0,
                7200.0,
                10800.0,
                14400.0,
                18000.0,
                21600.0,
                25200.0,
                28800.0,
                32400.0,
                36000.0,
            ],
            "Class": [
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                0,
                0,
                1,
            ],
        }
    )


def test_supervised_pipeline_end_to_end() -> None:
    df = make_dataset()

    features = build_features(df)
    x, y = prepare_features(features)

    model = train_logistic_regression(
        x,
        y,
        random_state=123,
    )

    metrics = evaluate_classifier(
        model,
        x,
        y,
    )

    assert "Class" in features.columns
    assert "Amount_log" in features.columns
    assert "Time_hour" in features.columns

    assert model.random_state == 123

    assert 0.0 <= metrics.precision <= 1.0
    assert 0.0 <= metrics.recall <= 1.0
    assert 0.0 <= metrics.f1 <= 1.0
    assert 0.0 <= metrics.roc_auc <= 1.0
    assert 0.0 <= metrics.pr_auc <= 1.0


def test_unsupervised_pipeline_end_to_end() -> None:
    df = make_dataset()

    features = build_features(df)

    model = fit_isolation_forest(
        features,
        contamination=0.1,
        random_state=123,
    )

    predictions = detect_anomalies(
        model,
        features,
    )

    assert model.contamination == 0.1
    assert model.random_state == 123

    assert isinstance(predictions, pd.Series)
    assert predictions.name == "anomaly"
    assert len(predictions) == len(features)
    assert set(predictions.unique()).issubset({0, 1})