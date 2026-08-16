import pandas as pd

from fraud_intelligence.models.hybrid import (
    predict_hybrid,
    train_hybrid_models,
)


def make_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "V1": [0.1, 0.2, 10.0, 0.3, 0.4, 12.0],
            "V2": [0.2, 0.1, 10.0, 0.4, 0.3, 11.0],
            "Amount": [10.0, 20.0, 5000.0, 15.0, 25.0, 6000.0],
            "Class": [0, 0, 1, 0, 0, 1],
        }
    )


def test_train_hybrid_models_returns_models() -> None:
    df = make_dataset()

    supervised_model, unsupervised_model = train_hybrid_models(df)

    assert supervised_model is not None
    assert unsupervised_model is not None


def test_predict_hybrid_returns_expected_columns() -> None:
    df = make_dataset()

    supervised_model, unsupervised_model = train_hybrid_models(df)

    features = df.drop(columns=["Class"])

    predictions = predict_hybrid(
        supervised_model,
        unsupervised_model,
        features,
    )

    assert len(predictions) == len(df)

    assert {
        "fraud_probability",
        "supervised_prediction",
        "anomaly_prediction",
        "hybrid_prediction",
    }.issubset(predictions.columns)


def test_predict_hybrid_returns_binary_predictions() -> None:
    df = make_dataset()

    supervised_model, unsupervised_model = train_hybrid_models(df)

    features = df.drop(columns=["Class"])

    predictions = predict_hybrid(
        supervised_model,
        unsupervised_model,
        features,
    )

    assert set(
        predictions["supervised_prediction"].unique()
    ).issubset({0, 1})

    assert set(
        predictions["anomaly_prediction"].unique()
    ).issubset({0, 1})

    assert set(
        predictions["hybrid_prediction"].unique()
    ).issubset({0, 1})