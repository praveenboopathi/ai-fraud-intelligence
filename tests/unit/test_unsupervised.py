import numpy as np
import pandas as pd

from fraud_intelligence.models.unsupervised import (
    detect_anomalies,
    fit_isolation_forest,
)


def make_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "amount": [
                10.0,
                11.0,
                12.0,
                10.5,
                11.5,
                500.0,
            ],
            "frequency": [
                1.0,
                1.1,
                0.9,
                1.0,
                1.2,
                20.0,
            ],
        }
    )


def test_fit_isolation_forest_returns_model() -> None:
    df = make_dataset()

    model = fit_isolation_forest(df)

    assert model is not None
    assert hasattr(model, "predict")


def test_isolation_forest_predicts_anomalies() -> None:
    df = make_dataset()

    model = fit_isolation_forest(df)
    predictions = detect_anomalies(model, df)

    assert len(predictions) == len(df)
    assert set(predictions.unique()).issubset({0, 1})


def test_detect_anomalies_returns_expected_shape() -> None:
    df = make_dataset()

    model = fit_isolation_forest(df)
    predictions = detect_anomalies(model, df)

    assert isinstance(predictions, pd.Series)
    assert predictions.shape == (len(df),)
    assert predictions.name == "anomaly"


def test_isolation_forest_is_reproducible() -> None:
    df = make_dataset()

    model_1 = fit_isolation_forest(df)
    model_2 = fit_isolation_forest(df)

    predictions_1 = detect_anomalies(model_1, df)
    predictions_2 = detect_anomalies(model_2, df)

    assert np.array_equal(
        predictions_1.to_numpy(),
        predictions_2.to_numpy(),
    )
def test_isolation_forest_accepts_configuration() -> None:
    df = make_dataset()

    model = fit_isolation_forest(
        df,
        contamination=0.01,
        random_state=123,
    )

    assert model.contamination == 0.01
    assert model.random_state == 123