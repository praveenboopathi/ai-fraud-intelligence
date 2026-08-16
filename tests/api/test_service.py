import pandas as pd

from fraud_intelligence.api.service import (
    run_supervised_pipeline,
    run_unsupervised_pipeline,
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


def test_run_supervised_pipeline_returns_model_and_metrics() -> None:
    model, metrics = run_supervised_pipeline(
        make_dataset(),
        random_state=123,
    )

    assert model is not None
    assert model.random_state == 123

    assert 0.0 <= metrics.precision <= 1.0
    assert 0.0 <= metrics.recall <= 1.0
    assert 0.0 <= metrics.f1 <= 1.0
    assert 0.0 <= metrics.roc_auc <= 1.0
    assert 0.0 <= metrics.pr_auc <= 1.0


def test_run_unsupervised_pipeline_returns_predictions() -> None:
    model, predictions = run_unsupervised_pipeline(
        make_dataset(),
        contamination=0.1,
        random_state=123,
    )

    assert model is not None
    assert model.contamination == 0.1
    assert model.random_state == 123

    assert isinstance(predictions, pd.Series)
    assert predictions.name == "anomaly"
    assert len(predictions) == len(make_dataset())
    assert set(predictions.unique()).issubset({0, 1})


def test_run_unsupervised_pipeline_is_reproducible() -> None:
    df = make_dataset()

    _, predictions_1 = run_unsupervised_pipeline(
        df,
        contamination=0.1,
        random_state=123,
    )

    _, predictions_2 = run_unsupervised_pipeline(
        df,
        contamination=0.1,
        random_state=123,
    )

    assert predictions_1.equals(predictions_2)