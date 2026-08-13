import pandas as pd
import pytest

from fraud_intelligence.features.engineering import (
    add_transaction_features,
    build_features,
)


def test_feature_engineering_adds_expected_features():
    df = pd.DataFrame(
        {
            "Time": [0.0, 3600.0, 7200.0],
            "Amount": [0.0, 10.0, 100.0],
            "Class": [0, 0, 1],
        }
    )

    result = add_transaction_features(df)

    assert "Amount_log" in result.columns
    assert "Time_hour" in result.columns
    assert result.shape[0] == df.shape[0]


def test_amount_log_is_non_negative():
    df = pd.DataFrame(
        {
            "Amount": [0.0, 10.0, 100.0],
            "Class": [0, 0, 1],
        }
    )

    result = add_transaction_features(df)

    assert (result["Amount_log"] >= 0).all()


def test_time_hour_is_within_day_range():
    df = pd.DataFrame(
        {
            "Time": [0.0, 3600.0, 86399.0],
            "Amount": [10.0, 20.0, 30.0],
            "Class": [0, 0, 1],
        }
    )

    result = add_transaction_features(df)

    assert (result["Time_hour"] >= 0).all()
    assert (result["Time_hour"] < 24).all()


def test_build_features_requires_target_column():
    df = pd.DataFrame(
        {
            "Time": [0.0],
            "Amount": [10.0],
        }
    )

    with pytest.raises(ValueError, match="Missing target column"):
        build_features(df)