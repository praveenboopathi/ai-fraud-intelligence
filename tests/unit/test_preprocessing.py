import pandas as pd
import pytest

from fraud_intelligence.data.preprocessing import fit_preprocessor


def make_sample_dataframe() -> pd.DataFrame:
    """Create a small dataset for preprocessing tests."""
    return pd.DataFrame(
        {
            "Time": [0.0, 10.0, 20.0, 30.0],
            **{f"V{i}": [0.0, 1.0, 2.0, 3.0] for i in range(1, 29)},
            "Amount": [10.0, 20.0, 30.0, 40.0],
            "Class": [0, 0, 1, 0],
        }
    )


def test_training_features_are_standardized() -> None:
    """Verify training features have mean zero and unit variance."""
    df = make_sample_dataframe()
    preprocessor = fit_preprocessor(df)

    transformed = preprocessor.transform(df)
    features = transformed.drop(columns="Class")

    assert features.mean().abs().max() < 1e-12
    assert (features.std(ddof=0) - 1).abs().max() < 1e-12


def test_target_column_is_preserved() -> None:
    """Verify the Class column is preserved without scaling."""
    df = make_sample_dataframe()
    preprocessor = fit_preprocessor(df)

    transformed = preprocessor.transform(df)

    assert transformed["Class"].tolist() == df["Class"].tolist()


def test_missing_target_column_is_rejected() -> None:
    """Verify a missing target column raises an error."""
    df = make_sample_dataframe().drop(columns="Class")

    with pytest.raises(ValueError, match="Target column"):
        fit_preprocessor(df)


def test_missing_feature_column_is_rejected() -> None:
    """Verify missing feature columns are detected during transformation."""
    df = make_sample_dataframe()
    preprocessor = fit_preprocessor(df)

    incomplete = df.drop(columns="V1")

    with pytest.raises(ValueError, match="Missing feature columns"):
        preprocessor.transform(incomplete)