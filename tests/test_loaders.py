import pytest
import pandas as pd
from src.data import loaders


# Config immutability ------------------------------------------------------------------
def test_generators_config_immutable():
    expected = {
        "regression": {
            "n_samples": 1000,
            "n_features": 10,
            "n_informative": 8,
            "noise": 3,
        },
        "friedman1": {"n_samples": 5000, "n_features": 20, "noise": 0.5},
        "friedman3": {"n_samples": 200, "noise": 0.3},
    }
    assert loaders.GENERATORS_CONFIG == expected


# Synthetic generators reproducibility -------------------------------------------------
def test_regression_reproducible():
    df1 = loaders._generate_regression_dataset()
    df2 = loaders._generate_regression_dataset()
    pd.testing.assert_frame_equal(df1, df2)


def test_friedman1_reproducible():
    df1 = loaders._generate_friedman1_dataset()
    df2 = loaders._generate_friedman1_dataset()
    pd.testing.assert_frame_equal(df1, df2)


def test_friedman3_reproducible():
    df1 = loaders._generate_friedman3_dataset()
    df2 = loaders._generate_friedman3_dataset()
    pd.testing.assert_frame_equal(df1, df2)


# Synthetic shape & seed‐stability -----------------------------------------------------
@pytest.mark.parametrize("name", ["regression", "friedman1"])
def test_synthetic_shape_matches_config(name):
    cfg = loaders.GENERATORS_CONFIG[name]
    # call the internal generator
    gen = getattr(loaders, f"_generate_{name}_dataset")
    df = gen()
    # rows should equal n_samples
    assert df.shape[0] == cfg["n_samples"]
    # cols should equal n_features + target
    assert df.shape[1] == cfg["n_features"] + 1


def test_friedman3_shape_and_seed():
    # friedman3 has no n_features key; by default make_friedman3 => 4 features
    df1 = loaders._generate_friedman3_dataset()
    df2 = loaders._generate_friedman3_dataset()
    # reproducible
    pd.testing.assert_frame_equal(df1, df2)
    # correct shape: 200 rows, 4 features + 1 target = 5 cols
    assert df1.shape == (loaders.GENERATORS_CONFIG["friedman3"]["n_samples"], 5)
