import numpy as np
import pandas as pd
from bagging_boosting_stacking_study.constants import DATASET_NAMES


def clean_regression(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Drop the column "feature_9" if it exists.

    Args:
        df_raw: raw regression dataset as loaded from `data/raw`

    Returns:
        Cleaned dataframe with the same rows but no "feature_9" column.
    """
    return df_raw.drop(columns="feature_9", errors="ignore")


def clean_friedman1(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-process the Friedman-1 regression dataset.

    The routine adds three deterministic features retained from EDA and
    then removes columns that showed negligible (< 0.20) Pearson
    correlation with the target.

    **Added features**

    * ``feature_0*feature_1`` — first-order interaction term
    * ``feature_2**2`` — quadratic term of *feature 2*
    * ``feature_2**2-feature_2`` — centered quadratic term
      (``feature_2**2 - feature_2``)

    **Dropped features**

    * ``feature_2``
    * ``feature_5`` through ``feature_19`` (inclusive)

    Args:
        df_raw: Original Friedman-1 dataframe containing at least
            ``feature_0`` … ``feature_19`` and ``target``.

    Returns:
        A copy of *df_raw* with the engineered features appended and the
        low-signal columns removed; row order is preserved.

    """
    df = df_raw.copy()  # work on copy

    # Interaction and polynomial terms
    df["feature_0*feature_1"] = df["feature_0"] * df["feature_1"]
    df["feature_2**2"] = df["feature_2"] ** 2
    df["feature_2**2-feature_2"] = df["feature_2**2"] - df["feature_2"]

    # Drop features with correlation < 0.2
    features_to_drop = [f"feature_{i}" for i in range(5, 20)] + ["feature_2"]
    df = df.drop(columns=features_to_drop, errors="ignore").sort_index(axis=1)

    return df


def clean_friedman3(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Generic preprocessing for the Friedman #3 dataset.

    The routine leaves every original column untouched and **adds**
    deterministic, model-agnostic features:

    * ``feature_1*feature_2`` (interaction term)
    * ``feature_1**2`` (quadratic term)
    * ``feature_2**2`` (quadratic term)
    * ``feature_2_bin`` - three-level categorical copy of *feature 2*
    * one-hot dummies: ``feature_2_low``, ``feature_2_med``, ``feature_2_high``

    No scaling, splitting, missing-value handling, or target transforms are
    applied—those remain the responsibility of downstream model pipelines.

    Args:
        df_raw: Raw Friedman #3 dataframe as produced by
            ``load_dataset("friedman3")``. It must contain the columns
            ``feature_0`` … ``feature_4`` **and** ``target``.

    Returns:
        Copy of the input with the additional engineered columns
        appended; row order is preserved and no rows are dropped.

    """
    df = df_raw.copy()  # work on copy

    # 1. Interaction & polynomial terms
    df["feature_1*feature_2"] = df["feature_1"] * df["feature_2"]
    df["feature_1**2"] = df["feature_1"] ** 2
    df["feature_2**2"] = df["feature_2"] ** 2

    # 2. Equal-width bins for feature_2 → categorical + dummies
    n_bins = 3
    labels = ["low", "med", "high"]

    f2_min, f2_max = df["feature_2"].min(), df["feature_2"].max()
    edges = np.linspace(f2_min, f2_max, num=n_bins + 1)

    df["feature_2_bin"] = pd.cut(
        df["feature_2"],
        bins=edges,
        labels=labels,
        include_lowest=True,
        # right=False,
    )

    dummies = pd.get_dummies(df["feature_2_bin"], prefix="feature_2")
    df = pd.concat([df, dummies], axis=1).sort_index(axis=1)

    return df


def clean_airfoil_self_noise(df_raw: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError("`clean_airfoil_self_noise` function is not impemented")


def clean_california_housing(df_raw: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError("`clean_california_housing` function is not impemented")


def clean_energy_efficiency(df_raw: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError("`clean_energy_efficiency` function is not impemented")


def preprocess_dataset(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    if not isinstance(dataset_name, str):
        raise TypeError(f"`dataset_name` must be a string, got {type(dataset_name)}")
    else:
        dataset_name = dataset_name.strip().lower()
        if dataset_name not in DATASET_NAMES:
            raise ValueError(
                f"{dataset_name} is not a valid dataset name. Possible options are "
                f"{DATASET_NAMES}"
            )

    preprocessing_pipelines = {
        "regression": clean_regression,
        "friedman1": clean_friedman1,
        "friedman3": clean_friedman3,
        "airfoil_self_noise": clean_airfoil_self_noise,
        "california_housing": clean_california_housing,
        "energy_efficiency": clean_energy_efficiency,
    }

    return preprocessing_pipelines[dataset_name](df_raw=df)
