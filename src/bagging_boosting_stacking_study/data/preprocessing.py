import numpy as np
import pandas as pd
from bagging_boosting_stacking_study.constants import DATASET_NAMES
from sklearn.preprocessing import OneHotEncoder


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
    raise NotImplementedError("`clean_friedman1` function is not impemented")


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
    """Performs feature engineering and selection for the Energy Efficiency dataset.

    This routine performs the following deterministic transformations and feature selections:

    * **Engineered Features (derived from architectural characteristics):**
        * `Volume` (Surface Area * Overall Height)
        * `Surface_to_Volume_Ratio` (Surface Area / Volume)
        * `Wall_to_Roof_Ratio` (Wall Area / Roof Area)
        * `Height_to_Wall_Ratio` (Overall Height / Wall Area)
    * **Interaction Features:**
        * `Glazing_x_Orientation` (Glazing Area * Orientation)
        * `Glazing_x_GlazingDist` (Glazing Area * Glazing Area Distribution)
        * `Surface_x_Compactness` (Surface Area * Relative Compactness)
    * **One-Hot Encoding:**
        * Converts `Orientation` and `Glazing Area Distribution` into binary (one-hot encoded) features.
    * **Column Dropping:**
        * Removes the `Cooling Load` target variable (`Heating Load` is a chosen target).
        * Drops original highly correlated features (`Surface Area`, `Overall Height`) that are superseded by engineered features.
        * Removes specific low-importance one-hot encoded and interaction features based on prior analysis (e.g., specific orientation dummies, low-impact interaction terms).

    Args:
        df_raw: Raw Energy Efficiency dataframe containing architectural characteristics
            (e.g., 'Relative Compactness', 'Surface Area', 'Wall Area', 'Roof Area',
            'Overall Height', 'Orientation', 'Glazing Area', 'Glazing Area Distribution')
            and target variables ('Heating Load', 'Cooling Load').

    Returns:
        Copy of the input dataframe with new engineered features, one-hot encoded
        categorical variables, and selected low-importance/redundant columns removed.
        Row order is preserved.
    """

    df = df_raw.copy()

    # 1. Engineered Features
    df['Volume'] = df['Surface Area'] * df['Overall Height']
    df['Surface_to_Volume_Ratio'] = df['Surface Area'] / df['Volume']

    df['Wall_to_Roof_Ratio'] = df['Wall Area'] / df['Roof Area']
    df['Height_to_Wall_Ratio'] = df['Overall Height'] / df['Wall Area']
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.median(), inplace=True)

    # 2. Interaction Features
    df['Glazing_x_Orientation'] = df['Glazing Area'] * df['Orientation']
    df['Glazing_x_GlazingDist'] = df['Glazing Area'] * df['Glazing Area Distribution']
    df['Surface_x_Compactness'] = df['Surface Area'] * df['Relative Compactness']

    # 3. One-Hot Encoding of Categorical Variables
    encoder_adv = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    df_encoded_adv = encoder_adv.fit_transform(df[['Orientation', 'Glazing Area Distribution']])
    encoded_feature_names_adv = encoder_adv.get_feature_names_out(['Orientation', 'Glazing Area Distribution'])
    df = df.drop(columns=['Orientation', 'Glazing Area Distribution'])
    df = pd.concat([df.reset_index(drop=True), pd.DataFrame(df_encoded_adv, columns=encoded_feature_names_adv)], axis=1)

    # 4. Dropping columns
    features_to_drop = ['Cooling Load', 'Surface Area', 'Overall Height', 'Orientation_3', 'Orientation_5', 'Orientation_2',
                        'Orientation_4', 'Glazing Area Distribution_3', 'Glazing Area Distribution_5', 'Glazing Area Distribution_1', 
                        'Glazing Area Distribution_4', 'Glazing Area Distribution_2']
    df = df.drop(columns=features_to_drop, errors='ignore').sort_index(axis=1)

    return df


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
