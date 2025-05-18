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
    raise NotImplementedError("`clean_friedman1` function is not impemented")


def clean_friedman3(df_raw: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError("`clean_friedman3` function is not impemented")


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
