import pandas as pd
from sklearn.datasets import (
    make_regression,
    make_friedman1,
    make_friedman3,
)

# Global Vars
from bagging_boosting_stacking_study.constants import (
    SEED,
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    DATASET_NAMES,
)

GENERATORS_CONFIG = {
    "regression": dict(n_samples=1000, n_features=10, n_informative=8, noise=3),
    "friedman1": dict(n_samples=5000, n_features=20, noise=0.5),
    "friedman3": dict(n_samples=200, noise=0.3),
}


def _assemble_df(
    X, y, feature_prefix: str = "feature", target_name: str = "target"
) -> pd.DataFrame:
    """Helper function for DataFrame assembly."""
    df = pd.DataFrame(X, columns=[f"{feature_prefix}_{i}" for i in range(X.shape[1])])
    df[target_name] = y
    return df


def _generate_regression_dataset() -> pd.DataFrame:
    """Generate a regression dataset using sklearn `make_regression`."""
    params = GENERATORS_CONFIG["regression"]
    X, y = make_regression(random_state=SEED, **params)
    return _assemble_df(X=X, y=y)


def _generate_friedman1_dataset() -> pd.DataFrame:
    """Generate a Friedman #1 regression dataset."""
    params = GENERATORS_CONFIG["friedman1"]
    X, y = make_friedman1(random_state=SEED, **params)
    return _assemble_df(X=X, y=y)


def _generate_friedman3_dataset() -> pd.DataFrame:
    """Generate a Friedman #3 regression dataset."""
    params = GENERATORS_CONFIG["friedman3"]
    X, y = make_friedman3(random_state=SEED, **params)
    return _assemble_df(X=X, y=y)


def _get_california_housing_dataset() -> pd.DataFrame:
    """Load the California housing CSV from raw data."""
    return pd.read_csv(RAW_DATA_PATH / "california_housing.csv")


def _get_airfoil_self_noise_dataset() -> pd.DataFrame:
    """Load the Airfoil Self-Noise CSV from raw data."""
    return pd.read_csv(RAW_DATA_PATH / "airfoil_self_noise.csv")


def _get_energy_efficiency_dataset() -> pd.DataFrame:
    """Load the Energy Efficiency CSV from raw data."""
    return pd.read_csv(RAW_DATA_PATH / "energy_efficiency.csv")


def load_dataset(dataset_name: str, raw: bool = True) -> pd.DataFrame:
    if not isinstance(dataset_name, str):
        raise TypeError(
            f"Argument `dataset_name` must be a string, got {type(dataset_name)}"
        )

    dataset_name = dataset_name.strip().lower()
    if dataset_name not in DATASET_NAMES:
        raise ValueError(
            f"{dataset_name} is not a valid dataset name. "
            f"Possible options are {DATASET_NAMES}"
        )

    if raw:
        # raw branch -- dispatch to the correct loader/generator
        loaders = {
            "regression": _generate_regression_dataset,
            "friedman1": _generate_friedman1_dataset,
            "friedman3": _generate_friedman3_dataset,
            "california_housing": _get_california_housing_dataset,
            "airfoil_self_noise": _get_airfoil_self_noise_dataset,
            "energy_efficiency": _get_energy_efficiency_dataset,
        }

        return loaders[dataset_name]()

    # processed branch
    csv_path = PROCESSED_DATA_PATH / f"{dataset_name}.csv"
    try:
        return pd.read_csv(csv_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Processed file not found at {csv_path}. "
            "Run `make-processed` first or set raw=True."
        ) from e
