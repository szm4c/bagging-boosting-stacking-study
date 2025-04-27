import pathlib
import pandas as pd
from sklearn.datasets import (
    make_regression,
    make_friedman1,
    make_friedman3,
)

# Global Vars
ROOT_PATH = pathlib.Path(__file__).resolve().parents[2]      # project-root
RAW_DATA_PATH = ROOT_PATH / "data" / "raw"
SEED = 333

def _generate_regression_dataset() -> pd.DataFrame:
    """Generate a toy regression dataset using sklearn."""
    X, y = make_regression(n_samples = 1000, n_features = 20, n_informative = 5, noise = 30, random_state = SEED)

    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y_df = pd.DataFrame(y, columns=['traget'])

    df = pd.concat([X_df, y_df], axis=1)

    return df

def _generate_friedman1_dataset() -> pd.DataFrame:
    """Generate a Friedman #1 regression dataset."""
    X, y = make_friedman1(n_samples = 500, noise = 0.5, random_state = SEED)

    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y_df = pd.DataFrame(y, columns=['traget'])

    df = pd.concat([X_df, y_df], axis=1)

    return df

def _generate_friedman3_dataset() -> pd.DataFrame:
    """Generate a Friedman #3 regression dataset."""
    X, y = make_friedman3(n_samples = 2000, noise = 1.0, random_state = SEED)

    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y_df = pd.DataFrame(y, columns=['traget'])

    df = pd.concat([X_df, y_df], axis=1)

    return df

def _get_california_housing_dataset() -> pd.DataFrame:
    """Load the California housing CSV from raw data."""
    return pd.read_csv(RAW_DATA_PATH / 'california_housing.csv')

def _get_airfloil_self_noise_dataset() -> pd.DataFrame:
    """Load the Airfoil Self-Noise CSV from raw data."""
    return pd.read_csv(RAW_DATA_PATH / 'airfoil_self_noise.csv')

def _get_energy_efficiency_dataset() -> pd.DataFrame:
    """Load the Energy Efficiency CSV from raw data."""
    return pd.read_csv(RAW_DATA_PATH / 'energy_efficiency.csv')

def load_dataset(dataset_name: str) -> pd.DataFrame:
    if not isinstance(dataset_name, str):
        raise TypeError(
            f"Argument `dataset_name` must be a string, got {type(dataset_name)}"
        )

    possible_dataset_names = [
        "regression",
        "friedman1",
        "friedman3",
        "california_housing",
        "airfoil_self_noise",
        "energy_efficiency",
    ]
    dataset_name = dataset_name.strip().lower()
    if dataset_name not in possible_dataset_names:
        raise ValueError(
            f"{dataset_name} is not a valid dataset name. "
            f"Possible options are {possible_dataset_names}"
        )

    # dispatch to the correct loader/generator
    loaders = {
        "regression": _generate_regression_dataset,
        "friedman1": _generate_friedman1_dataset,
        "friedman3": _generate_friedman3_dataset,
        "california_housing": _get_california_housing_dataset,
        "airfoil_self_noise": _get_airfloil_self_noise_dataset,
        "energy_efficiency": _get_energy_efficiency_dataset,
    }

    return loaders[dataset_name]()