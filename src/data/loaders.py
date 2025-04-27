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
SEED = 42

def _generate_regression_dataset() -> pd.DataFrame:
    """Generate a toy regression dataset using sklearn."""
    pass

def _generate_friedman1_dataset() -> pd.DataFrame:
    """Generate a Friedman #1 regression dataset."""
    pass

def _generate_friedman3_dataset() -> pd.DataFrame:
    """Generate a Friedman #3 regression dataset."""
    pass

def _get_california_housing_dataset() -> pd.DataFrame:
    """Load the California housing CSV from raw data."""
    return pd.load_csv()

def _get_airfloil_self_noise_dataset() -> pd.DataFrame:
    """Load the Airfoil Self-Noise CSV from raw data."""
    return pd.load_csv()

def _get_energy_efficiency_dataset() -> pd.DataFrame:
    """Load the Energy Efficiency CSV from raw data."""
    return pd.load_csv()

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
