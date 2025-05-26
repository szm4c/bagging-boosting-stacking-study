from pathlib import Path

DATASET_NAMES = (
    "regression",
    "friedman1",
    "friedman3",
    "california_housing",
    "airfoil_self_noise",
    "energy_efficiency",
)

SEED = 333

ROOT_PATH = Path(__file__).resolve().parents[2]  # project-root
RAW_DATA_PATH = ROOT_PATH / "data" / "raw"
PROCESSED_DATA_PATH = ROOT_PATH / "data" / "processed"
TRAINED_MODELS_PATH = ROOT_PATH / "trained_models"
