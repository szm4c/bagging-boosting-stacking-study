from __future__ import annotations
import joblib
from pathlib import Path
from bagging_boosting_stacking_study.constants import TRAINED_MODELS_PATH, DATASET_NAMES

_MODEL_NAMES = {"rf", "xgb", "stack"}


def _model_path(dataset_name: str, model_name: str) -> Path:
    """
    Build the full path where the model should live, normalising names
    to lower-snake case.
    """
    ds, mdl = dataset_name.lower(), model_name.lower()
    return Path(TRAINED_MODELS_PATH) / f"{ds}_{mdl}.joblib"


def load_model(dataset_name: str, model_name: str):
    """
    Load a trained model saved by `train_best_models.py`.

    Args:
        dataset_name: The canonical dataset key (e.g. "regression",
            "california_housing"). Case-insensitive; must exist in
            `DATASET_NAMES`.
        model_name: One of "rf", "xgb", "stack" (case-insensitive).

    Returns:
        The scikit-learn estimator (or Pipeline / StackingRegressor)
        ready for `predict` / `score` calls.
    """
    ds = dataset_name.lower()
    mdl = model_name.lower()

    if ds not in {d.lower() for d in DATASET_NAMES}:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. " f"Known: {', '.join(DATASET_NAMES)}"
        )

    if mdl not in _MODEL_NAMES:
        raise ValueError(
            f"Unknown model '{model_name}'. " f"Choose from {_MODEL_NAMES}"
        )

    file_path = _model_path(ds, mdl)
    if not file_path.exists():
        raise FileNotFoundError(
            f"Model file not found at '{file_path}'. "
            "Did you run `train_best_models.py` first?"
        )

    return joblib.load(file_path)
