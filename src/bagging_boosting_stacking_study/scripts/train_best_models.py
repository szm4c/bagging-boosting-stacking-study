import os
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
from joblib import parallel_backend

from bagging_boosting_stacking_study.data.loaders import load_dataset
from bagging_boosting_stacking_study.configs import load_params
from bagging_boosting_stacking_study.constants import (
    SEED,
    DATASET_NAMES,
    TRAINED_MODELS_PATH,
)


# helpers
def make_preprocessor(prep_choice: Optional[str], df) -> Optional[ColumnTransformer]:
    """
    Returns a ColumnTransformer or None depending on prep_choice.
    Currently only needed for friedman3.
    """
    if prep_choice is None:
        return None

    if prep_choice == "drop_all_cat":
        numeric_cols = [
            c
            for c in df.columns
            if c
            not in {
                "target",
                "feature_2_bin",
                "feature_2_low",
                "feature_2_med",
                "feature_2_high",
            }
        ]
        return ColumnTransformer(
            [("num", "passthrough", numeric_cols)], remainder="drop"
        )

    # encode_bin
    cat_col = ["feature_2_bin"]
    numeric_cols = [
        c
        for c in df.columns
        if c
        not in {
            "target",
            "feature_2_low",
            "feature_2_med",
            "feature_2_high",
            "feature_2_bin",
        }
    ]
    ord_enc = OrdinalEncoder(categories=[["low", "med", "high"]], dtype=float)
    return ColumnTransformer(
        [("ord", ord_enc, cat_col), ("num", "passthrough", numeric_cols)],
        remainder="drop",
    )


def _clean_rf_params(p: Dict[str, Any]) -> Dict[str, Any]:
    p = p.copy()
    cap_flag = p.pop("cap_depth", None)
    if cap_flag is not None:
        p["max_depth"] = p.pop("max_depth", None) if cap_flag else None

    if not p.get("bootstrap", False):
        p["max_samples"] = None
        p["oob_score"] = False

    p.update(dict(random_state=SEED, n_jobs=-1))
    # Remove stray keys if any
    allowed = RandomForestRegressor().get_params().keys()
    return {k: v for k, v in p.items() if k in allowed}


def _clean_xgb_params(p: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Returns cleaned dict and optional prep_choice.
    """
    p = p.copy()
    prep_choice = p.pop("prep_choice", None)

    if p["grow_policy"] == "depthwise":
        p.pop("max_leaves", None)
    else:
        p["max_depth"] = 0

    p.update(
        dict(
            objective="reg:squarederror",
            tree_method="hist",
            random_state=SEED,
            n_jobs=-1,
        )
    )
    allowed = XGBRegressor().get_params().keys()
    return {k: v for k, v in p.items() if k in allowed}, prep_choice


# builders
def build_rf(train_df, params):
    p = _clean_rf_params(params)
    prep = make_preprocessor(params.get("prep_choice"), train_df)
    rf = RandomForestRegressor(**p)
    return Pipeline([("prep", prep), ("rf", rf)]) if prep else rf


def build_xgb(train_df, params):
    p, prep_choice = _clean_xgb_params(params)
    prep = make_preprocessor(prep_choice, train_df)
    xgb = XGBRegressor(**p)
    return Pipeline([("prep", prep), ("xgb", xgb)]) if prep else xgb


def build_stack(train_df, dataset_name):
    # Prepare data for fiting base learners (I'm using cv="prefit" for speed)
    X = train_df.drop(columns="target")
    y = train_df["target"].values

    # Random-Forest base learner
    rf_params = load_params(dataset=dataset_name, model="rf")["params"]
    rf_pipeline = build_rf(train_df, rf_params)
    rf_pipeline.fit(X, y)

    # XGBoost base learner
    xgb_raw = load_params(dataset=dataset_name, model="xgb")["params"]
    xgb_pipeline = build_xgb(train_df, xgb_raw)
    xgb_pipeline.fit(X, y)

    # OLS base learner (with matching pre-processing)
    _, prep_choice = _clean_xgb_params(xgb_raw)  # just to reuse prep_choice
    ols_preproc = make_preprocessor(prep_choice, train_df)  # None for all-numeric
    ols_pipeline = Pipeline(
        steps=([("prep", ols_preproc)] if ols_preproc else [])
        + [("sc", StandardScaler()), ("ols", LinearRegression(n_jobs=-1))]
    )
    ols_pipeline.fit(X, y)  # single fit

    # Meta-learner (Ridge on OOF-like preds)
    best_alpha = load_params(dataset=dataset_name, model="stack")["params"]["alpha"]

    meta_final = Pipeline(
        steps=[
            ("sc", StandardScaler()),
            (
                "ridge",
                Ridge(
                    alpha=best_alpha,
                    solver="sag",
                    max_iter=10_000,
                    random_state=SEED,
                ),
            ),
        ]
    )

    # StackingRegressor with cv="prefit"
    base_estimators_prefit = [
        ("rf", rf_pipeline),
        ("xgb", xgb_pipeline),
        ("ols", ols_pipeline),
    ]

    return StackingRegressor(
        estimators=base_estimators_prefit,
        final_estimator=meta_final,
        cv="prefit",  # no internal CV, base models are already fitted
        passthrough=False,
        n_jobs=-1,
    )


def build_model(model_name: str, train_df, dataset_name: str):
    cfg = (
        load_params(dataset=dataset_name, model=model_name)["params"]
        if model_name != "stack"
        else None
    )

    if model_name == "rf":
        return build_rf(train_df, cfg)
    if model_name == "xgb":
        return build_xgb(train_df, cfg)
    if model_name == "stack":
        return build_stack(train_df, dataset_name)
    raise ValueError(f"Unknown model: {model_name}")


# main loop
MODELS = ["rf", "xgb", "stack"]


def main() -> None:
    Path(TRAINED_MODELS_PATH).mkdir(parents=True, exist_ok=True)

    for dataset_name in DATASET_NAMES:
        print(f"\nLoading {dataset_name} dataset...")
        df = load_dataset(dataset_name=dataset_name, raw=False)

        train_df, _ = train_test_split(df, test_size=0.1, random_state=SEED)

        X = train_df.drop(columns="target")
        y = train_df["target"].values

        for model_name in MODELS:
            print(f"Training {model_name} …", end=" ", flush=True)
            try:
                model = build_model(model_name, train_df, dataset_name)
            except FileNotFoundError as e:
                print("no config file found. Skipping.")
                continue

            with parallel_backend("loky"):
                model.fit(X, y)

            out_file = Path(TRAINED_MODELS_PATH) / f"{dataset_name}_{model_name}.joblib"
            joblib.dump(model, out_file)
            print(f"done. saved to {out_file.name}")
