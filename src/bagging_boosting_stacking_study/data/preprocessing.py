import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
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
    """
    Generic preprocessing for the Airfoil Self-Noise dataset.

    This function implements the full standard pipeline:
      1. Copy & rename SSPL to target.
      2. Remove domain-based outliers outside physically plausible ranges.
      3. Winsorize numeric features at the 1st and 99th percentiles.
      4. Apply natural log transformation plus one (log1p) to frequency (f) and thickness (delta).
      5. Create derived features: f * delta, delta squared (delta^2), alpha squared (alpha^2).
      6. Address multicollinearity by applying PCA on alpha and delta to produce alpha_delta_pc1.
      7. One-hot encode chord length (c) for categorical modeling.

    Args:
        df_raw: Raw Airfoil Self-Noise dataframe as loaded from CSV.
            Must contain columns: 'f', 'alpha', 'c', 'U_infinity', 'delta', and 'SSPL'.

    Returns:
        A cleaned pandas DataFrame with the following applied:
          - 'SSPL' renamed to 'target'
          - Rows outside defined physical bounds removed
          - Numeric features winsorized
          - Log1p transforms on 'f' and 'delta'
          - Derived features 'f_delta', 'delta_squared', 'alpha_squared'
          - PCA-combined 'alpha_delta_pc1'
          - One-hot encoded 'c' columns
        Row order is preserved.
    """

    # 1. Copy & rename
    df = df_raw.copy().rename(columns={"SSPL": "target"})

    # 2. Domain-based outlier removal
    bounds = {
        "f": (100, 10000),
        "alpha": (-20, 20),
        "c": (0.01, 0.5),
        "U_infinity": (1, 100),
        "delta": (0.001, 0.1),
        "target": (20, 200),
    }
    for col, (low, high) in bounds.items():
        df = df[(df[col] >= low) & (df[col] <= high)]

    # 3. Winsorize extremes
    for col in ["f", "alpha", "c", "U_infinity", "delta", "target"]:
        low_q, high_q = df[col].quantile([0.01, 0.99])
        df[col] = df[col].clip(low_q, high_q)

    # 4. Log1p transforms
    df["f"] = np.log1p(df["f"])
    df["delta"] = np.log1p(df["delta"])

    # 5. Derived features
    df["f_delta"] = df["f"] * df["delta"]
    df["delta_squared"] = df["delta"] ** 2
    df["alpha_squared"] = df["alpha"] ** 2

    # 6. PCA on alpha + delta
    pca = PCA(n_components=1)
    df["alpha_delta_pc1"] = pca.fit_transform(df[["alpha", "delta"]])
    df = df.drop(columns=["alpha", "delta"])

    # 7. One-hot encode chord 'c'
    df = pd.get_dummies(df, columns=["c"], prefix="c")

    return df


def clean_california_housing(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Generic preprocessing for the California Housing dataset.

    This function implements the full standard pipeline:
      1. Copy the raw DataFrame.
      2. Log1p-transform heavily skewed features (Population, AveRooms, AveBedrms).
      3. Winsorize those same features at the 1st and 99th percentiles.
      4. (Optional) Add binary outlier-flags for the skewed features.
      5. Combine pairs of correlated features via PCA:
         - Latitude & Longitude → Geo1
         - AveRooms & AveBedrms → RoomsPC
      6. Create derived features:
         - MedInc x RoomsPC
         - MedInc squared (MedInc²)
         - HouseAge binned into 4 groups
         - MedInc quartile bands
         - HouseholdDensity = Population / AveOccup
      7. Unsupervised segmentation:
         - 2-component PCA on all numeric features (excluding target)
         - KMeans clustering (3 clusters) on the PCA embedding
         - One-hot encode the resulting RegionCluster labels
      8. Robust-scaling (median & IQR) on all numeric features except target.

    Args:
        df_raw: Raw California Housing DataFrame as fetched from sklearn.
            Must contain columns: 'MedInc', 'HouseAge', 'AveRooms',
            'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude', and 'target'.

    Returns:
        A cleaned pandas DataFrame with:
          - Skewed features log1p-transformed, winsorized, and flagged
          - PCA components 'Geo1' and 'RoomsPC' replacing their originals
          - Derived interaction, polynomial, bin, and density features
          - One-hot encoded region clusters
          - All numeric predictors robust-scaled
        Row order is preserved.
    """
    df = df_raw.copy()

    # 1. Outlier handling: log1p + winsorize + outlier flags
    skewed = ["Population", "AveRooms", "AveBedrms"]
    for col in skewed:
        df[col] = np.log1p(df[col])
        lower, upper = df[col].quantile([0.01, 0.99])
        df[f"{col}_outlier"] = ((df[col] < lower) | (df[col] > upper)).astype(int)
        df[col] = df[col].clip(lower, upper)

    # 2. Correlated feature reduction via PCA
    pca_geo = PCA(n_components=1)
    df["Geo1"] = pca_geo.fit_transform(df[["Latitude", "Longitude"]])
    pca_rooms = PCA(n_components=1)
    df["RoomsPC"] = pca_rooms.fit_transform(df[["AveRooms", "AveBedrms"]])
    df.drop(columns=["Latitude", "Longitude", "AveRooms", "AveBedrms"], inplace=True)

    # 3. Derived features: interactions, polynomial, binning, density
    df["MedInc_x_RoomsPC"] = df["MedInc"] * df["RoomsPC"]
    df["MedInc_sq"] = df["MedInc"] ** 2
    df["HouseAge_bin"] = pd.cut(df["HouseAge"], bins=[0, 10, 20, 40, 100], labels=False)
    df["MedInc_bin"] = pd.qcut(df["MedInc"], 4, labels=False)
    df["HouseholdDensity"] = df_raw["Population"] / df_raw["AveOccup"]

    # 4. Unsupervised segmentation: KMeans on PCA of all numeric features
    pca_full = PCA(n_components=2)
    numeric = df.select_dtypes(include=np.number).drop(columns=["target"])
    proj = pca_full.fit_transform(numeric)
    kmeans = KMeans(n_clusters=3)
    df["RegionCluster"] = kmeans.fit_predict(proj)
    df = pd.get_dummies(df, columns=["RegionCluster"], prefix="Region")

    # 5. Scaling: RobustScaler on all numeric features except target
    scaler = RobustScaler()
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    num_cols.remove("target")
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df


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
    df["Volume"] = df["Surface Area"] * df["Overall Height"]
    df["Surface_to_Volume_Ratio"] = df["Surface Area"] / df["Volume"]

    df["Wall_to_Roof_Ratio"] = df["Wall Area"] / df["Roof Area"]
    df["Height_to_Wall_Ratio"] = df["Overall Height"] / df["Wall Area"]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.median(), inplace=True)

    # 2. Interaction Features
    df["Glazing_x_Orientation"] = df["Glazing Area"] * df["Orientation"]
    df["Glazing_x_GlazingDist"] = df["Glazing Area"] * df["Glazing Area Distribution"]
    df["Surface_x_Compactness"] = df["Surface Area"] * df["Relative Compactness"]

    # 3. One-Hot Encoding of Categorical Variables
    encoder_adv = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    df_encoded_adv = encoder_adv.fit_transform(
        df[["Orientation", "Glazing Area Distribution"]]
    )
    encoded_feature_names_adv = encoder_adv.get_feature_names_out(
        ["Orientation", "Glazing Area Distribution"]
    )
    df = df.drop(columns=["Orientation", "Glazing Area Distribution"])
    df = pd.concat(
        [
            df.reset_index(drop=True),
            pd.DataFrame(df_encoded_adv, columns=encoded_feature_names_adv),
        ],
        axis=1,
    )

    # 4. Dropping columns
    features_to_drop = [
        "Cooling Load",
        "Surface Area",
        "Overall Height",
        "Orientation_3",
        "Orientation_5",
        "Orientation_2",
        "Orientation_4",
        "Glazing Area Distribution_3",
        "Glazing Area Distribution_5",
        "Glazing Area Distribution_1",
        "Glazing Area Distribution_4",
        "Glazing Area Distribution_2",
    ]
    df = df.drop(columns=features_to_drop, errors="ignore").sort_index(axis=1)

    # 5. Rename target column
    df = df.rename(columns={"Heating Load": "target"})

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
