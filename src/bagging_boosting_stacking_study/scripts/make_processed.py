from bagging_boosting_stacking_study.data.loaders import load_dataset
from bagging_boosting_stacking_study.data.preprocessing import preprocess_dataset
from bagging_boosting_stacking_study.constants import DATASET_NAMES, PROCESSED_DATA_PATH


def main() -> None:
    for dataset in DATASET_NAMES:
        print(f"Loading {dataset} dataset.")
        df_raw = load_dataset(dataset)
        print(f"Preprocessing {dataset} dataset.")
        try:
            df_clean = preprocess_dataset(df=df_raw, dataset_name=dataset)
        except NotImplementedError as e:
            print(
                f"Preprocessing function is not impemented yet. Skipping {dataset} "
                "dataset."
            )
            continue
        print(f"Saving {dataset} dataset.")
        dataset_path = PROCESSED_DATA_PATH / f"{dataset}.csv"
        df_clean.to_csv(dataset_path, index=False)
