import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

# Get path of current file's directory
DIRNAME = os.path.dirname(os.path.realpath(__file__))


def remove_empty_rows(data: pd.DataFrame) -> pd.DataFrame:
    # TODO: Remove all rows that do not have enough data
    return data


def feature_selection_process(feature_data: pd.DataFrame) -> pd.DataFrame:

    # Drop empty columns (60% empty threshold)
    # TODO: Change this to a threshold?
    empty_threshold = (
        0.4  # 1 minus percentage empty to drop, so 60% empty is 0.1 for threshold
    )
    feature_data = feature_data.dropna(
        thresh=feature_data.shape[0] * empty_threshold, axis=1
    )

    # Drop all columns where all non empty data is the same value
    nunique = feature_data.nunique()
    feature_data = feature_data.drop(nunique[nunique == 1].index, axis=1)

    # TODO: Categorical data, use label encoding
    cat_cols = ["team", "position"]
    feature_data.loc[:, cat_cols] = feature_data.loc[:, cat_cols].astype("category")
    print(feature_data.dtypes)
    # TODO: TEMPORARY: DROP CATEGORICAL DATA
    feature_data = feature_data.drop(columns=cat_cols)

    # TODO: Remove selected columns
    cols_to_remove = ["season", "name"]
    feature_data = feature_data.drop(columns=cols_to_remove)

    return feature_data


if __name__ == "__main__":
    # Read in unfiltered feature data
    feature_data = pd.read_csv(
        os.path.join(DIRNAME, "training_data", "training_feature_data.csv"), index_col=0
    )
    final_data = feature_selection_process(feature_data)
    # Save to csv and pkl
    final_data.to_csv(os.path.join(DIRNAME, "training_data", "model_training_data.csv"))
    final_data.to_pickle(
        os.path.join(DIRNAME, "training_data", "model_training_data.pkl")
    )
