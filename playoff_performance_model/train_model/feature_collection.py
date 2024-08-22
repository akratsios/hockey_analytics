import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

# Get path of current file's directory
DIRNAME = os.path.dirname(os.path.realpath(__file__))

# Insert path to all hockey analytics files
sys.path.insert(0, os.path.join(DIRNAME, "../.."))
from common_functions.moneypuck_player_stats import (
    check_season_type_valid_mp,
    get_mp_player_data_for_year,
)


def add_offset_x_season_stats(
    data: pd.DataFrame, season_type: str, offset: int
) -> pd.DataFrame:
    # make sure season type valid
    check_season_type_valid_mp(season_type)

    years = data["action_season"].unique()
    print(years)

    # For each action season, get corresponding data for each offset season
    all_offset_data = []
    for year in years:
        player_ids = data.loc[data["action_season"] == year, "playerId"].values
        offset_data = get_mp_player_data_for_year(
            year + offset, season_type, situation="all", player_ids=player_ids
        )
        all_offset_data.append(offset_data)
    all_offset_data = pd.concat(all_offset_data)
    print(all_offset_data)

    # Subtract offset to season (reverse of collecting data) to merge it with original data
    all_offset_data["action_season"] = all_offset_data["season"] - offset

    # TODO: Change col names to indicate offset

    # Merge data with original data
    data = pd.merge(data, all_offset_data, on=["playerId", "action_season"], how="left")

    print(data)
    return data


def remove_empty_rows(data: pd.DataFrame) -> pd.DataFrame:
    # TODO: Remove all rows that do not have enough data
    return data


def feature_selection_process(feature_data: pd.DataFrame) -> pd.DataFrame:

    # Keep identifying information
    identifier_cols = ["playerId", "action_season", "action_date"]

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
    nunique = nunique.drop(identifier_cols)
    feature_data = feature_data.drop(nunique[nunique == 1].index, axis=1)

    # TODO: Categorical data, use label encoding
    cat_cols = ["team", "position"]
    feature_data.loc[:, cat_cols] = feature_data.loc[:, cat_cols].astype("category")
    print(feature_data.dtypes)
    # TODO: TEMPORARY: DROP CATEGORICAL DATA
    feature_data = feature_data.drop(columns=cat_cols)

    # TODO: Remove selected columns
    cols_to_remove = ["season", "name"]
    feature_data = feature_data.drop(columns=cols_to_remove, errors="ignore")

    return feature_data


def get_model_features(feature_data: pd.DataFrame) -> pd.DataFrame:
    # Add regular season offset data
    feature_data = add_offset_x_season_stats(feature_data, "regular", -1)

    # TODO: Larger lookback window than 1 year
    # TODO: Incorporate playoff data
    # TODO: Summary stats data (trends)

    # Feature selection
    feature_data = feature_selection_process(feature_data)

    return feature_data


def collect_training_features() -> pd.DataFrame:
    # Read in target variable data
    target_data = pd.read_csv(
        os.path.join(DIRNAME, "training_data", "target_variable.csv"), index_col=0
    )

    # Select only necessary columns from target variable data
    feature_data = target_data.loc[
        :, ["playerId", "action_season", "action_date", "gamescore_toi"]
    ].copy()
    print(feature_data)

    # Collect all features for model
    feature_data = get_model_features(feature_data)

    return feature_data


if __name__ == "__main__":
    # Collect all features for model training
    feature_data = collect_training_features()
    print(feature_data)
    # Save training data with features
    feature_data.to_csv(
        os.path.join(DIRNAME, "training_data", "training_feature_data.csv")
    )
    feature_data.to_pickle(
        os.path.join(DIRNAME, "training_data", "training_feature_data.pkl")
    )
