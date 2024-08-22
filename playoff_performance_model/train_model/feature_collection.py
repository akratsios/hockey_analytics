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
            player_ids, year + offset, season_type, situation="all"
        )
        all_offset_data.append(offset_data)
    all_offset_data = pd.concat(all_offset_data)
    print(all_offset_data)

    # Subtract offset to season (reverse of collecting data) to merge it with original data
    all_offset_data["action_season"] = all_offset_data["season"] - offset

    # TODO: Change cols to indicate offset
    data = pd.merge(data, all_offset_data, on=["playerId", "action_season"], how="left")

    print(data)
    return data


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

    # Add regular season offset data
    feature_data = add_offset_x_season_stats(feature_data, "regular", -1)

    # TODO: Larger lookback window than 1 year
    # TODO: Incorporate playoff data
    # TODO: Summary stats data (trends)

    return feature_data


if __name__ == "__main__":
    # Collect all features for model training
    feature_data = collect_training_features()
    print(feature_data)
    # Save training data with features
    feature_data.to_csv(
        os.path.join(DIRNAME, "training_data", "training_feature_data.csv")
    )
