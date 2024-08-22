import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

# Get path of current file's directory
DIRNAME = os.path.dirname(os.path.realpath(__file__))

# Insert path to all hockey analytics files
sys.path.insert(0, os.path.join(DIRNAME, "../.."))
from common_functions.moneypuck_player_stats import get_mp_player_data_for_year
from playoff_performance_model.train_model.feature_collection import get_model_features


def get_all_players_for_given_season_mp(season: int) -> pd.DataFrame:
    player_data = get_mp_player_data_for_year(season, "regular")

    # Only want playerId and season
    player_data = player_data.loc[:, ["playerId", "season"]]

    # Need unique playerId
    player_data = player_data.drop_duplicates(subset=["playerId"])
    player_data = player_data.reset_index(drop=True)

    return player_data


def get_season_from_action_date(action_date: str) -> int:
    # Read in season start and end dates
    nhl_start_end_dates = pd.read_csv(
        os.path.join(DIRNAME, "../../data", "nhl_season_start_end_dates.csv")
    )
    nhl_start_end_dates = nhl_start_end_dates.sort_values(
        by=["Finish (incl. playoffs)"], ascending=True
    )
    # Current season will not have ended yet
    nhl_start_end_dates = nhl_start_end_dates.loc[
        nhl_start_end_dates["Finish (incl. playoffs)"] > action_date
    ].reset_index(drop=True)

    # If no season date attached, then get from action date
    # using a mid-end June cutoff
    if nhl_start_end_dates.empty:
        dt = datetime.strptime(action_date, "%Y-%m-%d")
        if dt.month < 6 or (dt.month == 6 and dt.day < 25):
            return dt.year - 1
        else:
            return dt.year

    # Get appropriate season
    return nhl_start_end_dates.loc[0, "Season"]


def create_batch_data(pipeline_folder_path: str, action_date: str):
    # TODO: Get season from action date
    # *The 2023 season is the 2024 playoffs*
    season = get_season_from_action_date(action_date)

    # Read in players
    players = get_all_players_for_given_season_mp(season)
    # Change season to action_season for data collection
    players = players.rename(columns={"season": "action_season"})

    # Add action date to player data
    players.loc[:, "action_date"] = action_date

    # Get features for each player
    feature_data = get_model_features(players)

    # Save features to the batch
    feature_data.to_csv(os.path.join(pipeline_folder_path, "feature_data.csv"))
    feature_data.to_pickle(os.path.join(pipeline_folder_path, "feature_data.pkl"))


if __name__ == "__main__":
    # Test pipeline results
    batch_name = "batch_2023-06-28"
    pipeline_folder_path = os.path.join(DIRNAME, "../pipeline_results", batch_name)

    create_batch_data(pipeline_folder_path, "2023-06-28")
