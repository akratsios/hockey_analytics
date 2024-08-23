import os
import sys
from datetime import datetime
from functools import reduce

import numpy as np
import pandas as pd

# Get path of current file's directory
DIRNAME = os.path.dirname(os.path.realpath(__file__))

# Insert path to all hockey analytics files
sys.path.insert(0, os.path.join(DIRNAME, "../.."))
from collect_data.read_local_data import read_in_all_mp_data, read_mp_bio_data
from common_functions.moneypuck_player_stats import (
    check_season_type_valid_mp,
    get_mp_player_data_for_year,
)


def get_past_x_seasons_average_score(
    data: pd.DataFrame,
    season_type: str,
    columns: list,
    num_seasons: int,
    situation: str,
) -> pd.DataFrame:
    # make sure season type valid
    check_season_type_valid_mp(season_type)

    # Read in all moneypuck player data
    mp_data = read_in_all_mp_data(season_type)

    # Filter for situation
    mp_data = mp_data.loc[mp_data["situation"] == situation]

    # Filter columns
    mp_data = mp_data.loc[:, ["playerId", "season", *columns]]

    # Filter out unncessary players
    mp_data = mp_data.loc[mp_data["playerId"].isin(data["playerId"].unique())]
    mp_data = mp_data.reset_index(drop=True)

    # new column names
    col_mean_names = [
        f"mean_{num_seasons}years_{season_type}_{situation}_{col}" for col in columns
    ]
    print("here")
    data[col_mean_names] = data.apply(
        lambda x: get_past_x_seasons_average_score_helper(
            x["playerId"], x["action_season"], num_seasons, mp_data.copy()
        ),
        axis=1,
        result_type="expand",
    )
    return data


def get_past_x_seasons_average_score_helper(
    player_id: int,
    start_season: int,
    num_seasons: int,
    all_mp_player_data: pd.DataFrame,
) -> pd.DataFrame:
    # Only look at player data for given player id and appropriate seasons
    player_data = all_mp_player_data.loc[
        (all_mp_player_data["playerId"] == player_id)
        & (all_mp_player_data["season"] < start_season)
        & (all_mp_player_data["season"] >= start_season - num_seasons)
    ]
    # Get mean for each relevent column
    return player_data.drop(columns=["playerId", "season"]).mean()


def add_player_biographical_data_mp(data: pd.DataFrame) -> pd.DataFrame:
    # Read in MoneyPuck player bio data
    mp_bio_data = read_mp_bio_data()

    # Columns we want from features
    mp_bio_data = mp_bio_data.loc[
        :,
        [
            "playerId",
            "birthDate",
            "weight",
            "height",
            "nationality",
            "shootsCatches",
            "primaryPosition",
        ],
    ]

    # Merge with data
    data = pd.merge(data, mp_bio_data, on="playerId", how="left")

    # Get age in days
    data["age_in_days"] = data.apply(
        lambda x: (
            get_age_in_days(x["birthDate"], x["action_date"])
            if pd.notna(x["birthDate"])
            else np.nan
        ),
        axis=1,
    )

    # Get height in inches
    data["height_inches"] = data["height"].apply(
        lambda x: get_height_in_inches_from_str(x) if pd.notna(x) else x
    )

    # Remove unessary columns
    data = data.drop(columns=["birthDate", "height"])

    return data


def get_height_in_inches_from_str(str_height: str) -> int:
    str_height = str_height.replace(" ", "")
    feet = int(str_height.split("'")[0])
    inch = int(str_height.split("'")[1].split('"')[0])
    return feet * 12 + inch


def get_age_in_days(start_date: str, end_date: str) -> int:
    sd = datetime.strptime(start_date, "%Y-%m-%d")
    ed = datetime.strptime(end_date, "%Y-%m-%d")
    return reduce(lambda x, y: (y - x).days, [sd, ed])


def add_offset_x_season_stats(
    data: pd.DataFrame, season_type: str, offset: int
) -> pd.DataFrame:
    # make sure season type valid
    check_season_type_valid_mp(season_type)

    years = data["action_season"].unique()

    # For each action season, get corresponding data for each offset season
    all_offset_data = []
    for year in years:
        player_ids = data.loc[data["action_season"] == year, "playerId"].values
        offset_data = get_mp_player_data_for_year(
            year + offset, season_type, situation="all", player_ids=player_ids
        )
        all_offset_data.append(offset_data)
    all_offset_data = pd.concat(all_offset_data)

    # Subtract offset to season (reverse of collecting data) to merge it with original data
    all_offset_data["action_season"] = all_offset_data["season"] - offset

    # TODO: Change col names to indicate offset

    # Merge data with original data
    data = pd.merge(data, all_offset_data, on=["playerId", "action_season"], how="left")

    return data


def feature_selection_process(feature_data: pd.DataFrame) -> pd.DataFrame:

    # Keep identifying information
    identifier_cols = ["playerId", "action_season", "action_date"]
    target_col = "gamescore_toi"
    other_cols = [
        c for c in feature_data.columns if c != target_col and c not in identifier_cols
    ]

    # Remove rows with all missing data
    # TODO: Impute data, don't just remove
    feature_data = feature_data.dropna(subset=other_cols, how="any").reset_index(
        drop=True
    )
    # Print null rates
    # print(feature_data[["penalityMinutes"]])
    # print(feature_data.isnull().mean().round(4).mul(100).sort_values(ascending=False))

    # Save identifier data
    identifier_data = feature_data.loc[:, identifier_cols]
    # If present save the target data
    if target_col in feature_data.columns:
        target_data = feature_data.loc[:, target_col]
    else:
        target_data = pd.Series()
    feature_data = feature_data.drop(
        columns=[*identifier_cols, target_col], errors="ignore"
    )

    # Drop empty columns (30% empty threshold)
    empty_threshold = (
        0.7  # 1 minus percentage empty to drop, so 60% empty is 0.4 for threshold
    )
    feature_data = feature_data.dropna(
        thresh=feature_data.shape[0] * empty_threshold, axis=1
    )

    # Drop all columns where all non empty data is the same value
    nunique = feature_data.nunique()
    feature_data = feature_data.drop(nunique[nunique == 1].index, axis=1)

    # TODO: Categorical data, use label encoding
    cat_cols = ["team", "position", "nationality", "shootsCatches", "primaryPosition"]
    feature_data.loc[:, cat_cols] = feature_data.loc[:, cat_cols].astype("category")
    # TODO: TEMPORARY: DROP CATEGORICAL DATA
    # print(feature_data.dtypes)
    feature_data = feature_data.select_dtypes(exclude=["object"])

    # Remove selected columns
    cols_to_remove = ["season", "name"]
    feature_data = feature_data.drop(columns=cols_to_remove, errors="ignore")

    # Add identifier cols back
    if not target_data.empty:
        final_data = pd.concat([identifier_data, target_data, feature_data], axis=1)
    else:
        final_data = pd.concat([identifier_data, feature_data], axis=1)

    return final_data


def get_model_features(feature_data: pd.DataFrame) -> pd.DataFrame:
    # Add regular season offset data
    # TODO: Larger lookback window than 1 year
    # TODO: Incorporate playoff data
    # TODO: Summary stats data (trends)
    feature_data = add_offset_x_season_stats(feature_data, "regular", -1)

    # Get mean scores for past regular and playoff seasons
    numeric_cols = feature_data.select_dtypes(include=np.number).columns.tolist()
    numeric_cols = [
        c
        for c in numeric_cols
        if c not in ["playerId", "action_season", "gamescore_toi", "season"]
    ]
    feature_data = get_past_x_seasons_average_score(
        feature_data, "regular", numeric_cols, num_seasons=5, situation="all"
    )
    feature_data = get_past_x_seasons_average_score(
        feature_data, "playoffs", numeric_cols, num_seasons=5, situation="all"
    )

    # Add player biographical data (height, weight, age, etc.)
    feature_data = add_player_biographical_data_mp(feature_data)

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

    # Collect all features for model
    feature_data = get_model_features(feature_data)

    return feature_data


if __name__ == "__main__":
    # Collect all features for model training
    feature_data = collect_training_features()

    # Save training data with features
    feature_data.to_csv(
        os.path.join(DIRNAME, "training_data", "training_feature_data.csv")
    )
    feature_data.to_pickle(
        os.path.join(DIRNAME, "training_data", "training_feature_data.pkl")
    )
