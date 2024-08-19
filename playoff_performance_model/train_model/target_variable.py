import os
import sys

import numpy as np
import pandas as pd
from datetime import datetime

# Get path of current file's directory
DIRNAME = os.path.dirname(os.path.realpath(__file__))

# Insert path to all hockey analytics files
sys.path.insert(0, os.path.join(DIRNAME, "../.."))
from collect_data.read_local_data import read_in_all_mp_data


def calc_game_score(
    goals: int,
    primary_assists: int,
    secondary_assists: int,
    shots_on_goal: int,
    blocked_shots: int,
    penalties_drawn: int,
    penalties_taken: int,
    faceoff_win: int,
    faceoff_loss: int,
    corsi_forward_5v5: float,
    corsi_against_5v5: float,
    goals_forward_5v5: int,
    goals_against_5v5: int,
):
    player_gs = (
        (0.75 * goals)
        + (0.7 * primary_assists)
        + (0.55 * secondary_assists)
        + (0.075 * shots_on_goal)
        + (0.05 * blocked_shots)
        + (0.15 * penalties_drawn)
        - (0.15 * penalties_taken)
        + (0.01 * faceoff_win)
        - (0.01 * faceoff_loss)
        + (0.05 * corsi_forward_5v5)
        - (0.05 * corsi_against_5v5)
        + (0.15 * goals_forward_5v5)
        - (0.15 * goals_against_5v5)
    )
    return player_gs


def calc_corsi_against(on_ice_sa_f: int, on_ice_sa_a: int) -> float:
    corsi_against = on_ice_sa_a / (on_ice_sa_f + on_ice_sa_a)
    return round(corsi_against, 2)


def create_gamescore_toi(gamescore: float, time_on_ice: float) -> float:
    """Create the game score per second of time on ice variable.
    It is scaled by 100k to give the score a range in the low hundreds.

    Args:
        gamescore (float): Game score from MoneyPuck.
        time_on_ice (float): Time on ice in seconds.

    Returns:
        float: Game score per second of ice time scaled.
    """
    # Gt game score per second of time on ice
    # Multiply gs_toi by 100k to make score more readable
    return np.round(gamescore / time_on_ice * 100000)


def create_action_date_target_data(playoff_season_year: int) -> datetime:
    """Create the action date for the target data. Use the last
    day of the season (including playoffs) for the action date.

    Args:
        playoff_season_year (int): Target variable season.

    Returns:
        datetime: Action date.
    """
    # End date of each NHL season
    season_end_dates = {
        2008: "2009-06-12",
        2009: "2010-06-09",
        2010: "2011-06-15",
        2011: "2012-06-11",
        2012: "2013-06-24",
        2013: "2014-06-13",
        2014: "2015-06-15",
        2015: "2016-06-12",
        2016: "2017-06-11",
        2017: "2018-06-07",
        2018: "2019-06-12",
        2019: "2020-09-28",
        2020: "2021-07-07",
        2021: "2022-06-26",
        2022: "2023-06-13",
        2023: "2024-06-24",
    }
    # Want the action date to be the end of prior season
    return datetime.strptime(season_end_dates[playoff_season_year - 1], "%Y-%m-%d")


def create_target_variable_data(
    start_year: int, min_games_played: int, situation: str
) -> pd.DataFrame:
    """Create the target variable data for model training.
    Uses MoneyPuck player playoff data.

    Args:
        start_year (int): First season to get data for.
        min_games_played (int): Minimum number of games played in season to be included.
        situation (str): In game situation (e.g. all, 5on5, etc.).

    Returns:
        pd.DataFrame: Target variable data.
    """
    # Read in all moneypuck player playoff data
    mp_playoff_data = read_in_all_mp_data("playoffs")

    # Set season type to playoffs
    mp_playoff_data["season_type"] = "playoffs"

    # Filter out for appropriate situation
    mp_playoff_data = mp_playoff_data.loc[mp_playoff_data["situation"] == situation]

    # Only use players who played more than min_games_played games
    mp_playoff_data = mp_playoff_data.loc[
        mp_playoff_data["games_played"] >= min_games_played
    ]

    # Only keep years start_year and later for training and testing model
    mp_playoff_data = mp_playoff_data.loc[mp_playoff_data["season"] >= start_year]

    # Action date is the end of the previous season
    mp_playoff_data["action_date"] = mp_playoff_data["season"].apply(
        lambda x: create_action_date_target_data(x)
    )

    # Get game score per second of time on ice
    mp_playoff_data["gamescore_toi"] = create_gamescore_toi(
        mp_playoff_data["gameScore"].values, mp_playoff_data["icetime"].values
    )

    mp_playoff_data = mp_playoff_data.reset_index(drop=True)

    # For target variable only need player id, season, corsi per second of ice time
    # Include player name for readability
    return mp_playoff_data.loc[
        :,
        [
            "playerId",
            "name",
            "season_type",
            "season",
            "situation",
            "action_date",
            "gamescore_toi",
        ],
    ]


if __name__ == "__main__":
    # Start year of 2015
    # Min games played of 4 (healthy for min 1 series)
    # Filter out data to use "all" situation rows
    # AKA we want the target variable to reflect all game situations
    target_variable_data = create_target_variable_data(
        start_year=2015, min_games_played=4, situation="all"
    )
    # Save target variable data
    target_variable_data.to_csv(
        os.path.join(DIRNAME, "training_data", "target_variable.csv")
    )
    print(max(target_variable_data["gamescore_toi"]))
    print(target_variable_data.sort_values(by="gamescore_toi", ascending=False))

    print(
        target_variable_data.groupby(["playerId", "name"])["gamescore_toi"]
        .mean()
        .sort_values(ascending=False)
        .reset_index(drop=False)
        .head(20)
    )
