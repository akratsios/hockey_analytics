import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

# Get path of current file's directory
DIRNAME = os.path.dirname(os.path.realpath(__file__))

# Insert path to all hockey analytics files
sys.path.insert(0, os.path.join(DIRNAME, ".."))
from collect_data.read_local_data import read_in_all_mp_data


class InvalidSeasonType(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


def check_season_type_valid_mp(season_type: str) -> bool:
    """Indicates if season type is valid for MoneyPuck.
    Must be regular or playoffs.

    Args:
        season_type (str): Season type.

    Raises:
        InvalidSeasonType: Exception class for invalid season type.

    Returns:
        bool: True if valid season.
    """
    if season_type not in ["regular", "playoffs"]:
        raise InvalidSeasonType(
            f"Moneypuck Season type must be in [regular, playoffs]. {season_type} given."
        )
    else:
        return True


def get_mp_player_data_for_year(
    season: int,
    season_type: str,
    situation: str = None,
    player_ids: list = [],
) -> pd.DataFrame:
    # make sure season type valid
    check_season_type_valid_mp(season_type)

    # Read in all moneypuck player data
    mp_regular_data = read_in_all_mp_data(season_type)

    # Filter out player ids if present
    if len(player_ids) > 0:
        filtered_data = mp_regular_data.loc[
            mp_regular_data["playerId"].isin(player_ids)
        ]

    # Only keep data for the correct season
    filtered_data = mp_regular_data.loc[mp_regular_data["season"] == season]

    # If present also filter on situaton
    if situation != None:
        filtered_data = filtered_data.loc[filtered_data["situation"] == situation]

    return filtered_data


def get_players_games_played_before_season(
    season: int,
    season_type: str,
    player_ids: list = [],
) -> pd.DataFrame:
    # make sure season type valid
    check_season_type_valid_mp(season_type)

    # Read in all moneypuck player data
    mp_regular_data = read_in_all_mp_data(season_type)

    # Filter out player ids if present
    if len(player_ids) > 0:
        filtered_data = mp_regular_data.loc[
            mp_regular_data["playerId"].isin(player_ids)
        ]

    # Only keep data less than the season
    filtered_data = mp_regular_data.loc[mp_regular_data["season"] < season]

    # Only keep from "all" situations
    filtered_data = filtered_data.loc[filtered_data["situation"] == "all"]

    return filtered_data

    return 1


if __name__ == "__main__":
    # Test check valid season type
    check_season_type_valid_mp("regular")

    # Test MoneyPuck player data pull by season
    mp_player_data = get_mp_player_data_for_year(
        2008, "regular", situation="all", player_ids=[8445550, 8445735]
    )
    print(mp_player_data)
