import os
import sys
from pathlib import Path

# Get path of current file's directory
dirname = os.path.dirname(os.path.realpath(__file__))


sys.path.append(dirname)
from collect_data.read_local_data import read_mp_bio_data
from common_data_pull import download_csv


class TypeException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


def read_in_season_players_stats(season: int, type: str):
    """Pulls the skater data for a given season from moneypuck and stores it.

    Args:
        season (int): Season to pull data for. 2023-2024 is 2023.
        type (str): Season type. Only accepts regular or playoffs.

    Raises:
        TypeException: Parameter type must be regular or playoffs.
    """
    if type not in ["regular", "playoffs"]:
        raise TypeException(
            f"Stats type must be in 'regular' or 'playoffs', '{type}' given."
        )
    # Moneypuck player data url
    file_url = f"https://www.moneypuck.com/moneypuck/playerData/seasonSummary/{season}/{type}/skaters.csv"

    # Location to save data
    folder_path = os.path.join(dirname, "../data", "moneypuck", "player_data")
    # Create folder if doesn't exist
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    # Final filepath of csv
    filepath = os.path.join(
        folder_path,
        f"skater_stats_{season}_{type}.csv",
    )
    download_csv(file_url, filepath)


def read_in_all_seasons_players_stats():
    # Moneypuck has data from 2008 to 2024
    seasons = range(2008, 2024)
    # For each season pull regular season and playoff data
    for season in seasons:
        read_in_season_players_stats(season, "regular")
        read_in_season_players_stats(season, "playoffs")


def get_player_name(player_id: int) -> str:
    # read in MoneyPuck player bio data
    mp_bio_data = read_mp_bio_data()
    # Get palyers name
    name_match = mp_bio_data.loc[mp_bio_data["playerId"] == player_id, "name"]
    return name_match.values[0]


if __name__ == "__main__":
    read_in_all_seasons_players_stats()
