import os
import sys
from pathlib import Path

# Get path of current file's directory
dirname = os.path.dirname(os.path.realpath(__file__))


sys.path.append(dirname)
from common_data_pull import download_csv


class TypeException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


def read_in_season_players_stats(season: int, type: str):
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


if __name__ == "__main__":
    read_in_season_players_stats(2023, "regular")
    read_in_season_players_stats(2023, "playoffs")
