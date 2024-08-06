import os
import sys
from pathlib import Path

# Get path of current file's directory
dirname = os.path.dirname(os.path.realpath(__file__))


sys.path.append(dirname)
from common_data_pull import download_csv


def read_in_season_players_stats(season: int):
    # Moneypuck player data url
    file_url = f"https://www.moneypuck.com/moneypuck/playerData/seasonSummary/{season}/regular/skaters.csv"

    # Location to save data
    folder_path = os.path.join(dirname, "../data", "moneypuck", "player_data")
    # Create folder if doesn't exist
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    # Final filepath of csv
    filepath = os.path.join(
        folder_path,
        f"skater_stats_{season}.csv",
    )
    download_csv(file_url, filepath)


if __name__ == "__main__":
    # data = test(2023)
    # exit()
    data = read_in_season_players_stats(2023)
