import os

import pandas as pd

# Get path of current file's directory
DIRNAME = os.path.dirname(os.path.realpath(__file__))


def read_in_all_mp_data(season_type: str):
    mp_player_data_filepath = os.path.join(DIRNAME, "../data/moneypuck/player_data")
    mp_files = os.listdir(mp_player_data_filepath)

    if season_type in ["playoffs", "regular"]:
        mp_files = [f for f in mp_files if season_type in f]
    print(mp_files)


def create_target_variable_data():
    mp_playoff_data = read_in_all_mp_data("playoffs")
    print(mp_playoff_data)


if __name__ == "__main__":
    print(DIRNAME)
    create_target_variable_data()
