import os
import sys

import pandas as pd

# Get path of current file's directory
DIRNAME = os.path.dirname(os.path.realpath(__file__))


def read_in_all_mp_data(season_type: str) -> pd.DataFrame:
    mp_player_data_filepath = os.path.join(DIRNAME, "../data/moneypuck/player_data")
    mp_files = os.listdir(mp_player_data_filepath)

    if season_type in ["playoffs", "regular"]:
        mp_files = [f for f in mp_files if season_type in f]

    all_data = []
    for file in mp_files:
        all_data.append(pd.read_csv(os.path.join(mp_player_data_filepath, file)))

    all_data = pd.concat(all_data)
    return all_data.sort_values(by=["season", "playerId"], ascending=True)


def read_mp_bio_data() -> pd.DataFrame:
    mp_data_filepath = os.path.join(DIRNAME, "../data/moneypuck")
    bio_data = pd.read_csv(os.path.join(mp_data_filepath, "player_biography_data.csv"))
    return bio_data


def read_in_salary_data_puckpedia() -> pd.DataFrame:
    mp_data_filepath = os.path.join(DIRNAME, "../data")
    salary_data = pd.read_csv(
        os.path.join(mp_data_filepath, "nhl_salaries_2024-2025.csv"), index_col=0
    )
    salary_data = salary_data.reset_index(drop=True)
    return salary_data
