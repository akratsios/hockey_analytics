import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

# Get path of current file's directory
DIRNAME = os.path.dirname(os.path.realpath(__file__))


# Insert path to all hockey analytics files
sys.path.insert(0, os.path.join(DIRNAME, "../.."))
from collect_data.moneypuck_players import get_player_name
from collect_data.read_local_data import read_in_salary_data_puckpedia


def team_rankings_for_batch(data: pd.DataFrame):
    pass


def player_rankings_for_batch(data: pd.DataFrame):
    print(
        data.loc[:, ["playerId", "name", "prediction"]].sort_values(
            by=["prediction"], ascending=False
        )
    )


def add_salary_data(scored_data) -> pd.DataFrame:
    # read in salary data
    salary_data = read_in_salary_data_puckpedia()

    # modify name format
    salary_data["name"] = salary_data["player"].apply(lambda x: f"{x.split(",")[1].strip()} {x.split(",")[0].strip()}")

    # Select name and cap hit column
    salary_data = salary_data.loc[:, ["name", "current_cap_hit"]]

    print(salary_data)
    print(scored_data)

    exit()
    return scored_data


def batch_analysis(pipeline_folder_path: str):
    # Read in scored data
    scored_data = pd.read_csv(
        os.path.join(pipeline_folder_path, "scored_data.csv"), index_col=0
    )

    # Add player name
    scored_data["name"] = scored_data["playerId"].apply(lambda x: get_player_name(x))

    # Add salary data
    scored_data = add_salary_data(scored_data)

    # Get team rankings for batch
    team_rankings_for_batch(scored_data)

    # Get player rankings for batch
    print("\nPlayer Rankings for Batch:")
    player_rankings_for_batch(scored_data)


if __name__ == "__main__":
    # Test pipeline results
    batch_name = "batch_2023-06-28"
    pipeline_folder_path = os.path.join(DIRNAME, "../pipeline_results", batch_name)

    # Run batch analysis
    batch_analysis(pipeline_folder_path)
