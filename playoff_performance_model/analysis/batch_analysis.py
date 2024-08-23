import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz, process

# Get path of current file's directory
DIRNAME = os.path.dirname(os.path.realpath(__file__))


# Insert path to all hockey analytics files
sys.path.insert(0, os.path.join(DIRNAME, "../.."))
from collect_data.moneypuck_players import get_player_name
from collect_data.read_local_data import read_in_salary_data_puckpedia


def team_rankings_for_batch(data: pd.DataFrame):
    pass


def player_rankings_for_batch(data: pd.DataFrame):
    # First by overall
    print("\nPlayer Rankings for Batch by predicted gs_toi:")
    gs_rank = data.loc[:, ["playerId", "name", "prediction"]].sort_values(
        by=["prediction"], ascending=False
    )
    print(gs_rank)

    # Ranking by game score per dollar of salary
    print("\nPlayer Rankings for Batch by pred game score per salary:")
    gs_cap_hit_rank = (
        data.loc[:, ["playerId", "name", "prediction", "pred_gs_toi_cap_hit"]]
        .sort_values(by=["pred_gs_toi_cap_hit"], ascending=False)
        .reset_index()
    )
    print(gs_cap_hit_rank.tail(20))


def fuzzy_match_pd_col(
    df_1: pd.DataFrame, df_2: pd.DataFrame, col: str
) -> pd.DataFrame:
    # empty lists for storing the matches
    # later
    mat1 = []
    mat2 = []
    p = []

    # converting dataframe column to
    # list of elements
    # to do fuzzy matching
    list1 = df_1[col].tolist()
    list2 = df_2[col].tolist()

    # taking the threshold as 80
    threshold = 80

    # iterating through list1 to extract
    # it's closest match from list2
    for i in list1:
        mat1.append(process.extractOne(i, list2, scorer=fuzz.ratio))
    df_1["matches"] = mat1

    # iterating through the closest matches
    # to filter out the maximum closest match
    for j in df_1["matches"]:
        if j[1] >= threshold:
            p.append(j[0])
        mat2.append(",".join(p))
        p = []

    # storing the resultant matches back to dframe1
    df_1["matches"] = mat2
    return df_1


def add_salary_data(scored_data) -> pd.DataFrame:
    # read in salary data
    salary_data = read_in_salary_data_puckpedia()

    # modify name format
    salary_data["name"] = salary_data["player"].apply(
        lambda x: x.split(",")[1].strip() + " " + x.split(",")[0].strip()
    )

    # Select name and cap hit column
    salary_data = salary_data.loc[:, ["name", "current_cap_hit"]]

    # Fuzzy match names
    scored_data = fuzzy_match_pd_col(scored_data, salary_data, "name")

    # Merge salary with data
    salary_data = salary_data.rename(columns={"name": "matches"})
    scored_data = pd.merge(scored_data, salary_data, on="matches", how="left")
    scored_data = scored_data.drop(columns=["matches"])

    # There are two names that are duplicated in the NHL: Sebastian Aho and Elias Pettersson
    duplicate_names = [
        {
            "name": "Elias Pettersson",
            "playerId": 8480012,
            "current_cap_hit": "$11,600,000",
        },
        {"name": "Elias Pettersson", "playerId": None, "current_cap_hit": "$838,333"},
        {"name": "Sebastian Aho", "playerId": 8478427, "current_cap_hit": "$9,750,000"},
        {"name": "Sebastian Aho", "playerId": 8480222, "current_cap_hit": "$775,000"},
    ]
    for entry in duplicate_names:
        mask = (
            (scored_data["name"] == entry["name"])
            & (scored_data["current_cap_hit"] == entry["current_cap_hit"])
            & (scored_data["playerId"] != entry["playerId"])
        )
        scored_data = scored_data[~mask]
    scored_data = scored_data.reset_index(drop=True)

    # Turn salary into a number
    scored_data["current_cap_hit"] = scored_data["current_cap_hit"].apply(
        lambda x: float(str(x).replace("$", "").replace(",", ""))
    )

    return scored_data


def batch_analysis(pipeline_folder_path: str):
    # Read in scored data
    scored_data = pd.read_csv(
        os.path.join(pipeline_folder_path, "scored_data.csv"), index_col=0
    )

    # Add player name
    scored_data["name"] = scored_data["playerId"].apply(lambda x: get_player_name(x))
    # If no MoneyPuck match to player name, then remove the row from analysis
    scored_data = scored_data[~scored_data["name"].isin(["", None])]

    # Add salary data
    scored_data = add_salary_data(scored_data)
    # Add game score per dollar of cap hit
    print(scored_data)
    scored_data["pred_gs_toi_cap_hit"] = (
        scored_data["prediction"].values / scored_data["current_cap_hit"].values
    )

    # Get team rankings for batch
    team_rankings_for_batch(scored_data)

    # Get player rankings for batch
    player_rankings_for_batch(scored_data)


if __name__ == "__main__":
    # Test pipeline results
    action_date = "2024-06-29"
    batch_name = f"batch_{action_date}"
    pipeline_folder_path = os.path.join(DIRNAME, "../pipeline_results", batch_name)

    # Run batch analysis
    batch_analysis(pipeline_folder_path)
