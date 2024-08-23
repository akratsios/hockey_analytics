import json
import os
import sys
import urllib
from shutil import copyfileobj

import pandas as pd

# Get path of current file's directory
DIRNAME = os.path.dirname(os.path.realpath(__file__))

sys.path.insert(0, os.path.join(DIRNAME, ".."))
from collect_data.common_data_pull import download_json


def read_in_current_team_roster(team_abr: str, season: int) -> pd.DataFrame:
    roster_data = download_json(
        f"https://api-web.nhle.com/v1/roster/{team_abr}/{season}{season+1}"
    )

    # Create the roster with necessary data
    formatted_roster = []
    for key, value in roster_data.items():
        for player in value:
            player_dict = {}
            for p_k, p_v in player.items():
                if isinstance(p_v, dict):
                    player_dict[p_k] = p_v["default"]
                else:
                    player_dict[p_k] = p_v
            formatted_roster.append(player_dict)
    roster_df = pd.DataFrame(formatted_roster)

    # Add team
    roster_df["team"] = team_abr

    return roster_df


def read_in_all_team_names() -> pd.DataFrame:
    team_data = download_json(f"https://api.nhle.com/stats/rest/en/team")
    return pd.DataFrame(team_data["data"])


def get_all_nhl_rosters(season: int) -> pd.DataFrame:
    team_data = read_in_all_team_names()

    rosters = []
    # Read in each roster
    for team_abr in team_data["triCode"].unique():
        try:
            rosters.append(read_in_current_team_roster(team_abr, season))
        except urllib.error.HTTPError:
            pass
    rosters = pd.concat(rosters).reset_index(drop=True)

    return rosters


if __name__ == "__main__":
    tor_roster = read_in_current_team_roster("TOR", 2024)
    print(tor_roster)

    nhl_rosters = get_all_nhl_rosters(2024)
    print(nhl_rosters)
