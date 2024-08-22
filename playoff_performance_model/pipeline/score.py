import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import plot_importance

# Get path of current file's directory
DIRNAME = os.path.dirname(os.path.realpath(__file__))


def score_players(pipeline_folder_path: str, model_version: str) -> pd.DataFrame:
    # Read in player feature data
    player_data = pd.read_pickle(os.path.join(pipeline_folder_path, "feature_data.pkl"))
    print(player_data)


if __name__ == "__main__":
    # Test pipeline results
    batch_name = ""
    pipeline_folder_path = os.path.join(DIRNAME, "../pipeline_results", batch_name)

    # Score contracts in pipeline
    score_players(pipeline_folder_path, model_version="version_1")
