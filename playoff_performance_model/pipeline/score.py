import json
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
    X = pd.read_pickle(os.path.join(pipeline_folder_path, "feature_data.pkl"))

    # Save identifier data
    identifier_data = X.loc[:, ["playerId", "action_season", "action_date"]]
    identifier_data = identifier_data.reset_index()

    # Load in saved xgboost model
    model_path = os.path.join(DIRNAME, "../models", model_version)
    model = xgb.XGBRegressor()
    model.load_model(os.path.join(model_path, "model.json"))

    # Read in feature names
    with open(os.path.join(model_path, "feature_names.json")) as f:
        cols_when_model_builds = json.load(f)
    X = X.loc[:, cols_when_model_builds]

    # Compute model prediction
    y_pred = model.predict(X)
    # Add model prediction to feature data
    identifier_data.loc[:, "prediction"] = y_pred
    X = X.reset_index()
    all_scored_data = pd.merge(identifier_data, X, how="inner", on="index")
    all_scored_data = all_scored_data.drop(columns=["index"])

    # Save scored date
    all_scored_data.to_csv(os.path.join(pipeline_folder_path, "scored_data.csv"))

    print(all_scored_data.sort_values(by=["prediction"]))


if __name__ == "__main__":
    # Test pipeline results
    batch_name = "batch_2023-06-28"
    pipeline_folder_path = os.path.join(DIRNAME, "../pipeline_results", batch_name)

    # Score contracts in pipeline
    score_players(pipeline_folder_path, model_version="version_1")
