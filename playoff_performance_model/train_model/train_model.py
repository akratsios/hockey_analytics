import json
import os
import sys
from datetime import datetime
from functools import reduce
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import plot_importance

# Get path of current file's directory
DIRNAME = os.path.dirname(os.path.realpath(__file__))


def select_features(features_df: pd.DataFrame, target: pd.Series, num_features: int):
    # configure to select num_features
    fs = SelectKBest(score_func=mutual_info_regression, k=num_features)
    # learn relationship from training data
    fs.fit(features_df, target)
    # Get columns to keep and create new dataframe with those only
    cols_idxs = fs.get_support(indices=True)
    features_df_new = features_df.iloc[:, cols_idxs]
    return features_df_new


def train_model(model_version: str):
    # Read in data
    all_data = pd.read_pickle(
        os.path.join(DIRNAME, "training_data", "training_feature_data.pkl")
    )

    # Identifier data will not be included in training or scoring
    identifier_cols = ["playerId", "action_season", "action_date"]
    target_variable = "gamescore_toi"
    y = all_data.loc[:, target_variable]
    X = all_data.drop(columns=[*identifier_cols, target_variable])

    # Select K best features for the model
    X = select_features(X, y, 100)

    # Split data into train, test, and oot
    y.copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    print("\nTraining model...")
    # =========================================================================
    # XGBoost regression:
    # Parameters:
    # n_estimators  "Number of gradient boosted trees. Equivalent to number
    #                of boosting rounds."
    # learning_rate "Boosting learning rate (also known as “eta”)"
    # max_depth     "Maximum depth of a tree. Increasing this value will make
    #                the model more complex and more likely to overfit."
    # =========================================================================
    model = xgb.XGBRegressor(
        eval_metric="rmse",
        enable_categorical=True,
    )

    # =========================================================================
    # exhaustively search for the optimal hyperparameters
    # =========================================================================

    # set up our search grid
    param_grid = {
        "max_depth": [4, 5, 6],
        "n_estimators": [500, 600, 700],
        "learning_rate": [0.01, 0.015],
    }

    # try out every combination of the above values
    search = GridSearchCV(model, param_grid, cv=5).fit(X_train, y_train)

    print("The best hyperparameters are ", search.best_params_)

    model = xgb.XGBRegressor(
        learning_rate=search.best_params_["learning_rate"],
        n_estimators=search.best_params_["n_estimators"],
        max_depth=search.best_params_["max_depth"],
        eval_metric="rmse",
        enable_categorical=True,
    )
    model.fit(X_train, y_train)

    # Save the model
    model_path = os.path.join(DIRNAME, "../models", model_version)
    Path(model_path).mkdir(parents=True, exist_ok=True)
    model.save_model(os.path.join(model_path, "model.json"))

    # Save the feature names (preserve order)
    feature_names = model.get_booster().feature_names
    with open(os.path.join(model_path, "feature_names.json"), "w") as f:
        json.dump(feature_names, f)

    # Save model params to json file
    params = model.get_xgb_params()
    # np.int64 can't be dumped to json
    for key, value in params.items():
        if type(value) == np.int64:
            params[key] = int(value)
    with open(os.path.join(model_path, "params.json"), "w") as f:
        json.dump(params, f)
    print("Finished training model.")

    # =========================================================================
    # To use early_stopping_rounds:
    # "Validation metric needs to improve at least once in every
    # early_stopping_rounds round(s) to continue training."
    # =========================================================================
    # first perform a test/train split

    # X_train,X_test,y_train,y_test = train_test_split(X_train,y_train, test_size = 0.2)
    # regressor.fit(X_train, y_train, early_stopping_rounds=6, eval_set=[(X_test, y_test)], verbose=False)

    # =========================================================================
    # use the model to predict the prices for the test data
    # =========================================================================
    predictions = model.predict(X_train)
    rmse_train = np.sqrt(root_mean_squared_error(y_train, predictions))
    print("The RMSE train score is %.5f" % rmse_train)

    predictions = model.predict(X_test)
    rmse_test = np.sqrt(root_mean_squared_error(y_test, predictions))
    print("The RMSE test score is %.5f" % rmse_test)

    # Plot feature importance
    plt.style.use("fivethirtyeight")
    plt.rcParams.update({"font.size": 16})

    fig, ax = plt.subplots(figsize=(12, 6))
    plot_importance(model, max_num_features=8, ax=ax)
    # plt.savefig(os.path.join(model_path, "feature_importance_xgboost.png"))
    plt.show()


if __name__ == "__main__":
    train_model("version_2")
