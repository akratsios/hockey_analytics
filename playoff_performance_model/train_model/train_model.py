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


def train_model(model_version: str):
    # Read in data
    all_data = pd.read_pickle(
        os.path.join(DIRNAME, "training_data", "model_training_data.pkl")
    )

    # Identifier data will not be included in training or scoring
    identifier_cols = ["playerId", "action_season", "action_date"]
    all_data = all_data.drop(columns=identifier_cols)

    # Split data into train, test, and oot
    target_variable = "gamescore_toi"
    X = all_data.drop(columns=[target_variable]).copy()
    y = all_data.loc[:, target_variable].copy()
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
    regressor = xgb.XGBRegressor(
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
    search = GridSearchCV(regressor, param_grid, cv=5).fit(X_train, y_train)

    print("The best hyperparameters are ", search.best_params_)

    regressor = xgb.XGBRegressor(
        learning_rate=search.best_params_["learning_rate"],
        n_estimators=search.best_params_["n_estimators"],
        max_depth=search.best_params_["max_depth"],
        eval_metric="rmse",
        enable_categorical=True,
    )
    regressor.fit(X_train, y_train)

    # Save the model
    model_path = os.path.join(DIRNAME, "../models", model_version)
    Path(model_path).mkdir(parents=True, exist_ok=True)
    regressor.save_model(os.path.join(model_path, "model.json"))
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
    predictions = regressor.predict(X_train)
    rmse_train = np.sqrt(root_mean_squared_error(y_train, predictions))
    print("The RMSE train score is %.5f" % rmse_train)

    predictions = regressor.predict(X_test)
    rmse_test = np.sqrt(root_mean_squared_error(y_test, predictions))
    print("The RMSE test score is %.5f" % rmse_test)

    # Plot feature importance
    plt.style.use("fivethirtyeight")
    plt.rcParams.update({"font.size": 16})

    fig, ax = plt.subplots(figsize=(12, 6))
    plot_importance(regressor, max_num_features=8, ax=ax)
    plt.show()


if __name__ == "__main__":
    train_model("version_1")
