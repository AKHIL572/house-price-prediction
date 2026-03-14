"""
Model Training Script
"""

import os
import joblib
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.pipeline import Pipeline

from src.data_loader import load_clean_data
from src.preprocessor import preprocess_data


np.random.seed(42)


def train_models():

    print("\nLoading dataset...")

    df = load_clean_data()

    (
        X_train,
        X_test,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler,
        feature_columns
    ) = preprocess_data(df)

    # ---------------------------------------------------
    # Define Models
    # ---------------------------------------------------

    models = {

        "Linear Regression": LinearRegression(),

        "Decision Tree": DecisionTreeRegressor(random_state=42),

        "Random Forest": RandomForestRegressor(random_state=42),

        "Gradient Boosting": GradientBoostingRegressor(random_state=42)

    }

    results = []

    # ---------------------------------------------------
    # Train Baseline Models
    # ---------------------------------------------------

    print("\nTraining baseline models...")

    for name, model in models.items():

        if name == "Linear Regression":

            model.fit(X_train_scaled, y_train)

            preds = model.predict(X_test_scaled)

            cv_score = cross_val_score(
                model,
                X_train_scaled,
                y_train,
                cv=5,
                scoring="r2"
            ).mean()

        else:

            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            cv_score = cross_val_score(
                model,
                X_train,
                y_train,
                cv=5,
                scoring="r2"
            ).mean()

        mae = mean_absolute_error(y_test, preds)

        rmse = np.sqrt(mean_squared_error(y_test, preds))

        r2 = r2_score(y_test, preds)

        results.append([name, mae, rmse, r2, cv_score])

    results_df = pd.DataFrame(
        results,
        columns=["Model", "MAE", "RMSE", "R2", "CV R2"]
    )

    print(results_df.sort_values("RMSE"))

    # ---------------------------------------------------
    # Select Top Models
    # ---------------------------------------------------

    top_models = results_df.sort_values("RMSE").head(2)["Model"].values

    best_models = {}

    for model_name in top_models:

        print(f"\nTuning {model_name}")

        if model_name == "Random Forest":

            param_dist = {

                "n_estimators": [200, 300, 400, 500],

                "max_depth": [10, 20, 30, None],

                "min_samples_split": [2, 5, 10],

                "min_samples_leaf": [1, 2, 4],

                "max_features": ["sqrt", "log2"]

            }

            model = RandomForestRegressor(random_state=42)

        else:

            param_dist = {

                "n_estimators": [100, 200, 300],

                "learning_rate": [0.01, 0.05, 0.1],

                "max_depth": [3, 4, 5],

                "subsample": [0.8, 1.0]

            }

            model = GradientBoostingRegressor(random_state=42)

        search = RandomizedSearchCV(

            model,

            param_distributions=param_dist,

            n_iter=25,

            cv=5,

            scoring="neg_root_mean_squared_error",

            n_jobs=-1,

            random_state=42,

            verbose=1

        )

        search.fit(X_train, y_train)

        best_models[model_name] = search.best_estimator_

    # ---------------------------------------------------
    # Evaluate Tuned Models
    # ---------------------------------------------------

    final_results = []

    for name, model in best_models.items():

        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)

        rmse = np.sqrt(mean_squared_error(y_test, preds))

        r2 = r2_score(y_test, preds)

        final_results.append([name, mae, rmse, r2])

    final_df = pd.DataFrame(
        final_results,
        columns=["Model", "MAE", "RMSE", "R2"]
    )

    print(final_df.sort_values("RMSE"))

    best_model_name = final_df.sort_values("RMSE").iloc[0]["Model"]

    final_model = best_models[best_model_name]

    print(f"\nFinal Selected Model: {best_model_name}")

    # ---------------------------------------------------
    # Save Model
    # ---------------------------------------------------

    pipeline = Pipeline([
        ("model", final_model)
    ])

    pipeline.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)

    joblib.dump(pipeline, "models/house_price_pipeline.pkl")

    joblib.dump(scaler, "models/scaler.pkl")

    with open("models/columns.pkl", "wb") as f:
        pickle.dump(feature_columns, f)

    print("\nModel saved successfully!")


if __name__ == "__main__":
    train_models()
