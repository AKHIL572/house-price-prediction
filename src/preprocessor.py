import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_data(df):
    """
    Perform preprocessing and return train/test datasets
    """

    df = df.copy()

    # ------------------------------------------------
    # Target variable
    # ------------------------------------------------

    if "price_log" not in df.columns:
        df["price_log"] = np.log1p(df["price"])

    # ------------------------------------------------
    # Feature Engineering
    # ------------------------------------------------

    if "sale_year" in df.columns and "yr_built" in df.columns:
        df["house_age"] = df["sale_year"] - df["yr_built"]

    if "sqft_basement" in df.columns and "sqft_living" in df.columns:

        df["basement_ratio"] = df["sqft_basement"] / df["sqft_living"]

        df["basement_ratio"] = df["basement_ratio"].replace(
            [np.inf, -np.inf], 0
        )

        df["basement_ratio"] = df["basement_ratio"].fillna(0)

    # ------------------------------------------------
    # Remove unused columns
    # ------------------------------------------------

    drop_cols = [
        "id",
        "date",
        "total_sqft",
        "yr_built",
        "yr_renovated",
        "sale_year"
    ]

    df.drop(
        columns=[c for c in drop_cols if c in df.columns],
        inplace=True
    )

    # ------------------------------------------------
    # Encode zipcode
    # ------------------------------------------------

    if "zipcode" in df.columns:

        df["zipcode"] = df["zipcode"].astype(str)

        df = pd.get_dummies(
            df,
            columns=["zipcode"],
            drop_first=True
        )

    # ------------------------------------------------
    # Feature / Target split
    # ------------------------------------------------

    X = df.drop(["price", "price_log"], axis=1)

    y = df["price_log"]

    feature_columns = X.columns.tolist()

    # ------------------------------------------------
    # Train Test Split
    # ------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # ------------------------------------------------
    # Scaling (continuous features only)
    # ------------------------------------------------

    continuous_cols = [
        "sqft_living",
        "sqft_lot",
        "sqft_above",
        "sqft_basement",
        "sqft_living15",
        "sqft_lot15",
        "lat",
        "long",
        "house_age"
    ]

    scaler = StandardScaler()

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    cols_to_scale = [
        c for c in continuous_cols if c in X_train.columns
    ]

    X_train_scaled[cols_to_scale] = scaler.fit_transform(
        X_train[cols_to_scale]
    )

    X_test_scaled[cols_to_scale] = scaler.transform(
        X_test[cols_to_scale]
    )

    return (
        X_train,
        X_test,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler,
        feature_columns
    )
