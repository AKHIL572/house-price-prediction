import joblib
import pickle
import numpy as np
import pandas as pd
import os


MODEL_PATH = os.path.join("models", "house_price_pipeline.pkl")
SCALER_PATH = os.path.join("models", "scaler.pkl")
COLUMNS_PATH = os.path.join("models", "columns.pkl")


pipeline = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

with open(COLUMNS_PATH, "rb") as f:
    feature_columns = pickle.load(f)


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


def prepare_input(input_dict):

    df = pd.DataFrame([input_dict])

    # --------------------------------
    # Feature Engineering
    # --------------------------------

    df["basement_ratio"] = df["sqft_basement"] / df["sqft_living"]

    df["basement_ratio"] = df["basement_ratio"].replace(
        [np.inf, -np.inf], 0
    )

    df["basement_ratio"] = df["basement_ratio"].fillna(0)

    # --------------------------------
    # Zipcode Encoding
    # --------------------------------

    df["zipcode"] = df["zipcode"].astype(str)

    df = pd.get_dummies(df, columns=["zipcode"])

    # --------------------------------
    # Align Columns
    # --------------------------------

    for col in feature_columns:

        if col not in df.columns:
            df[col] = 0

    df = df[feature_columns]

    # --------------------------------
    # Scaling
    # --------------------------------

    cols_to_scale = [
        c for c in continuous_cols if c in df.columns
    ]

    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    return df


def predict_price(input_dict):

    processed = prepare_input(input_dict)

    log_price = pipeline.predict(processed)[0]

    price = np.expm1(log_price)

    return float(price)
