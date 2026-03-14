import os
import pandas as pd

# --------------------------------------------
# Project Paths
# --------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_PATH = os.path.join(BASE_DIR, "dataset", "kc_house_data.csv")
CLEAN_DATA_PATH = os.path.join(BASE_DIR, "dataset", "cleaned_dataset.csv")


# --------------------------------------------
# Load Raw Dataset
# --------------------------------------------

def load_raw_data():
    """Load the raw dataset"""

    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Raw dataset not found at {RAW_DATA_PATH}")

    df = pd.read_csv(RAW_DATA_PATH)

    print(f"\nRaw dataset loaded successfully")
    print(f"Shape: {df.shape}")

    return df


# --------------------------------------------
# Load Cleaned Dataset
# --------------------------------------------

def load_clean_data():
    """Load the cleaned dataset"""

    if not os.path.exists(CLEAN_DATA_PATH):
        raise FileNotFoundError(
            f"Clean dataset not found at {CLEAN_DATA_PATH}")

    df = pd.read_csv(CLEAN_DATA_PATH)

    print(f"\nClean dataset loaded successfully")
    print(f"Shape: {df.shape}")

    return df


# --------------------------------------------
# Dataset Summary
# --------------------------------------------

def get_dataset_info(df):
    """Print dataset summary"""

    print("\nDataset Information")
    print("-" * 40)

    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nData Types:")
    print(df.dtypes)
