import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Define required columns (final form)
set_a = [
    'DAYS_TO_MATURITY', 'previous_loan_default_count', 'Is_Active_loans_Inactive',
    'ARR_STATUS_CURRENT', 'DAY_SINCEPAYMENT', 'CURR_BAL', 'DAYS_RUNNING_SINCE'
]
set_b = set_a + ['previous_loans_count', 'INTEREST_RATE', 'sector_GENERAL']


def check_and_transform_data(df: pd.DataFrame):
    """
    Checks if dataset is transformed.
    If not, performs preprocessing:
    - Removes duplicates (if loan_id exists)
    - Drops rows with missing values in required raw columns
    - Encodes categorical fields as binary:
        * Is_Active_loans → Is_Active_loans_Inactive (1 if Inactive, else 0)
        * ARR_STATUS → ARR_STATUS_CURRENT (1 if Current, else 0)
        * sector → sector_GENERAL (1 if General, else 0)
    - Converts INTEREST_RATE to float
    - Standardizes numerical data
    Returns transformed DataFrame ready for prediction.
    """

    # --- Step 1: Detect if data already transformed (all required cols exist & numeric) ---
    if all(col in df.columns for col in set_b):
        # Check if all are numeric
        if all(np.issubdtype(df[col].dtype, np.number) for col in set_b):
            print("✅ Data already transformed. Returning as is...")
            return df[set_b]

    # --- Step 2: Handle raw data ---
    print("⚙️ Detected raw data. Performing transformation...")

    # If loan_id exists → drop duplicates
    if "loan_id" in df.columns:
        df = df.drop_duplicates(subset=["loan_id"], keep="first")

    # Map categorical columns to new binary encoded form
    if "Is_Active_loans" in df.columns:
        df["Is_Active_loans_Inactive"] = df["Is_Active_loans"].str.strip().str.upper().eq("INACTIVE").astype(int)
    else:
        raise ValueError("Missing required column: Is_Active_loans")

    if "ARR_STATUS" in df.columns:
        df["ARR_STATUS_CURRENT"] = df["ARR_STATUS"].str.strip().str.upper().eq("CURRENT").astype(int)
    else:
        raise ValueError("Missing required column: ARR_STATUS")

    if "sector" in df.columns:
        df["sector_GENERAL"] = df["sector"].str.strip().str.upper().eq("GENERAL").astype(int)
    else:
        raise ValueError("Missing required column: sector")

    # Convert INTEREST_RATE to numeric
    if "INTEREST_RATE" in df.columns:
        df["INTEREST_RATE"] = (
            df["INTEREST_RATE"]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.strip()
        )
        df["INTEREST_RATE"] = pd.to_numeric(df["INTEREST_RATE"], errors="coerce")
    else:
        raise ValueError("Missing required column: INTEREST_RATE")

    # Drop rows with missing values in required columns
    required_raw = [
        "DAYS_TO_MATURITY", "previous_loan_default_count", "Is_Active_loans",
        "ARR_STATUS", "DAY_SINCEPAYMENT", "CURR_BAL", "DAYS_RUNNING_SINCE",
        "previous_loans_count", "INTEREST_RATE", "sector"
    ]
    df = df.dropna(subset=required_raw)

    # Select transformed set_b
    df_final = df[set_b].copy()

    # Scale numeric features
    numeric_cols = [c for c in df_final.columns if c not in ["Is_Active_loans_Inactive", "ARR_STATUS_CURRENT", "sector_GENERAL"]]
    scaler = StandardScaler()
    df_final[numeric_cols] = scaler.fit_transform(df_final[numeric_cols])

    print("✅ Data transformed successfully.")
    return df_final

