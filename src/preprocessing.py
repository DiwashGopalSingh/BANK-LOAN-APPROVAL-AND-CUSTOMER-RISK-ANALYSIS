"""Data Preprocessing & Feature Engineering Module.

This module cleans raw loan application records, derives financial ratio features,
encodes categorical variables, and produces train/test splits for downstream modeling.
Conforms to PEP 8 standards with explicit docstrings and named constants.
"""

import os
from typing import Dict, List, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ==========================================
# Named Constants & Configuration
# ==========================================
RAW_DATA_PATH = "data/raw/loan_approval_dataset.csv"
PROCESSED_DATA_PATH = "data/processed/loan_approval_processed.csv"
PROCESSED_TRAIN_PATH = "data/processed/train.csv"
PROCESSED_TEST_PATH = "data/processed/test.csv"

TEST_SPLIT_RATIO = 0.20
RANDOM_STATE = 42

TARGET_COLUMN = "loan_status"
ID_COLUMN = "loan_id"

EDUCATION_MAPPING = {"Graduate": 1, "Not Graduate": 0}
SELF_EMPLOYED_MAPPING = {"Yes": 1, "No": 0}
TARGET_MAPPING = {"Approved": 1, "Rejected": 0}

FEATURE_COLUMNS: List[str] = [
    "no_of_dependents",
    "education",
    "self_employed",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
    "total_assets",
    "loan_to_income",
    "asset_to_loan",
]


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans raw dataframe by stripping whitespace from column headers and string values.

    Args:
        df: Raw input DataFrame.

    Returns:
        DataFrame with stripped column names and normalized text entries.
    """
    cleaned_df = df.copy()
    cleaned_df.columns = cleaned_df.columns.str.strip()

    # Strip whitespaces from object/string columns
    string_cols = cleaned_df.select_dtypes(include=["object", "string"]).columns
    for col in string_cols:
        cleaned_df[col] = cleaned_df[col].astype(str).str.strip()

    return cleaned_df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes financial ratios and composite asset indicators.

    Engineered features:
    - total_assets: Residential + Commercial + Luxury + Bank assets.
    - loan_to_income: Loan amount divided by annual income.
    - asset_to_loan: Total assets divided by loan amount (collateral coverage).

    Args:
        df: DataFrame containing financial and asset attributes.

    Returns:
        DataFrame enriched with engineered feature columns.
    """
    data = df.copy()

    # Total asset value
    data["total_assets"] = (
        data["residential_assets_value"]
        + data["commercial_assets_value"]
        + data["luxury_assets_value"]
        + data["bank_asset_value"]
    )

    # Loan to income ratio (safeguard division by zero)
    income_safe = data["income_annum"].replace(0, np.nan)
    data["loan_to_income"] = (data["loan_amount"] / income_safe).fillna(0.0)

    # Asset to loan ratio (safeguard division by zero)
    loan_safe = data["loan_amount"].replace(0, np.nan)
    data["asset_to_loan"] = (data["total_assets"] / loan_safe).fillna(0.0)

    return data


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encodes binary categorical features and target into numeric 0/1 representation.

    Args:
        df: DataFrame with raw categorical variables.

    Returns:
        DataFrame with encoded numerical variables.
    """
    encoded_df = df.copy()

    if "education" in encoded_df.columns:
        encoded_df["education"] = encoded_df["education"].map(EDUCATION_MAPPING)

    if "self_employed" in encoded_df.columns:
        encoded_df["self_employed"] = encoded_df["self_employed"].map(SELF_EMPLOYED_MAPPING)

    if TARGET_COLUMN in encoded_df.columns:
        encoded_df[TARGET_COLUMN] = encoded_df[TARGET_COLUMN].map(TARGET_MAPPING)

    return encoded_df


def prepare_applicant_features(applicant_dict: Dict[str, Union[int, float, str]]) -> pd.DataFrame:
    """Transforms a single applicant dictionary into model-ready features.

    Used by the Streamlit application and inference endpoints.

    Args:
        applicant_dict: Dictionary containing applicant inputs.

    Returns:
        Single-row DataFrame matching the exact feature order of trained models.
    """
    record_df = pd.DataFrame([applicant_dict])
    cleaned_df = clean_raw_data(record_df)
    engineered_df = engineer_features(cleaned_df)
    encoded_df = encode_categorical_features(engineered_df)

    # Ensure all required features exist in correct order
    return encoded_df[FEATURE_COLUMNS]


def preprocess_pipeline(
    raw_data_path: str = RAW_DATA_PATH,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """Runs the full preprocessing pipeline on the raw dataset and returns train/test splits.

    Args:
        raw_data_path: Path to the raw CSV file.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_names).
    """
    raw_df = pd.read_csv(raw_data_path)
    cleaned_df = clean_raw_data(raw_df)
    engineered_df = engineer_features(cleaned_df)
    processed_df = encode_categorical_features(engineered_df)

    # Drop identifier if present
    if ID_COLUMN in processed_df.columns:
        processed_df = processed_df.drop(columns=[ID_COLUMN])

    X = processed_df[FEATURE_COLUMNS]
    y = processed_df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SPLIT_RATIO,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test, FEATURE_COLUMNS


def run_and_save_preprocessing() -> None:
    """Executes the pipeline and persists processed datasets to data/processed/."""
    os.makedirs("data/processed", exist_ok=True)

    raw_df = pd.read_csv(RAW_DATA_PATH)
    cleaned_df = clean_raw_data(raw_df)
    engineered_df = engineer_features(cleaned_df)
    processed_df = encode_categorical_features(engineered_df)

    if ID_COLUMN in processed_df.columns:
        processed_df = processed_df.drop(columns=[ID_COLUMN])

    # Save full processed dataset
    processed_df.to_csv(PROCESSED_DATA_PATH, index=False)

    # Generate split and save
    X_train, X_test, y_train, y_test, _ = preprocess_pipeline(RAW_DATA_PATH)

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_df.to_csv(PROCESSED_TRAIN_PATH, index=False)
    test_df.to_csv(PROCESSED_TEST_PATH, index=False)

    print("Preprocessing completed successfully:")
    print(f" - Full processed dataset: {PROCESSED_DATA_PATH} (Shape: {processed_df.shape})")
    print(f" - Train set: {PROCESSED_TRAIN_PATH} (Shape: {train_df.shape})")
    print(f" - Test set: {PROCESSED_TEST_PATH} (Shape: {test_df.shape})")


if __name__ == "__main__":
    run_and_save_preprocessing()
