import re
import os
import sys
import pandas as pd
from sklearn.impute import SimpleImputer

# Setup logger for cleaning operations
sys.path.append(os.path.join(os.path.abspath(__file__), '..', '..', '..'))
from scripts.utils.logger import setup_logger


logger = setup_logger("cleaning")

# ==========================================
# Helper Functions for Cleaning Operations
# ==========================================

def generate_summary_statistics(data: pd.DataFrame, output_path: str = None) -> pd.DataFrame:
    summary = data.describe(include="all").transpose()
    if output_path:
        summary.to_csv(output_path, index=True)
        logger.info(f"Summary statistics saved to {output_path}")
    return summary

def analyze_missing_values(data: pd.DataFrame, output_path: str = None) -> pd.DataFrame:
    missing_summary = data.isnull().sum().reset_index()
    missing_summary.columns = ['Column', 'MissingCount']
    missing_summary['MissingPercentage'] = (missing_summary['MissingCount'] / len(data)) * 100
    missing_summary = missing_summary[missing_summary['MissingCount'] > 0]
    
    logger.info("Missing value analysis completed.")
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        missing_summary.to_csv(output_path, index=False)
        logger.info(f"Missing values analysis saved to {output_path}")
    
    return missing_summary

def drop_irrelevant_columns(data: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Drops irrelevant columns from the DataFrame.
    """
    data = data.drop(columns=[col for col in columns if col in data], errors="ignore")
    return data

def drop_missing(data: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Drops irrelevant columns from the DataFrame.
    """
    logger.info("Removing row with null values...")
    data = data.dropna(subset=columns)
    return data

def handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """
    Handles missing values by imputing:
    - Mode for categorical columns
    - Mean for numerical columns
    """
    missing_value_columns = data.columns[data.isnull().any()]
    for col in missing_value_columns:
        if data[col].dtype == "object":
            imputer = SimpleImputer(strategy="most_frequent")
        else:
            imputer = SimpleImputer(strategy="mean")
        data[col] = imputer.fit_transform(data[[col]])
    return data

def remove_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    """
    Removes duplicate rows from the DataFrame.
    """
    return data.drop_duplicates()

def convert_data_types(data: pd.DataFrame) -> pd.DataFrame:
    """
    Converts relevant columns to proper data types.
    - Converts TransactionStartTime to datetime
    - Converts CountryCode to integer
    """
    if "TransactionStartTime" in data.columns:
        data["TransactionStartTime"] = pd.to_datetime(data["TransactionStartTime"], errors="coerce")
        data = data.dropna(subset=["TransactionStartTime"])  # Drop invalid datetimes
    if "CountryCode" in data.columns:
        data["CountryCode"] = data["CountryCode"].astype(int, errors="ignore")
    return data

def standardize_categorical_columns(data: pd.DataFrame, categorical_columns: list) -> pd.DataFrame:
    """
    Standardizes categorical columns by:
    - Stripping whitespace
    - Converting to uppercase
    """
    for col in categorical_columns:
        if col in data.columns:
            data[col] = data[col].str.strip().str.upper()
    return data
