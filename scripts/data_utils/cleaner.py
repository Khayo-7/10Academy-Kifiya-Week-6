import re
import os
import sys
import pandas as pd
from sklearn.impute import SimpleImputer
from typing import List, Dict, Optional, Any

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

def drop_irrelevant_columns(data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Drops irrelevant columns from the DataFrame.
    """
    original_cols = data.columns
    data = data.drop(columns=[col for col in columns if col in data], errors="ignore")
    logger.info(f"Dropped columns: {set(original_cols) - set(data.columns)}. Remaining columns: {data.columns.tolist()}")
    return data

def drop_missing(data: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Drops irrelevant columns from the DataFrame.
    """
    logger.info("Removing row with null values...")
    data = data.dropna(subset=columns)
    return data

def handle_missing_values(data: pd.DataFrame, strategies: Dict[str, str]) -> pd.DataFrame:
    """
    Handles missing values with column-specific strategies.
    """
    for col, strategy in strategies.items():
        if col in data.columns:
            imputer = SimpleImputer(strategy=strategy)
            try:
                data[col] = imputer.fit_transform(data[[col]]).ravel()
                logger.info(f"Imputed missing values in {col} using strategy: {strategy}")
            except Exception as e:
                logger.error(f"Error imputing column {col}: {e}")
    return data

# def handle_missing_values_general(data: pd.DataFrame) -> pd.DataFrame:
#     """
#     Handles missing values by imputing:
#     - Mode for categorical columns
#     - Mean for numerical columns
#     """
#     missing_value_columns = data.columns[data.isnull().any()]
#     for col in missing_value_columns:
#         if data[col].dtype == "object" or data[col].dtype.name == "category":
#             imputer = SimpleImputer(strategy="most_frequent")
#         else:
#             imputer = SimpleImputer(strategy="mean")
#         try:
#             data[col] = imputer.fit_transform(data[[col]]).ravel()
#             logger.info(f"Imputed missing values in {col}")
#         except Exception as e:
#             logger.error(f"Error imputing column {col}: {e}")
#     return data

def remove_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    """
    Removes duplicate rows from the DataFrame.
    """
    initial_count = len(data)
    data = data.drop_duplicates()    
    logger.info(f"Removed {initial_count - len(data)} duplicate rows.")
    return data

def convert_data_types(data: pd.DataFrame, conversions: Dict[str, str]) -> pd.DataFrame:
    """
    Converts specified columns to appropriate data types.
    """
    for col, dtype in conversions.items():
        if col in data.columns:
            try:
                data[col] = data[col].astype(dtype, errors="ignore")
                logger.info(f"Converted column {col} to {dtype}")
            except Exception as e:
                logger.error(f"Failed to convert {col} to {dtype}: {e}")
    return data

def convert_date(data: pd.DataFrame, date_column) -> pd.DataFrame:
    """
    Converts date_column to datetime
    """
    if date_column in data.columns:
        data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
        data = data.dropna(subset=[date_column])  # Drop invalid datetimes
    return data

def standardize_categorical_columns(data: pd.DataFrame, categorical_columns: List[str]) -> pd.DataFrame:
    """
    Standardizes categorical columns.
    """
    for col in categorical_columns:
        if col in data.columns:
            data[col] = data[col].str.strip().str.upper()
            
        logger.info(f"Standardized categorical column: {col}")
    return data
