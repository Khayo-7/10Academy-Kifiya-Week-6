import os
import sys
import pandas as pd

# Setup logger for data_loader
sys.path.append(os.path.join(os.path.abspath(__file__), '..', '..', '..'))
from scripts.utils.logger import setup_logger
from scripts.data_utils.loaders import save_data
from scripts.data_utils.cleaner import *

logger = setup_logger("preprocess")

# ==========================================
# Main Data Preprocessing Function
# ==========================================

def clean_data(data: pd.DataFrame, irrelevant_columns, categorical_columns, missing_value_strategies, date_column, dtype_conversions) -> pd.DataFrame:
    """
    Main cleaning pipeline function.
    """
    data = data.copy()

    # Cleaning pipeline
    data = drop_irrelevant_columns(data, irrelevant_columns)
    data = handle_missing_values(data, missing_value_strategies)
    # data = drop_missing(data, columns)
    data = remove_duplicates(data)
    data = convert_date(data, date_column)
    data = convert_data_types(data, dtype_conversions)
    data = standardize_categorical_columns(data, categorical_columns)

    logger.info("Data cleaning pipeline executed successfully.")
    return data

def preprocess_data(data: pd.DataFrame, irrelevant_columns: str, categorical_columns, numerical_columns, missing_value_strategies, date_column, dtype_conversions, filename: str, output_dir: str) -> pd.DataFrame:
    """
    Loads, preprocesses, and saves cleaned data.
    """
    logger.info("Cleaning...")

    cleaned_data = clean_data(data, irrelevant_columns, categorical_columns, missing_value_strategies, date_column, dtype_conversions)

    preprocessed_data = cleaned_data.reset_index(drop=True)

    logger.info("Saving preprocessed data...")
    output_file = os.path.join(output_dir, filename)
    save_data(data, output_file + ".csv")
    save_data(data, output_file + ".json")
    logger.info(f"Preprocessed data saved to {output_dir}")

    return preprocessed_data
