import os
import sys
import pandas as pd

# Setup logger for data_loader
sys.path.append(os.path.join(os.path.abspath(__file__), '..', '..', '..'))
from scripts.utils.logger import setup_logger
from scripts.data_utils.loaders import save_csv, save_json

logger = setup_logger("preprocess")

# ==========================================
# Main Data Preprocessing Function
# ==========================================

def clean_data(data: pd.DataFrame, column: str) -> pd.DataFrame:
    
    pass
    
    return data

def preprocess_data(data: pd.DataFrame, column: str, filename: str, output_dir: str, explode: bool = False, save_in_csv: bool = True, save_in_json: bool = False) -> pd.DataFrame:
    """
    Loads, preprocesses, and saves cleaned data.
    """
    logger.info("Preprocessing text data...")
    cleaned_data = clean_data(data, column)

    logger.info("Removing empty...")
    preprocessed_data = cleaned_data.dropna(subset=[column])

    preprocessed_data = preprocessed_data.reset_index(drop=True)

    logger.info("Saving preprocessed data...")
    output_file = os.path.join(output_dir, filename)
    if save_in_csv:
        save_csv(preprocessed_data, output_file + ".csv")
    if save_in_json:
        save_json(preprocessed_data, output_file + ".json")

    logger.info(f"Preprocessed data saved to {output_dir}")

    return preprocessed_data
