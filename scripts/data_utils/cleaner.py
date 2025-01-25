import re
import os
import sys
import pandas as pd

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