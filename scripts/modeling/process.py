from datetime import date
import os
import sys
import pandas as pd

# Setup logger for data_loader
sys.path.append(os.path.join(os.path.abspath(__file__), '..', '..', '..'))
from scripts.utils.logger import setup_logger
from scripts.data_utils.loaders import save_data
from scripts.data_utils.feature_engineering import *
from scripts.modeling.rfms_woe import woe_rfms_pipeline

logger = setup_logger("process")

# ==========================================
# Main Data Processing Function
# ==========================================
def process_data(data, numerical_features, date_column, customer_column, recency_column, frequency_column, monetary_column, severity_column, target_column, customer_label, columns, rfms_features, scaler=None, output_dir: str = None) -> pd.DataFrame:
    """
    Loads, processes, and saves data.
    """
    logger.info("Cleaning...")

    data_aggregated = create_aggregate_features(data)
    data_temporal = extract_temporal_features(data_aggregated, date_column)
    data_scaled, scaler = normalize_standardize_numerical_features(data_temporal, numerical_features, mode='standard', scaler=scaler)
    data_rfms_classified, _, scaler = woe_rfms_pipeline(data_scaled, customer_column, recency_column, frequency_column, monetary_column, 
                                                severity_column, target_column, customer_label, columns, rfms_features, scaler, output_dir)
    
    data_processed = data_rfms_classified.reset_index(drop=True)

    if output_dir:
        logger.info("Saving processed data...")
        output_file = os.path.join(output_dir, "data_preprocessed")
        save_data(data_processed, output_file + ".csv")
        save_data(data_processed, output_file + ".json")
        save_data(scaler, output_file + ".pkl")
        logger.info(f"Processed data saved to {output_dir}")
    
    return data_processed