from datetime import date
import os
import sys
import joblib
import logging
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

try:
    from scripts.utils.logger import setup_logger
    from scripts.data_utils.loaders import load_csv
    from scripts.modeling.process import process_data
    from scripts.data_utils.preprocess import preprocess_data, clean_data
except ImportError as e:
    logging(f"Import error: {e}. Please check the module path.")

# Setup logger for deployement
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs')
logger = setup_logger("deployement", log_dir)  

resources_dir = os.path.join('resources')
scaler_filepath = os.path.join(resources_dir, 'scaler.pkl')

try:
    # Load previously saved scaler (used during training)
    logger.info("Loading previously saved scaler...")
    scaler = joblib.load(scaler_filepath)
except Exception as e:
    logger.error(f"Error initializing application: {e}")
    raise

def preprocess_input(data: pd.DataFrame) -> np.ndarray:
    """
    Preprocess raw input for prediction.
    Args:
        data (pd.DataFrame): Input DataFrame containing features.
    Returns:
        np.ndarray: Preprocessed and ready-to-predict input array.
    """

    logger.info("Starting preprocessing of input data...")

    # Ensure consistent column ordering
    initial_columns = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId',
                        'CurrencyCode', 'CountryCode', 'ProviderId', 'ProductId',
                        'ProductCategory', 'ChannelId', 'Amount', 'Value',
                        'TransactionStartTime', 'PricingStrategy']
    drop_columns = ['TransactionStartTime', 'TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId']
    final_columns = ['CurrencyCode', 'CountryCode', 'ProviderId', 'ProductId',
                    'ProductCategory', 'ChannelId', 'Amount', 'Value', 'PricingStrategy',
                    'total_transaction_amount', 'avg_transaction_amount',
                    'transaction_count', 'std_transaction_amount', 'transaction_hour',
                    'transaction_day', 'transaction_month', 'transaction_year', 'Recency',
                    'Frequency', 'Monetary', 'Severity', 'Cluster', 'RFMS_Label']

    if not all(col in data.columns for col in initial_columns):
        missing_cols = [col for col in initial_columns if col not in data.columns]
        raise ValueError(f"Input data is missing required columns: {missing_cols}")

    # Define the target feature and feature lists
    irrelevant_columns = ['Unnamed: 16', 'Unnamed: 17']
    numerical_columns = ['Amount', 'Value', 'PricingStrategy']
    rfms_features = ['Recency', 'Frequency', 'Monetary', 'Severity']
    categorical_columns = ["CurrencyCode", "CountryCode", "ProviderId", "ProductId", "ProductCategory", "ChannelId"]
    aggregated_features = ['total_transaction_amount', 'avg_transaction_amount', 'transaction_count', 'std_transaction_amount']
    numerical_features = numerical_columns + aggregated_features
    columns = categorical_columns + numerical_features

    cluster_label = 'RFMS_Label'
    date_column = "TransactionStartTime"
    target_column = 'FraudResult'
    customer_column = "CustomerId"
    label_column = "RFMS_Label"
    recency_column = date_column
    frequency_column = date_column
    monetary_column = "Amount"
    severity_column = "Value"

    dtype_conversions = {"CountryCode": "int64", "CountryCode": "str"}
    missing_value_strategies = {
            "CountryCode": "most_frequent",
            "AccountId": "most_frequent",
            "ProviderId": "most_frequent",
            "PricingStrategy": "median",
            "Value": "mean",
    }

    logger.info("Cleaning data...")
    data = clean_data(data, irrelevant_columns, categorical_columns, numerical_columns, 
                      missing_value_strategies, date_column, dtype_conversions)
    
    logger.info("Processing data...")
    data = process_data(data, numerical_features, date_column, customer_column, recency_column, frequency_column,
                                monetary_column, severity_column, target_column, label_column, columns, rfms_features, scaler)

    data[cluster_label] = data[cluster_label].apply(lambda x: 1 if x == 'Good' else 0)

    logger.info("Dropping unnecessary columns...")
    data = date.drop(columns=list(set(drop_columns)))
    
    # Reorder columns to match final_columns
    logger.info("Reordering columns...")
    logger.info(data.columns)
    data = data[final_columns]

    # Convert to NumPy array
    logger.info("Converting data to NumPy array...")
    preprocessed_data = np.array(data.values, dtype=np.float32)

    logger.info("Preprocessing complete.")
    return preprocessed_data

