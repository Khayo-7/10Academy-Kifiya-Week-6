import os
import sys
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler

# Setup logger for data_loader
sys.path.append(os.path.join(os.path.abspath(__file__), '..', '..', '..'))
from scripts.data_utils.cleaner import validate_convert_date_column
from scripts.utils.logger import setup_logger

logger = setup_logger("feature_engineering")

def create_aggregate_features(data: pd.DataFrame, output_path: str = None) -> pd.DataFrame:
    """
    Creates aggregate features such as total transaction amount, 
    average transaction amount, transaction count, and standard deviation of amounts.

    Args:
        data (pd.DataFrame): Input DataFrame.
        output_path (str, optional): Path to save the resulting DataFrame. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame with additional aggregate features.
    """
    logger.info("Creating aggregate features.")
    
    # Aggregate features grouped by CustomerId
    aggregated_data = data.groupby('CustomerId').agg(
        total_transaction_amount=('Amount', 'sum'),
        avg_transaction_amount=('Amount', 'mean'),
        transaction_count=('Amount', 'count'),
        std_transaction_amount=('Amount', 'std')
    ).reset_index()

    # Handle NaN for std deviation (single/no transaction customers)
    aggregated_data['std_transaction_amount'] = aggregated_data['std_transaction_amount'].fillna(0)
    
    logger.info("Aggregate features created successfully.")
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        aggregated_data.to_csv(output_path, index=False)
        logger.info(f"Aggregate features saved to {output_path}")
    
    data = data.merge(aggregated_data, on="CustomerId", how="left")
    logger.info("Aggregate features created and merged into the dataframe.")
    return data

def extract_temporal_features(data: pd.DataFrame, date_column, output_path: str = None) -> pd.DataFrame:
    """
    Extract temporal features like hour, day, month, and year from date column.

    Args:
        data (pd.DataFrame): Input DataFrame.
        output_path (str, optional): Path to save the resulting DataFrame. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame with extracted features.
    """
    logger.info("Extracting datetime features.")
    
    data[date_column] = pd.to_datetime(data[date_column], errors='coerce')
    data['transaction_hour'] = data[date_column].dt.hour
    data['transaction_day'] = data[date_column].dt.day
    data['transaction_month'] = data[date_column].dt.month
    data['transaction_year'] = data[date_column].dt.year
    logger.info("Temporal features extracted.")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        data.to_csv(output_path, index=False)
        logger.info(f"Datetime features saved to {output_path}")

    return data

def encode_categorical_variables(data: pd.DataFrame, one_hot_columns: list, label_columns: list, output_path: str = None) -> pd.DataFrame:
    """
    Encodes categorical variables using both One-Hot Encoding and Label Encoding.

    Args:
        data (pd.DataFrame): Input DataFrame.
        one_hot_columns (list): List of columns to one-hot encode.
        label_columns (list): List of columns to label encode.
        output_path (str, optional): Path to save the resulting DataFrame. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame with encoded categorical features.
    """
    # One-Hot Encoding
    if one_hot_columns:
        logger.info("Performing One-Hot Encoding.")
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoded_data = pd.DataFrame(
            encoder.fit_transform(data[one_hot_columns]),
            columns=encoder.get_feature_names_out(one_hot_columns),
            index=data.index,
        )
        data = pd.concat([data.drop(columns=one_hot_columns), encoded_data], axis=1)

    # Label Encoding
    encoder_output = {}
    if label_columns:
        logger.info("Performing Label Encoding.")
        for col in label_columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            encoder_output[col] = le.classes_.tolist()
    
    # Save processed data if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        data.to_csv(output_path, index=False)
        logger.info(f"Encoded features saved to {output_path}")

    logger.info("Categorical variables encoded successfully.")    
    return data, encoder_output

def normalize_standardize_numerical_features(data: pd.DataFrame, columns: list, mode: str = 'standard', scaler=None, output_path: str = None) -> pd.DataFrame:
    """
    Normalizes or standardizes numerical columns.

    Args:
        data (pd.DataFrame): Input DataFrame.
        columns (list): List of numerical columns to scale.
        mode (str): Scaling method ('standard' or 'normalize').
        output_path (str, optional): Path to save the resulting DataFrame. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame with scaled numerical features.
    """
    logger.info(f"Scaling numerical features using {mode} method.")
    
    if scaler is None:
        scaler = StandardScaler() if mode == 'standard' else MinMaxScaler()
    
    data[columns] = scaler.fit_transform(data[columns])
    logger.info(f"Numerical features scaled using {mode} mode.")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        data.to_csv(output_path, index=False)
        logger.info(f"Scaled numerical features saved to {output_path}")
    
    return data, scaler

def calculate_rfms(data, customer_column, recency_column, frequency_column, monetary_column, severity_column, fraud_result_column):
    """Calculate RFMS scores dynamically."""

    logger.info("Calculating RFMS scores with recency.")

    # Group by customer and calculate RFMS scores
    reference_date = data[recency_column].max()
    data = validate_convert_date_column(data, recency_column)
    data['Recency'] = (reference_date - data[recency_column]).dt.days

    rfms_data = data.groupby(customer_column).agg(
        Recency=('Recency', 'min'),
        Frequency=(frequency_column, 'count'),
        # Monetary=(monetary_column, 'sum'),
        Monetary=(monetary_column, lambda x: x.abs().sum()),
        Severity=(severity_column, 'mean'),
        Fraud_Rate=(fraud_result_column, lambda x: x.sum() / x.count())
    ).reset_index()
    
     # Compute rolling fraud rate (only past transactions)
    fraud_rates = data.groupby(customer_column).apply(
        lambda group: group.assign(
            Fraud_Rate=group[fraud_result_column].expanding().mean().shift(1)  # Shift to avoid leakage
        )
    )[['Fraud_Rate']].dropna()

    # Merge fraud rates with RFMS
    print("rfms_data", data.columns, rfms_data.columns)
    rfms_data = rfms_data.merge(fraud_rates, on=customer_column, how='left').fillna(0)
    print("rfms_data", data.columns, rfms_data.columns)

    # merging
    data_processed = data.merge(rfms_data, on=customer_column, how='left')
    print("rfms_data", data.columns, rfms_data.columns)

    logger.debug(f"RFMS Metrics calculated with shape: {rfms_data.shape}")
    return data_processed

