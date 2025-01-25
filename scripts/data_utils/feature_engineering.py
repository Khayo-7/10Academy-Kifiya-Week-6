import os
import sys
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler

# Setup logger for data_loader
sys.path.append(os.path.join(os.path.abspath(__file__), '..', '..', '..'))
from scripts.utils.logger import setup_logger

logger = setup_logger("feature_engineering")

def create_aggregate_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Creates aggregate features for each customer.

    Args:
        data (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with aggregate features added.
    """
    agg_features = data.groupby("CustomerId").agg(
        total_transaction_amount=("Amount", "sum"),
        avg_transaction_amount=("Amount", "mean"),
        transaction_count=("Amount", "count"),
        std_transaction_amount=("Amount", "std")
    ).reset_index()
    
    data = data.merge(agg_features, on="CustomerId", how="left")
    logger.info("Aggregate features created and merged into the dataframe.")
    return data

def extract_temporal_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts temporal features from the TransactionStartTime column.

    Args:
        data (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with temporal features added.
    """
    data['TransactionStartTime'] = pd.to_datetime(data['TransactionStartTime'])
    data['transaction_hour'] = data['TransactionStartTime'].dt.hour
    data['transaction_day'] = data['TransactionStartTime'].dt.day
    data['transaction_month'] = data['TransactionStartTime'].dt.month
    data['transaction_year'] = data['TransactionStartTime'].dt.year
    logger.info("Temporal features extracted.")
    return data

def encode_categorical_variables(data: pd.DataFrame, one_hot_cols: list, label_cols: list) -> pd.DataFrame:
    """
    Encodes categorical variables using one-hot and label encoding.

    Args:
        data (pd.DataFrame): The input DataFrame.
        one_hot_cols (list): List of columns to one-hot encode.
        label_cols (list): List of columns to label encode.

    Returns:
        pd.DataFrame: DataFrame with encoded categorical features.
    """
    # One-Hot Encoding
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded_data = pd.DataFrame(encoder.fit_transform(data[one_hot_cols]),
                              columns=encoder.get_feature_names_out(one_hot_cols))
    
    data = pd.concat([data, encoded_data], axis=1).drop(columns=one_hot_cols)
    
    # Label Encoding
    for col in label_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
    
    logger.info("Categorical variables encoded.")
    return data

def normalize_standardize(data: pd.DataFrame, columns: list, mode: str = "normalize") -> pd.DataFrame:
    """
    Normalizes or standardizes numerical features.

    Args:
        data (pd.DataFrame): The input DataFrame.
        columns (list): List of columns to scale.
        mode (str): Scaling method - "normalize" (default) or "standardize".

    Returns:
        pd.DataFrame: DataFrame with scaled features.
    """
    if mode == "normalize":
        scaler = MinMaxScaler()
    elif mode == "standardize":
        scaler = StandardScaler()
    else:
        raise ValueError("Invalid mode. Choose either 'normalize' or 'standardize'.")
    
    data[columns] = scaler.fit_transform(data[columns])
    logger.info(f"Numerical features scaled using {mode} mode.")
    return data
