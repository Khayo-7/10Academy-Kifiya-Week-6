import os
import sys
import pandas as pd
from typing import List, Dict
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

# Setup logger for data_loader
sys.path.append(os.path.join(os.path.abspath(__file__), '..', '..', '..'))
from scripts.data_utils.cleaner import validate_convert_date_column
from scripts.utils.logger import setup_logger

logger = setup_logger("data_transformers")


class PreprocessingTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        irrelevant_columns: List[str],
        missing_value_strategies: Dict[str, str],
        date_column: str,
        categorical_columns: List[str],
        numerical_columns: List[str],
        dtype_conversions: Dict[str, str],
        timezone: str = "Africa/Addis_Ababa"
    ):
        self.irrelevant_columns = irrelevant_columns
        self.missing_value_strategies = missing_value_strategies
        self.date_column = date_column
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.dtype_conversions = dtype_conversions
        self.timezone = timezone

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Ensure X is a DataFrame
        X = X.copy()

        # Drop irrelevant columns
        X = self.drop_irrelevant_columns(X, self.irrelevant_columns)

        # Handle missing values while preserving feature-target alignment
        X = self.handle_missing_values(X, self.missing_value_strategies)
        # X = self.handle_missing_values_general(X, self.missing_value_strategies)

        # Convert data types
        X = self.convert_data_types(X, self.dtype_conversions)

        # Handle dates
        X = validate_convert_date_column(X, self.date_column, self.timezone)

        # Standardize categorical data
        X = self.standardize_categorical_columns(X, self.categorical_columns)

        # Remove duplicates and align with y
        X = self.remove_duplicates(X)

        return X

    @staticmethod
    def drop_irrelevant_columns(data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        return data.drop(columns=[col for col in columns if col in data], errors="ignore")

    @staticmethod
    def handle_missing_values(data: pd.DataFrame, strategies: Dict[str, str]):
        """ Apply missing value imputation. """
        
        for col, strategy in strategies.items():
            if col in data.columns:
                imputer = SimpleImputer(strategy=strategy)
                data[col] = imputer.fit_transform(data[[col]]).ravel()

        return data

    @staticmethod
    def handle_missing_values_general(data: pd.DataFrame) -> pd.DataFrame:
        """
        Handles missing values by imputing:
        - Mode for categorical columns
        - Mean for numerical columns
        """
        missing_value_columns = data.columns[data.isnull().any()]
        for col in missing_value_columns:
            if data[col].dtype == "object" or data[col].dtype.name == "category":
                imputer = SimpleImputer(strategy="most_frequent")
            else:
                imputer = SimpleImputer(strategy="mean")
            try:
                data[col] = imputer.fit_transform(data[[col]]).ravel()
                logger.info(f"Imputed missing values in {col}")
            except Exception as e:
                logger.error(f"Error imputing column {col}: {e}")
        return data

    @staticmethod
    def remove_duplicates(data: pd.DataFrame):
        """ Remove duplicates while keeping X and y aligned. """
        data.drop_duplicates()
        return data

    @staticmethod
    def convert_data_types(data: pd.DataFrame, conversions: Dict[str, str]) -> pd.DataFrame:
        for col, dtype in conversions.items():
            if col in data.columns:
                try:
                    data[col] = data[col].astype(dtype, errors="ignore")
                except Exception as e:
                    pass  # Log error if needed
        return data 

    @staticmethod
    def standardize_categorical_columns(data: pd.DataFrame, categorical_columns: List[str]) -> pd.DataFrame:
        for col in categorical_columns:
            if col in data.columns:
                data[col] = data[col].astype(str).str.strip().str.upper()
        return data
