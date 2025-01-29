import os
import sys
import numpy as np
import pandas as pd
from xverse.transformer import WOE
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, TransformerMixin
# from joblib import Parallel, delayed

# Setup logger
sys.path.append(os.path.join(os.path.abspath(__file__), '..', '..', '..'))
from scripts.utils.logger import setup_logger
from scripts.data_utils.loaders import save_data
from scripts.data_utils.cleaner import handle_outliers
from scripts.modeling.clustering import classify_users
from scripts.utils.visualization import visualize_clusters, plot_box
from scripts.data_utils.feature_engineering import calculate_rfms, normalize_standardize_numerical_features

logger = setup_logger("rfms_woe_binning")

class WOETransformer(BaseEstimator, TransformerMixin):
    """Custom WoE transformer for binning and encoding."""

    def __init__(self, max_bins=5, handle_unseen='default'):
        """
        Initialize the transformer.

        Parameters:
        - max_bins (int): Maximum number of bins for numeric features.
        - handle_unseen (str): Strategy for handling unseen categories during transform.
                              Options: 'default' (map to default WoE), 'error' (raise error).
        """
        self.max_bins = max_bins
        self.handle_unseen = handle_unseen
        self.woe_maps = {}
        self.default_woe = 0

    def fit(self, X, y):
        """
        Fit the transformer using input features (X) and a target variable (y).

        Parameters:
        - X (pd.DataFrame): Input features.
        - y (pd.Series): Target variable.

        Returns:
        - self: Fitted transformer.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame.")

        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                # Bin numeric features using DecisionTreeClassifier
                dt = DecisionTreeClassifier(max_leaf_nodes=self.max_bins, random_state=42)
                bins = dt.fit(X[[col]], y).apply(X[[col]])
                woe_stats = self._calculate_woe_iv(X[col], bins, y)
            else:
                # Treat non-numeric features as categorical
                woe_stats = self._calculate_woe_iv(X[col], X[col], y)

            # Store WoE mappings
            self.woe_maps[col] = woe_stats.set_index('Bin')['WoE'].to_dict()

        return self

    def transform(self, X):
        """
        Transform input features using pre-computed WoE mappings.

        Parameters:
        - X (pd.DataFrame): Input features.

        Returns:
        - X_woe (pd.DataFrame): Transformed features with WoE encoding.
        """
        if not self.woe_maps:
            raise ValueError("Transformer must be fitted before calling transform.")

        X_woe = X.copy()
        for col in X.columns:
            if col in self.woe_maps:
                if self.handle_unseen == 'default':
                    X_woe[col] = X[col].map(self.woe_maps[col]).fillna(self.default_woe)
                elif self.handle_unseen == 'error':
                    unseen_categories = set(X[col].unique()) - set(self.woe_maps[col].keys())
                    if unseen_categories:
                        raise ValueError(f"Unseen categories found in column {col}: {unseen_categories}")
                    X_woe[col] = X[col].map(self.woe_maps[col])
                else:
                    raise ValueError("Invalid value for handle_unseen. Use 'default' or 'error'.")

        return X_woe

    @staticmethod
    def _calculate_woe_iv(feature, bins, target):
        """
        Calculate WoE and IV for a feature.

        Parameters:
        - feature (pd.Series): Input feature.
        - bins (pd.Series): Binned feature values.
        - target (pd.Series): Target variable.

        Returns:
        - woe_data (pd.DataFrame): DataFrame containing WoE and IV statistics.
        """
        stats = []
        for bin_val in np.unique(bins):
            bin_data = (bins == bin_val)
            good = ((target == 0) & bin_data).sum()
            bad = ((target == 1) & bin_data).sum()
            good_prop = good / ((target == 0).sum() + 1e-9)
            bad_prop = bad / ((target == 1).sum() + 1e-9)
            woe = np.log((good_prop + 1e-9) / (bad_prop + 1e-9))
            iv = (good_prop - bad_prop) * woe
            stats.append([bin_val, good, bad, woe, iv])

        woe_data = pd.DataFrame(stats, columns=['Bin', 'Good', 'Bad', 'WoE', 'IV'])
        logger.info(f"WoE binning and IV calculation completed for {feature.name}.")
        return woe_data

def woe_encode(data: pd.DataFrame, target: str, categorical_columns: list) -> pd.DataFrame:
    """ Applies Weight of Evidence (WoE) encoding to categorical columns."""
    
    woe_encoder = WOE()
    data = data.copy()
    woe_data = woe_encoder.fit_transform(data[categorical_columns], data[target])
    data = pd.concat([data, woe_data], axis=1)
    logger.info("WoE encoding applied.")
    
    return data

def calculate_default_rate(data: pd.DataFrame, target: str) -> float:
    """ Calculates the default rate for a binary target column and optionally saves it. """
    
    default_rate = data[target].mean()    
    logger.info(f"Default rate for target ({target}): {default_rate*100:.2f}%")
    
    return default_rate

def normalize_rfms(rfms: pd.DataFrame, customer_column: str, mode='standard', output_path=None) -> pd.DataFrame:
    """Normalize RFMS values."""
    logger.info("Normalizing RFMS values.")

    numeric_columns = rfms.select_dtypes(include=np.number).columns.difference([customer_column]).tolist()
    
    # Separate numeric and non-numeric columns
    # normalized_numeric = scaler.fit_transform(rfms[numeric_columns])

    normalized_rfms = normalize_standardize_numerical_features(rfms, numeric_columns, mode=mode, output_path=output_path)
    
    print(normalized_rfms.columns, numeric_columns)

    logger.info("RFMS normalization completed.")
    return normalized_rfms

def woe_binning(data, target_column, columns, max_bins=5, output_dir=None):
    """Perform WoE binning dynamically."""
    
    logger.info("Applying WoE binning.")

    woe_transformer = WOETransformer(max_bins=max_bins, handle_unseen='default')
    transformed_data = woe_transformer.fit_transform(data[columns], data[target_column])
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        transformed_data.to_csv(os.path.join(output_dir, "woe_encoded_columns.csv"), index=False)
        logger.info(f"WoE encoded features saved to {output_dir}")
    
    data[columns] = transformed_data
    
    return data

def woe_rfms_pipeline(data, customer_column, recency_column, frequency_column, monetary_column, severity_column, target_column, label_column, columns, rfms_features, output_dir=None):
    
    """Execute the RFMS and WoE pipeline dynamically."""

    # Calculate RFMS scores
    rfms = calculate_rfms(data, customer_column, recency_column, frequency_column, monetary_column, severity_column)

    rfms_outliers = handle_outliers(rfms, rfms_features)
    plot_box(rfms_outliers, rfms_features)
    rfms_normalized = normalize_rfms(rfms_outliers, customer_column)
    plot_box(rfms_normalized, rfms_features)
    rfms_normalized_outlier = handle_outliers(rfms_normalized, rfms_features)
    plot_box(rfms_normalized_outlier, rfms_features)

    # merging
    data_processed = data.merge(rfms_normalized_outlier, on=customer_column, how='left')

    # encoding with woe binning
    data_woe_encoded = woe_binning(data_processed, target_column, columns, max_bins=5, output_dir=output_dir)

    # User Classification
    score_column = rfms_features[2]
    data_rfms_classified = classify_users(data_woe_encoded, rfms_features, score_column=score_column, label_column=label_column)
    
    cluster_feature = rfms_features[0]
    visualize_clusters(data_rfms_classified, cluster_feature, score_column, label_column, output_dir=output_dir)
    
    if output_dir:
        # Save the final transformed dataset
        os.makedirs(output_dir, exist_ok=True)
        save_data(data_rfms_classified, os.path.join(output_dir, 'data_rfms_classified.csv'))
        logger.info(f"Classified RFMS data saved to {output_dir}")

    return data_rfms_classified, data_woe_encoded