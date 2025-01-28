import os
import sys
import pytz
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from joblib import Parallel, delayed

# Setup logger
sys.path.append(os.path.join(os.path.abspath(__file__), '..', '..', '..'))
from scripts.utils.logger import setup_logger
from scripts.utils.visualization import plot_box, plot_optimal_k, visualize_clusters, visualize_pca_clusters

logger = setup_logger("woe_binning")

class WOETransformer(BaseEstimator, TransformerMixin):
    """Custom WoE transformer for binning and encoding."""

    def __init__(self, max_bins=5):
        self.max_bins = max_bins
        self.woe_maps = {}

    def fit(self, X, y):
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                dt = DecisionTreeClassifier(max_leaf_nodes=self.max_bins, random_state=42)
                bins = dt.fit(X[[col]], y).apply(X[[col]])
                woe_stats = self._calculate_woe_iv(X[col], bins, y)
            else:
                woe_stats = self._calculate_woe_iv(X[col], X[col], y)
            self.woe_maps[col] = woe_stats.set_index('Bin')['WoE'].to_dict()
        return self

    def transform(self, X):
        X = X.copy()
        for col, woe_map in self.woe_maps.items():
            X[f'{col}_WoE'] = X[col].map(woe_map)
        return X

    @staticmethod
    def _calculate_woe_iv(feature, bins, target):
        logger.info(f"Binning {feature.name} and calculating WoE/IV.")
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
        woe_data['IV_Sum'] = woe_data['IV'].sum()
        logger.info(f"WoE binning and IV calculation completed for {feature.name}.")
        return woe_data

def calculate_rfms(data, customer_column, recency_column, frequency_column, monetary_column, severity_column, reference_datetime=None):
    """Calculate RFMS scores dynamically."""

    logger.info("Calculating RFMS scores with recency in seconds.")
    reference_datetime = reference_datetime or datetime.now()
    reference_datetime = reference_datetime.replace(tzinfo=None) if reference_datetime.tzinfo else reference_datetime
    
    # Calculate recency in seconds
    data[recency_column] = validate_datetime_column(data[recency_column])
    data['Recency_Seconds'] = (reference_datetime - data[recency_column]).dt.total_seconds()

    # Group by customer and calculate RFMS scores
    reference_date = data[recency_column].max()
    rfms = data.groupby(customer_column).agg(
        Recency=(recency_column, lambda x: (reference_date - x.max()).days),
        # Recency=('Recency_Seconds': 'min'),
        Frequency=(recency_column, 'count'),
        Monetary=(monetary_column, 'sum'),
        Severity=(severity_column, 'mean')
    ).reset_index()

    logger.debug(f"RFMS Metrics calculated with shape: {rfms.shape}")
    return rfms

def normalize_rfms(rfms: pd.DataFrame, customer_column: str, mode='standard') -> pd.DataFrame:
    """Normalize RFMS values."""
    logger.info("Normalizing RFMS values.")

    numeric_columns = rfms.select_dtypes(include=np.number).columns.difference([customer_column]).tolist()
    scaler = StandardScaler() if mode == 'standard' else MinMaxScaler()
    
    # Separate numeric and non-numeric columns
    normalized_numeric = scaler.fit_transform(rfms[numeric_columns])
    normalized_rfms = rfms[[customer_column]].copy()
    normalized_rfms[numeric_columns] = normalized_numeric

    logger.info("RFMS normalization completed.")
    return normalized_rfms

def classify_users(data, recency_column, frequency_column, monetary_column, severity_column, score_column=None, label_column='Customer_Label', scoring_method="threshold", threshold=0.5, num_clusters=2, output_dir=None):
    """Classify users into Good/Bad based on RFMS scores."""
    logger.info(f"Classifying customers using {scoring_method} method.")
    if score_column and scoring_method == "threshold":
        data[label_column] = np.where(data[score_column] >= threshold, "Good", "Bad")
    elif scoring_method == "kmeans":
        logger.info(f"Clustering RFMS data into {num_clusters} clusters.")
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        data['Cluster'] = kmeans.fit_predict(data[[recency_column, frequency_column, monetary_column, severity_column]])
        visualize_pca_clusters(data[[recency_column, frequency_column, monetary_column, severity_column]], data['Cluster'], output_dir)
        cluster_means = data.groupby('Cluster')[monetary_column].mean()
        good_cluster = cluster_means.idxmax()
        data[label_column] = data['Cluster'].apply(lambda x: "Good" if x == good_cluster else "Bad")
    else:
        raise ValueError(f"Unsupported scoring method: {scoring_method}")
    logger.info(f"Customer labels assigned to column: {label_column}.")
    return data

def woe_binning(data, target_column, feature_columns, max_bins=5, output_dir=None):
    """Perform WoE binning dynamically."""
    logger.info("Applying WoE binning.")
    woe_transformer = WOETransformer(max_bins=max_bins)
    transformed_data = woe_transformer.fit_transform(data[feature_columns], data[target_column])
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        transformed_data.to_csv(os.path.join(output_dir, "woe_encoded_features.csv"), index=False)
        logger.info(f"WoE encoded features saved to {output_dir}")
    return transformed_data

def calculate_default_rate(data: pd.DataFrame, target: str) -> float:
    """ Calculates the default rate for a binary target column and optionally saves it. """
    default_rate = data[target].mean()    
    logger.info(f"Default rate for target ({target}): {default_rate*100:.2f}%")
    return default_rate

def validate_datetime_column(column, timezone="Africa/Addis_Ababa"):
    
    if not pd.api.types.is_datetime64_any_dtype(column):
        column = pd.to_datetime(column, errors='coerce')
    column = (
        column.dt.tz_localize(None)#.dt.tz_localize(pytz.UTC).dt.tz_convert(timezone) 
        if column.dt.tz is not None
        else column
    )
    return column
    
def handle_outliers(data, columns, lower_quantile=0.01, upper_quantile=0.99):
    for col in columns:
        data[col] = data[col].clip(
            lower=data[col].quantile(lower_quantile),
            upper=data[col].quantile(upper_quantile)
        )
    return data

def search_optimal_k(data, min_k=2, max_k=10):
    best_k = min_k
    best_score = -1
    scores = []
    for k in range(min_k, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42).fit(data)
        score = silhouette_score(data, kmeans.labels_)
        scores.append(score)
        logger.debug(f"Silhouette score for k={k}: {score}")
        
        if score > best_score:
            best_score = score
            best_k = k
    logger.info(f"Optimal k determined: {best_k} with silhouette score: {best_score}")
    
    plot_optimal_k(scores)
    return best_k

def cluster_summary(data, cluster_column):
    return data.groupby(cluster_column).agg({
        "Recency": ["mean", "median"],
        "Frequency": ["mean", "sum"],
        "Monetary": ["mean", "sum"],
        "Severity": ["mean"]
    })

def woe_rfms_pipeline(data, customer_column, recency_column, frequency_column, monetary_column, severity_column, target_column, feature_columns, output_dir=None):
    """Execute the RFMS and WoE pipeline dynamically."""
    columns = ['Recency', 'Frequency', 'Monetary', 'Severity']
    rfms = calculate_rfms(data, customer_column, recency_column, frequency_column, monetary_column, severity_column)
    rfms_normalized = normalize_rfms(rfms, customer_column)
    plot_box(rfms_normalized, columns)
    rfms_normalized_2 = handle_outliers(rfms_normalized, columns)
    plot_box(rfms_normalized_2, columns)
    optimal_k = search_optimal_k(rfms_normalized[columns])
    rfms_classified = classify_users(rfms_normalized_2, "Recency", "Frequency", "Monetary", "Severity", score_column="Monetary", label_column="RFMS_Label", scoring_method="kmeans", threshold=0.5, num_clusters=optimal_k)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        rfms_classified.to_csv(os.path.join(output_dir, "rfms_classified.csv"), index=False)
        logger.info(f"Classified RFMS data saved to {output_dir}")

    print('cluster_summary', cluster_summary(rfms_classified, customer_column))
    visualize_clusters(rfms_classified, "Recency", "Monetary", "RFMS_Label", output_dir=output_dir)
    woe_encoded_features = woe_binning(data, target_column, feature_columns, max_bins=5, output_dir=output_dir)
    
    return rfms_classified, woe_encoded_features