import os
import sys
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
# from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler


# Setup logger for data_loader
sys.path.append(os.path.join(os.path.abspath(__file__), '..', '..', '..'))
from scripts.data_utils.cleaner import validate_convert_date_column, handle_outliers
from scripts.utils.logger import setup_logger

logger = setup_logger("transformers")

# Integrated Processing Transformer
class FeatureGenerator(BaseEstimator, TransformerMixin):
    def __init__(self, customer_column: str, amount_column: str, date_column: str):
        self.customer_column = customer_column
        self.amount_column = amount_column
        # self.transaction_column = transaction_column
        self.date_column = date_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Feature engineering
        X = self.create_aggregate_features(X, self.customer_column, self.amount_column)#, self.transaction_column)
        X = self.extract_temporal_features(X, self.date_column)
        return X
        
    @staticmethod
    def create_aggregate_features(data: pd.DataFrame, customer_column, amount_column, output_path: str = None) -> pd.DataFrame:
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
        aggregated_data = data.groupby(customer_column).agg(
            total_transaction_amount=(amount_column, 'sum'),
            avg_transaction_amount=(amount_column, 'mean'),
            # transaction_count=(transaction_column, 'count'),
            std_transaction_amount=(amount_column, 'std')
        ).reset_index()

        # Handle NaN for std deviation (single/no transaction customers)
        aggregated_data['std_transaction_amount'] = aggregated_data['std_transaction_amount'].fillna(0)
        
        logger.info("Aggregate features created successfully.")
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            aggregated_data.to_csv(output_path, index=False)
            logger.info(f"Aggregate features saved to {output_path}")
        
        data = data.merge(aggregated_data, on=customer_column, how="left")
        logger.info("Aggregate features created and merged into the dataframe.")
        return data

    @staticmethod
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

class EncodeCategoricalVariables(BaseEstimator, TransformerMixin):

    def __init__(self, one_hot_columns, label_columns):
        self.one_hot_columns = one_hot_columns
        self.label_columns = label_columns
        # self.one_hot_encoder = None
        self.one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.label_encoders = {}

    def fit(self, X, y):
        
        if self.one_hot_columns:
            logger.info("Performing One-Hot Encoding.")
            # self.one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            encoded_X = pd.DataFrame(
                self.one_hot_encoder.fit(X[self.one_hot_columns]),
                columns=self.one_hot_encoder.get_feature_names_out(self.one_hot_columns),
                index=X.index,
            )
            X = pd.concat([X.drop(columns=self.one_hot_columns), encoded_X], axis=1)

        # Label Encoding
        if self.label_columns:
            logger.info("Performing Label Encoding.")
            for col in self.label_columns:
                le = LabelEncoder()
                X[col] = le.fit(X[col].astype(str))
                self.label_encoders[col] = le.classes_.tolist()
                
        return X

    def transform(self, X):
        if self.one_hot_columns:
            encoded_X = pd.DataFrame(
                self.one_hot_encoder.transform(X[self.one_hot_columns]),
                columns=self.one_hot_encoder.get_feature_names_out(self.one_hot_columns),
                index=X.index,
            )
            X = pd.concat([X.drop(columns=self.one_hot_columns), encoded_X], axis=1)

        if self.label_columns:
            for col in self.label_columns:
                le = self.label_encoders[col]
                X[col] = le.transform(X[col].astype(str))
        logger.info("Categorical variables encoded successfully.") 

        return X
    
class ScaleNumericalFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, columns, mode='standard'):
        self.mode = mode
        self.columns = columns
        self.scaler = StandardScaler() if self.mode == 'standard' else MinMaxScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns])
        return self

    def transform(self, X):
        logger.info(f"Scaling numerical features using {self.mode} method.")
        X[self.columns] = self.scaler.transform(X[self.columns])
        logger.info(f"Numerical features scaled using {self.mode} mode.")
        return X

class RFMSFeatureGenerator(BaseEstimator, TransformerMixin):
    def __init__(self, customer_column, recency_column, frequency_column, monetary_column, severity_column, rfms_features, timezone):
        self.customer_column = customer_column
        self.recency_column = recency_column
        self.frequency_column = frequency_column
        self.monetary_column = monetary_column
        self.severity_column = severity_column
        self.rfms_features = rfms_features
        self.fraud_rates = {} # Stores past fraud rates per customer
        self.timezone = timezone

    def fit(self, X, y=None):

        # data = pd.concat([X, y], axis=1)
        X = X.copy()
        X = validate_convert_date_column(X, self.recency_column, self.timezone)
        X.sort_values(by=[self.customer_column, self.recency_column], inplace=True)
        
        # Compute rolling fraud rate (only past transactions)
        fraud_rates = (
            X.groupby(self.customer_column)[self.severity_column]
            .apply(lambda x: x.shift(1).expanding().mean())   # Shift to avoid leakage
            .reset_index()
            .rename(columns={self.severity_column: 'Severity'})
        )

        # Store last known fraud rate per customer for inference
        self.fraud_rates = fraud_rates.groupby(self.customer_column)['Severity'].last().to_dict()

        return self

    def transform(self, X):
        """Calculate RFMS scores dynamically."""

        logger.info("Calculating RFMS scores with recency.")

        X = X.copy()

        # Group by customer and calculate RFMS scores
        X = validate_convert_date_column(X, self.recency_column, self.timezone)
        reference_date = X[self.recency_column].max()
        X['Recency_days'] = (reference_date - X[self.recency_column]).dt.days

        rfms_data = X.groupby(self.customer_column).agg(
            Recency=('Recency_days', 'min'),
            Frequency=(self.frequency_column, 'count'),
            Monetary=(self.monetary_column, 'sum'),
            Intensity=(self.monetary_column, 'mean'),
            Volatility=(self.monetary_column, 'std')
        ).reset_index()

        # Assign past fraud rates to customers
        rfms_data['Severity'] = rfms_data[self.customer_column].map(self.fraud_rates)

        # Handle missing fraud rates for new customers (default to 0)
        # global_fraud_rate = 0 # Innocent until proven guiltyyy
        global_fraud_rate = sum(self.fraud_rates.values()) / len(self.fraud_rates) if self.fraud_rates else 0 # Humasn are potentially fraud
        rfms_data['Severity'] = rfms_data['Severity'].fillna(global_fraud_rate)

        # Merge RFMS features back into the original data
        X.drop(columns=['Recency_days'], inplace=True)
        X = X.merge(rfms_data, on=self.customer_column, how='left')

        # Handle NaNs for new customers
        X['Severity'] = X['Severity'].fillna(0)
        X['Volatility'] = X['Volatility'].fillna(0)

        # Handle outliers in RFMS Features
        X = handle_outliers(X, self.rfms_features, lower_quantile=0.01, upper_quantile=0.99)

        logger.debug(f"RFMS Metrics calculated with shape: {rfms_data.shape}")
        return X

class WOETransformer(BaseEstimator, TransformerMixin):
    def __init__(self, target_column, columns, max_bins=5, handle_unseen='default'):
        self.target_column = target_column
        self.columns = columns
        self.max_bins = max_bins
        self.handle_unseen = handle_unseen
        self.woe_maps = {}
        self.default_woe = 0

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame.") 

        for col in self.columns:

            if col not in X.columns:
                continue

            if pd.api.types.is_numeric_dtype(X[col]):
                # Use DecisionTreeClassifier to create bins
                dt = DecisionTreeClassifier(max_leaf_nodes=self.max_bins, random_state=42)
                dt.fit(X[[col]], X[self.target_column])
                bins = dt.apply(X[[col]])  # Get leaf indices (bins)
                
                # Ensure bins are mapped correctly
                bin_map = pd.DataFrame({col: X[col], "Bin": bins.flatten()})  
                woe_stats = self._calculate_woe_iv(bin_map["Bin"], bin_map["Bin"], X[self.target_column])
            else:
                # Treat non-numeric features as categorical
                woe_stats = self._calculate_woe_iv(X[col], X[col], X[self.target_column])

            # Store WoE mappings
            self.woe_maps[col] = woe_stats.set_index('Bin')['WoE'].to_dict()

        return self

    def transform(self, X):

        if not self.woe_maps:
            raise ValueError("Transformer must be fitted before calling transform.")

        X_woe = X.copy()
        for col in self.columns:

            if col not in X.columns:
                continue

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
        stats = []
        unique_bins = np.unique(bins)

        for bin_val in unique_bins:
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
    
class UserClusterClassifier(BaseEstimator, TransformerMixin):
 
    def __init__(
        self,
        columns: list,
        score_column: str,
        customer_label: str = 'Customer_Label',
        cluster_column: str = 'Cluster',
        min_k: int = 2,
        max_k: int = 10,
        n_clusters: int = None
    ):
        self.columns = columns
        self.score_column = score_column
        self.customer_label = customer_label
        self.cluster_column = cluster_column
        self.min_k = min_k
        self.max_k = max_k
        self.n_clusters = n_clusters
        self.kmeans = None

    def _search_optimal_k(self, data: pd.DataFrame) -> int:
        best_k = self.min_k
        best_score = -1
        scores = []
        for k in range(self.min_k, self.max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(data)
            score = silhouette_score(data, labels)
            scores.append(score)
            logger.debug(f"Silhouette score for k={k}: {score}")
            
            if score > best_score:
                best_score = score
                best_k = k

        logger.info(f"Optimal k determined: {best_k} with silhouette score: {best_score}")
        return best_k

    def fit(self, X: pd.DataFrame, y=None):
        logger.info("Fitting UserClusterClassifier.")
        if self.n_clusters is None:
            self.n_clusters = self._search_optimal_k(X[self.columns])
        
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.kmeans.fit(X[self.columns])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info("Transforming data with UserClusterClassifier.")
        X = X.copy()

        # Cluster users
        X[self.cluster_column] = self.kmeans.predict(X[self.columns])
        logger.info(f"Clustered data into {self.n_clusters} clusters.")

        # Classify users as Good/Bad
        good_cluster = X.groupby(self.cluster_column)[self.score_column].mean().idxmax()
        X[self.customer_label] = X[self.cluster_column].apply(lambda x: "Good" if x == good_cluster else "Bad")
        logger.info(f"Customer labels assigned to column: {self.customer_label}.")

        return X
