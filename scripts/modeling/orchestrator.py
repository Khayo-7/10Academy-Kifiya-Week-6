import os
import sys
import joblib
from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif

# Setup logger for data_loader
sys.path.append(os.path.join(os.path.abspath(__file__), '..', '..', '..'))
from script.model import X_test
from scripts.utils.logger import setup_logger
from scripts.modeling.transformers import *
from scripts.data_utils.data_transformers import *
from scripts.modeling.model import CreditScoringModel

logger = setup_logger("orchestrator")

# woe_transformer = ColumnTransformer([
# ], remainder='passthrough')

class FullPipelineModel(BaseEstimator, TransformerMixin):
    def __init__(self, transformation_pipeline, modeling_pipeline):
        """
        Initialize the full pipeline.
        """
        self.transformation_pipeline = transformation_pipeline
        self.modeling_pipeline = modeling_pipeline

    def fit(self, X, y=None):
        """
        Fit the full pipeline.
        """
        # Apply the transformation pipeline
        X_transformed = self.transformation_pipeline.fit_transform(X, y)

        # Fit the modeling pipeline
        self.modeling_pipeline.fit(X_transformed, y)
        return self

    def predict(self, X):
        """
        Make predictions using the full pipeline.
        """
        # Apply the transformation pipeline
        X_transformed = self.transformation_pipeline.transform(X)

        # Make predictions using the modeling pipeline
        return self.modeling_pipeline.predict(X_transformed)

    def save(self, path):
        """
        Save the full pipeline to disk.
        """
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        """
        Load the full pipeline from disk.
        """
        return joblib.load(path)
    
class DataPreparationTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, drop_columns, customer_label):
        """
        Initialize the transformer.
        """
        self.drop_columns = drop_columns
        self.customer_label = customer_label

    def fit(self, X, y=None):
        """
        Fit the transformer (no learning required).
        """
        return self

    def transform(self, X):
        """
        Apply the data preparation steps.
        """
        # Drop unnecessary columns
        X_transformed = X.drop(columns=self.drop_columns)

        # Encode the target variable
        X_transformed[self.customer_label] = X_transformed[self.customer_label].apply(
            lambda x: 1 if x == 'Good' else 0
        )

        return X_transformed

# Define the transformation pipeline
def get_transformation_pipeline(
    irrelevant_columns, missing_value_strategies, date_column, categorical_columns, numerical_columns,
    dtype_conversions, timezone, customer_column, amount_column, numerical_features,
    recency_column, frequency_column, monetary_column, severity_column, rfms_features,
    target_column, max_bins, score_column, customer_label
):
    mode = 'standard'
    handle_unseen = 'default'
    cluster_column = 'Cluster'
    woe_columns = categorical_columns + numerical_columns + numerical_features

    transformation_pipeline = Pipeline([
        ('preprocessing', PreprocessingTransformer(
            irrelevant_columns, missing_value_strategies, date_column, categorical_columns, numerical_columns,
            dtype_conversions, timezone
        )),
        ('feature_generator', FeatureGenerator(customer_column, amount_column, date_column)),
        ('normalize_numerical', ScaleNumericalFeatures(columns=numerical_features, mode=mode)),
        ('rfms_features_generator', RFMSFeatureGenerator(customer_column, recency_column, frequency_column,
                                                        monetary_column, severity_column, rfms_features, timezone)),
        ('normalize_rfms', ScaleNumericalFeatures(columns=rfms_features, mode=mode)),
        ('woe_transformer', WOETransformer(target_column, woe_columns, max_bins=max_bins, handle_unseen=handle_unseen)),
        ('normalizer', ScaleNumericalFeatures(columns=woe_columns, mode=mode))
        ('user_classifier', UserClusterClassifier(rfms_features, score_column, customer_label, cluster_column=cluster_column)),
        ('final_scaler', ScaleNumericalFeatures(columns=woe_columns, mode=mode)),
    ])

    return transformation_pipeline

def get_modeling_pipeline(drop_columns, customer_label):
    """
    Define the modeling pipeline.
    """
    modeling_pipeline = Pipeline([
        # Prepare the data (drop columns, encode target, etc.)
        ('data_preparation', DataPreparationTransformer(drop_columns=drop_columns, customer_label=customer_label)),
        ('feature_selection', SelectKBest(score_func=f_classif, k=10)),
        # ('classifier', CreditScoringModel())
    ])

    return modeling_pipeline

