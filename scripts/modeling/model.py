import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from script.model import X_train
from scripts.modeling.rfms_woe import WOETransformer

# Setup logger
sys.path.append(os.path.join(os.path.abspath(__file__), '..', '..', '..'))
from scripts.utils.logger import setup_logger

logger = setup_logger("model")

class CreditScoringModel:
    def __init__(self, data: pd.DataFrame, target_column: str):
        """
        Initialize the model pipeline.
        :param data: Input dataframe with features and target.
        :param target_column: Name of the target variable.
        """
        self.data = data
        self.target_column = target_column
        self.models = {
            'logistic_regression': LogisticRegression(),
            'random_forest': RandomForestClassifier(),
            'gradient_boosting': GradientBoostingClassifier(),
            'decision_tree': DecisionTreeClassifier()
        }
        self.best_model = None

    def preprocess_data(self):
        """Handles missing values, scales data, and splits into train-validation sets."""
        logger.info("Preprocessing data...")
        
        # Drop rows with missing target
        self.data = self.data.dropna(subset=[self.target_column])

        # Identify numerical and categorical features
        num_features = self.data.select_dtypes(include=[np.number]).columns.tolist()
        num_features.remove(self.target_column)

        # Fill missing values for numerical columns
        self.data[num_features] = self.data[num_features].fillna(self.data[num_features].median())

        # Standardize numerical features
        # scaler = StandardScaler()
        # self.data[num_features] = scaler.fit_transform(self.data[num_features])
        # return 

    def split_data(self):

        # Split into features (X) and target (y)
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]

        # Split into training and validating data
        X_train_, X_test, y_train_, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train_, y_train_, test_size=0.2, random_state=42)
        
        logger.info(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_models(self, X_train, y_train):
        """Train multiple models and store them."""
        trained_models = {}
        logger.info("Training models...")
        
        for model_name, model in self.models.items():
            model.fit(X_train, y_train)
            trained_models[model_name] = model
            logger.info(f"Trained {model_name} model.")

        return trained_models

    def evaluate_model(self, model, X_val, y_val):
        """Evaluate a model on validation data."""
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else y_pred

        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred),
            'recall': recall_score(y_val, y_pred),
            'f1_score': f1_score(y_val, y_pred),
            'roc_auc': roc_auc_score(y_val, y_proba)
        }

        return metrics

    def hyperparameter_tuning(self, model_name, X_train, y_train):
        """Perform hyperparameter tuning using GridSearch or RandomizedSearch."""
        logger.info(f"Hyperparameter tuning for {model_name}...")

        param_grids = {
            'logistic_regression': {'C': [0.001, 0.01, 0.1, 1, 10, 100]},
            'random_forest': {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, None]},
            'gradient_boosting': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1, 0.2]},
            'decision_tree': {'max_depth': [5, 10, 20], 'criterion': ['gini', 'entropy']}
        }

        model = self.models[model_name]
        search = GridSearchCV(model, param_grids[model_name], scoring='roc_auc', cv=5, n_jobs=-1)
        search.fit(X_train, y_train)

        logger.info(f"Best parameters for {model_name}: {search.best_params_}")

        return search.best_estimator_

    def run_pipeline(self):
        """Complete pipeline from preprocessing to training and evaluation."""
        
        self.preprocess_data()
                
        # # Create pipeline
        # pipeline = Pipeline([
        #     ('woe', WOETransformer(max_bins=5)),
        #     ('model', LogisticRegression())
        # ])

        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data()
        
        trained_models = self.train_models(X_train, y_train)
        best_auc = 0
        
        for model_name, model in trained_models.items():
            metrics = self.evaluate_model(model, X_val, y_val)
            logger.info(f"Metrics for {model_name}: {metrics}")

            # Hyperparameter tuning
            tuned_model = self.hyperparameter_tuning(model_name, X_train, y_train)
            tuned_metrics = self.evaluate_model(tuned_model, X_val, y_val)
            logger.info(f"Tuned Metrics for {model_name}: {tuned_metrics}")

            # Track best model
            if tuned_metrics['roc_auc'] > best_auc:
                best_auc = tuned_metrics['roc_auc']
                self.best_model = tuned_model
        
        # Save the best model
        if self.best_model:
            joblib.dump(self.best_model, 'best_credit_scoring_model.pkl')
            logger.info("Best model saved.")

        return self.best_model, [X_test, y_test]

def prepare_for_modelling(data, cluster_label, drop_columns):

    data = data.copy()
    data = data.drop(columns=drop_columns)
    data[cluster_label] = data[cluster_label].apply(lambda x: 1 if x == 'Good' else 0)

    return data