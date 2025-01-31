import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Setup logger
sys.path.append(os.path.join(os.path.abspath(__file__), '..', '..', '..'))
from scripts.utils.logger import setup_logger

logger = setup_logger("modeling")

class CreditScoringModel:
    def __init__(self, models=None, model_path=None):
        self.models = models if models else {
            'logistic_regression': LogisticRegression(),
            'random_forest': RandomForestClassifier(),
            'gradient_boosting': GradientBoostingClassifier(),
            'decision_tree': DecisionTreeClassifier()
        }
        self.best_model = None
        self.model_path = os.path.join(model_path if model_path else '', 'best_credit_scoring_model.pkl')

    def train(self, X, y):

        best_auc = 0
        best_model_name = None
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        for name, model in self.models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_val)
            auc = roc_auc_score(y_val, y_pred)
            if auc > best_auc:
                best_auc = auc
                self.best_model = model
                best_model_name = name
        
        if best_model_name:
            self.best_model = self.hyperparameter_tuning(best_model_name, X_train, y_train)

        joblib.dump(self.best_model, self.model_path)
        return self.best_model

    def predict(self, X):
        if self.best_model is None:
            self.best_model = joblib.load(self.model_path)
        return self.best_model.predict(X)

    def evaluate_model(self, model, X_val, y_val):
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
        param_grids = {
            'logistic_regression': {'C': [0.001, 0.01, 0.1, 1, 10, 100]},
            'random_forest': {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, None]},
            'gradient_boosting': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1, 0.2]},
            'decision_tree': {'max_depth': [5, 10, 20], 'criterion': ['gini', 'entropy']}
        }

        model = self.models[model_name]
        search = GridSearchCV(model, param_grids[model_name], scoring='roc_auc', cv=5, n_jobs=-1)
        search.fit(X_train, y_train)

        return search.best_estimator_

def prepare_for_modeling(data, customer_label, drop_columns):

    data = data.copy()
    data = data.drop(columns=drop_columns)
    data[customer_label] = data[customer_label].apply(lambda x: 1 if x == 'Good' else 0)

    return data