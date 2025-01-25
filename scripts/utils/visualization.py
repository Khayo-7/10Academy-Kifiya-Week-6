import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Setup logger for data_loader
sys.path.append(os.path.join(os.path.abspath(__file__), '..', '..', '..'))
from scripts.utils.logger import setup_logger

logger = setup_logger("visualize")

def plot_boxplot(data, x, y, title, xlabel, ylabel, palette='pastel'):
    sns.boxplot(data=data, x=x, y=y, hue=x, palette=palette)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def plot_scatterplot(data, x, y, hue, title, xlabel, ylabel, palette='Set1'):
    sns.scatterplot(data=data, x=x, y=y, hue=hue, palette=palette)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(title=hue)
    plt.show()

def plot_histogram(data, column, title, xlabel, ylabel, kde=True, bins=20, color='blue'):
    sns.histplot(data[column], kde=kde, bins=bins, color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def plot_pairplot(data, hue, vars=None):
    sns.pairplot(data, hue=hue, vars=vars)
    plt.show()

def plot_numerical_distributions(data: pd.DataFrame, numerical_columns: list, output_dir: str) -> None:
    for col in numerical_columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(data[col], kde=True, bins=30, color="blue")
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = f"{output_dir}/{col}_distribution.png"
            plt.savefig(output_path)
            logger.info(f"Distribution plot saved for {col} at {output_path}")
        plt.show()

def plot_categorical_distributions(data: pd.DataFrame, categorical_columns: list, output_dir: str) -> None:
    for col in categorical_columns:
        plt.figure(figsize=(10, 6))
        sns.countplot(y=col, data=data, order=data[col].value_counts().index, palette="viridis")
        plt.title(f"Distribution of {col}")
        plt.ylabel(col)
        plt.xlabel("Count")
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = f"{output_dir}/{col}_distribution.png"
            plt.savefig(output_path)
            logger.info(f"Distribution plot saved for {col} at {output_path}")
        plt.show()

def plot_correlation_matrix(data: pd.DataFrame, output_dir: str) -> None:
    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
    corr_matrix = data[numerical_cols].corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "missing_values_plot.png")
        plt.savefig(output_path)
        logger.info(f"Correlation matrix saved to {output_path}")
    plt.show()

def visualize_missing_values(data: pd.DataFrame, output_dir: str = None) -> None:
    missing_values = data.isnull().mean().sort_values(ascending=False) * 100
    missing_values = missing_values[missing_values > 0]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=missing_values.values, y=missing_values.index, palette="viridis")
    plt.title("Percentage of Missing Values")
    plt.xlabel("Missing Percentage")
    plt.ylabel("Columns")
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "missing_values_plot.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Missing values plot saved to {output_path}")
    
    plt.show()    

def detect_outliers(data: pd.DataFrame, numerical_columns: list, output_dir: str) -> None:
    for col in numerical_columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=data, x=col, color="orange")
        plt.title(f"Box Plot for {col}")
        plt.xlabel(col)
        plt.ylabel("Values")
        plt.tight_layout()

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{col}_boxplot.png")
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info(f"Outlier detection plot for column {col} saved to {output_path}")
        
        plt.show()
