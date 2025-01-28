import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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

def plot_outliers(data: pd.DataFrame, numerical_columns: list, output_dir: str) -> None:
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


def plot_box(data, columns):
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=data[columns])
    plt.title("Boxplot of RFMS Features")
    plt.show()

def visualize_clusters(data, x_col, y_col, label_col, output_dir=None):
    """Visualize clustering dynamically."""
    logger.info("Generating cluster visualization.")
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=data[x_col], y=data[y_col], hue=data[label_col], palette="viridis")
    plt.title("Cluster Visualization")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend(title=label_col)
    plt.tight_layout()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "clusters.png")
        plt.savefig(output_path)
        logger.info(f"Cluster visualization saved to {output_path}.")
    plt.show()

def visualize_rfms_scores(data, cluster_col, output_dir=None):
    """Visualize RFMS scores dynamically by cluster."""
    logger.info("Visualizing RFMS scores by cluster.")
    plt.figure(figsize=(10, 6))
    for cluster in data[cluster_col].unique():
        cluster_data = data[data[cluster_col] == cluster]
        plt.scatter(range(len(cluster_data)), cluster_data.index, alpha=0.6, label=f"Cluster {cluster}")
    plt.xlabel("Data Point Index")
    plt.ylabel("RFMS Scores")
    plt.title("RFMS Cluster Distribution")
    plt.legend()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "rfms_cluster_distribution.png")
        plt.savefig(output_path)
        logger.info(f"RFMS cluster scores visualization saved to {output_path}.")
    plt.show()

def plot_optimal_k(scores, min_k=2, max_k=10):
    plt.plot(range(min_k, max_k+1), scores, marker='o')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Elbow Method for Optimal k")
    plt.show()

def visualize_pca_clusters(data, labels, output_dir=None):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(data)
    reduced_df = pd.DataFrame(reduced, columns=['PCA1', 'PCA2'])
    reduced_df['Cluster'] = labels

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=reduced_df, x='PCA1', y='PCA2', hue='Cluster', palette='viridis', alpha=0.6)
    plt.title("Cluster Visualization via PCA")
    if output_dir:
        plt.savefig(os.path.join(output_dir, "pca_clusters.png"))
    plt.show()
