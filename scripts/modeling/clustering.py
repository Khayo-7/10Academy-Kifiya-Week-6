import os
import sys
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Setup logger
sys.path.append(os.path.join(os.path.abspath(__file__), '..', '..', '..'))
from scripts.utils.logger import setup_logger
from scripts.utils.visualization import plot_optimal_k, visualize_pca_clusters

logger = setup_logger("clustering")

def search_optimal_k(data, min_k=2, max_k=10):
    """
    Perform K-Means clustering for a range of values and return the optimal number of clusters.
    """
    best_k = min_k
    best_score = -1
    scores = []
    for k in range(min_k, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(data)
        score = silhouette_score(data, labels)
        scores.append(score)
        logger.debug(f"Silhouette score for k={k}: {score}")
        
        if score > best_score:
            best_score = score
            best_k = k

    logger.info(f"Optimal k determined: {best_k} with silhouette score: {best_score}")
    
    plot_optimal_k(scores)
    return best_k

def cluster_users(data, n_clusters=None):
    """Classify users into Good/Bad based on RFMS featuress."""
    
    logger.info(f"Clustering customers using kmeans method.")
    if n_clusters is None:
        n_clusters = search_optimal_k(data)

    logger.info(f"Started clustering RFMS data into {n_clusters} clusters.")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(data)
    logger.info(f"Finished clustering RFMS data into {n_clusters} clusters.")
    return clusters

def classify_users(data, columns, score_column, label_column='Customer_Label', cluster_col='Cluster', n_clusters=None, output_dir=None):
    
    logger.info(f"Classifying customers.")
        
    data[cluster_col] = cluster_users(data[columns], n_clusters=n_clusters)

    visualize_pca_clusters(data[columns], data[cluster_col], output_dir)

    logger.info(f"Customer labels assigned to column: {label_column}.")

    good_cluster = data.groupby(cluster_col)[score_column].mean().idxmax()
    data[label_column] = data[cluster_col].apply(lambda x: "Good" if x == good_cluster else "Bad")
    
    logger.info(f"Customer labels assigned to column: {label_column}.")
    return data