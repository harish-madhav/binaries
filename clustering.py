# 🔹 General-Purpose Clustering Framework
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import sys

def load_and_preprocess(file_path, sample_size=10000):
    df = pd.read_csv(file_path)
    df_numeric = df.select_dtypes(include=[np.number])
    if df_numeric.empty:
        raise ValueError("No numeric columns found in the dataset.")
    df_sampled = df_numeric.sample(n=min(sample_size, len(df_numeric)), random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_sampled)
    return X_scaled

def apply_pca(X_scaled, n_components=2):
    if X_scaled.shape[1] > 1:
        pca = PCA(n_components=min(n_components, X_scaled.shape[1]))
        X_pca = pca.fit_transform(X_scaled)
        return X_pca
    return X_scaled

def evaluate_model(name, labels, data):
    if len(set(labels)) <= 1:
        print(f"\n⚠️ {name} failed to form clusters properly.")
        return
    sil_score = silhouette_score(data, labels)
    db_score = davies_bouldin_score(data, labels)
    ch_score = calinski_harabasz_score(data, labels)
    unique, counts = np.unique(labels, return_counts=True)
    imbalance = max(counts) - min(counts)
    print(f"\n📌 {name} Clustering Results")
    print(f"Silhouette Score: {sil_score:.4f}")
    print(f"Davies-Bouldin Index: {db_score:.4f}")
    print(f"Calinski-Harabasz Index: {ch_score:.4f}")
    print(f"Cluster Distribution: {dict(zip(unique, counts))}")
    print(f"Bias Proxy (Imbalance): {imbalance}")

def optimal_kmeans(X_pca, k_range=(3, 6)):
    print("\n🔹 Finding Optimal K for MiniBatchKMeans...\n")
    best_k, best_score = None, -1
    for k in range(*k_range):
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=1000)
        labels = kmeans.fit_predict(X_pca)
        score = silhouette_score(X_pca, labels)
        print(f"K = {k} → Silhouette Score = {score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k
    print(f"\n✅ Best K = {best_k} with Silhouette Score = {best_score:.4f}")
    return best_k

def run_clustering_algorithms(X_pca, best_k):
    # MiniBatch KMeans
    kmeans = MiniBatchKMeans(n_clusters=best_k, random_state=42, batch_size=1000)
    evaluate_model("MiniBatchKMeans", kmeans.fit_predict(X_pca), X_pca)

    # Agglomerative Clustering
    evaluate_model("Agglomerative", AgglomerativeClustering(n_clusters=best_k).fit_predict(X_pca), X_pca)

    # Gaussian Mixture Model
    evaluate_model("Gaussian Mixture", GaussianMixture(n_components=best_k, random_state=42).fit_predict(X_pca), X_pca)

    # DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=10)
    evaluate_model("DBSCAN", dbscan.fit_predict(X_pca), X_pca)

def main():
    try:
        file_path = input("Enter path to CSV file: ").strip()
        X_scaled = load_and_preprocess(file_path)
        X_pca = apply_pca(X_scaled)
        best_k = optimal_kmeans(X_pca)
        run_clustering_algorithms(X_pca, best_k)
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
