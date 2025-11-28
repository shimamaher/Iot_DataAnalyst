
# =============================================================================
# CLASS 03 – ADVANCED CLUSTERING ANALYSIS (FINAL VERSION)
# =============================================================================
# This module performs an advanced clustering analysis on the same cleaned CSV
# that was used in corelation_new.py. It includes:
#
#   - Elbow Method + Silhouette Score (for optimal K)
#   - K-Means++ Clustering
#   - Hierarchical Clustering (Memory-safe version)
#   - DBSCAN (density-based clustering)
#   - Gaussian Mixture Model (GMM)
#   - PCA visualizations
#
# Outputs:
#   - 9_elbow_method.png
#   - 10_dendrogram.png
#   - 11_clustering_visualization.png
#   - kmeans_cluster_statistics.csv
#   - clustering_results.csv
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors, NearestCentroid
from scipy.cluster.hierarchy import dendrogram, linkage

plt.style.use("ggplot")

print("\n" + "=" * 80)
print("ADVANCED CLUSTERING ANALYSIS – CLASS 03")
print("=" * 80)

# =============================================================================
# 1. CONFIGURATION & LOAD DATA
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(
    BASE_DIR,
    "cleaned_datasets",
    "cleaned_data_20251126_194829.csv"
)

df_cleaned = pd.read_csv(INPUT_FILE, low_memory=False)
print(f"\n✓ Loaded {len(df_cleaned):,} rows from: {INPUT_FILE}")

# =============================================================================
# 2. FEATURE SELECTION FOR CLUSTERING
# =============================================================================

# IMPORTANT: column names must match the cleaned dataset
clustering_features = [
    "Voltage (V)",
    "Current (A)",
    "Power Consumption (kW)",
    "Power Factor",
    "Reactive Power (kVAR)",
]

# Keep only columns that exist
clustering_features = [c for c in clustering_features if c in df_cleaned.columns]

if len(clustering_features) < 2:
    print("\n Not enough features found for clustering!")
    print("   Features found:", clustering_features)
    exit()

print("\n✓ Features used for clustering:")
for c in clustering_features:
    print("   -", c)

# Remove rows with NaN
X = df_cleaned[clustering_features].dropna()
print(f"\n✓ Number of samples used for clustering: {len(X):,}")

# Save original index for merging results later
original_index = X.index

# =============================================================================
# 3. SCALING
# =============================================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =============================================================================
# 4. FIND OPTIMAL K – ELBOW + SILHOUETTE
# =============================================================================

print("\n" + "-" * 80)
print("1) Finding Optimal Number of Clusters (Elbow + Silhouette)")
print("-" * 80)

inertias = []
silhouette_scores_list = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=15,
        random_state=42
    )
    kmeans.fit(X_scaled)
    labels = kmeans.labels_

    inertias.append(kmeans.inertia_)
    sil_score = silhouette_score(X_scaled, labels)
    silhouette_scores_list.append(sil_score)

    print(f"  k={k:2d} → Inertia={kmeans.inertia_:,.0f}  Silhouette={sil_score:.4f}")

# Save elbow/silhouette plot
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].plot(list(K_range), inertias, "bo-", linewidth=2)
axes[0].set_title("Elbow Method – Inertia")
axes[0].set_xlabel("k")
axes[0].set_ylabel("Inertia")
axes[0].grid(True, alpha=0.3)

axes[1].plot(list(K_range), silhouette_scores_list, "ro-", linewidth=2)
axes[1].set_title("Silhouette Score vs k")
axes[1].set_xlabel("k")
axes[1].set_ylabel("Silhouette Score")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
elbow_file = os.path.join(BASE_DIR, "9_elbow_method.png")
plt.savefig(elbow_file, dpi=300)
plt.close()

print(f"\n✓ Elbow & Silhouette Plot saved → {elbow_file}")

# Choose best k by silhouette
optimal_k = list(K_range)[silhouette_scores_list.index(max(silhouette_scores_list))]
print(f"\n✓ Optimal number of clusters (Silhouette) = k = {optimal_k}")

# =============================================================================
# 5. K-MEANS++ CLUSTERING
# =============================================================================

print("\n" + "-" * 80)
print(f"2) K-Means++ Clustering (k={optimal_k})")
print("-" * 80)

kmeans = KMeans(
    n_clusters=optimal_k,
    init="k-means++",
    n_init=20,
    random_state=42
)
kmeans_labels = kmeans.fit_predict(X_scaled)

sil_km = silhouette_score(X_scaled, kmeans_labels)
db_km = davies_bouldin_score(X_scaled, kmeans_labels)
ch_km = calinski_harabasz_score(X_scaled, kmeans_labels)

print(f"   Silhouette Score:        {sil_km:.4f}")
print(f"   Davies-Bouldin Index:    {db_km:.4f}")
print(f"   Calinski-Harabasz Index: {ch_km:.4f}")

df_clusters = X.copy()
df_clusters["Cluster_KMeans"] = kmeans_labels

cluster_stats = df_clusters.groupby("Cluster_KMeans").agg(["mean", "count"])
print("\n   K-Means Cluster Statistics:")
print(cluster_stats)

stats_file = os.path.join(BASE_DIR, "kmeans_cluster_statistics.csv")
cluster_stats.to_csv(stats_file, encoding="utf-8-sig")
print(f"\n✓ Saved → {stats_file}")

# =============================================================================
# 6. HIERARCHICAL CLUSTERING (MEMORY-SAFE VERSION)
# =============================================================================

print("\n" + "-" * 80)
print("3) Hierarchical Clustering – Memory Safe Version")
print("-" * 80)

sample_size = min(1000, len(X_scaled))
sample_idx = np.random.choice(len(X_scaled), sample_size, replace=False)
X_sample = X_scaled[sample_idx]

linkage_matrix = linkage(X_sample, method="ward")

plt.figure(figsize=(16, 8))
dendrogram(
    linkage_matrix,
    truncate_mode='lastp',
    p=30,
    leaf_rotation=90,
    leaf_font_size=10
)
plt.title("Hierarchical Clustering Dendrogram (Sampled)")
plt.tight_layout()
dendro_file = os.path.join(BASE_DIR, "10_dendrogram.png")
plt.savefig(dendro_file, dpi=300)
plt.close()
print(f"   ✓ Dendrogram saved → {dendro_file}")

# Fit hierarchical on sample
hier_cluster = AgglomerativeClustering(n_clusters=optimal_k)
hier_sample_labels = hier_cluster.fit_predict(X_sample)

# Assign all points by nearest centroid
centroid_model = NearestCentroid()
centroid_model.fit(X_sample, hier_sample_labels)
hierarchical_labels = centroid_model.predict(X_scaled)

df_clusters["Cluster_Hierarchical"] = hierarchical_labels
print("   ✓ Hierarchical Clustering completed")

# =============================================================================
# 7. DBSCAN
# =============================================================================

print("\n" + "-" * 80)
print("4) DBSCAN Clustering")
print("-" * 80)

neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(X_scaled)
distances, indices = neighbors_fit.kneighbors(X_scaled)
distances = np.sort(distances[:, -1])

eps = np.percentile(distances, 95)

dbscan = DBSCAN(eps=eps, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

n_clusters_db = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)

print(f"   Estimated eps:          {eps:.4f}")
print(f"   Number of clusters:     {n_clusters_db}")
print(f"   Noise points:           {n_noise} ({n_noise/len(X_scaled)*100:.2f}%)")

df_clusters["Cluster_DBSCAN"] = dbscan_labels

# =============================================================================
# 8. GAUSSIAN MIXTURE MODEL
# =============================================================================

print("\n" + "-" * 80)
print("5) Gaussian Mixture Model (GMM)")
print("-" * 80)

gmm = GaussianMixture(
    n_components=optimal_k,
    covariance_type="full",
    random_state=42
)
gmm_labels = gmm.fit_predict(X_scaled)

sil_gmm = silhouette_score(X_scaled, gmm_labels)
print(f"   Silhouette Score (GMM): {sil_gmm:.4f}")

df_clusters["Cluster_GMM"] = gmm_labels

# =============================================================================
# 9. PCA VISUALIZATION
# =============================================================================

print("\n" + "-" * 80)
print("6) PCA Cluster Visualization")
print("-" * 80)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# --- K-Means ---
sc1 = axes[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap="viridis", s=10)
axes[0, 0].set_title(f"K-Means++ (k={optimal_k})")
plt.colorbar(sc1, ax=axes[0, 0])

# --- Hierarchical ---
sc2 = axes[0, 1].scatter(X_pca[:, 0], X_pca[:, 1], c=hierarchical_labels, cmap="plasma", s=10)
axes[0, 1].set_title("Hierarchical")
plt.colorbar(sc2, ax=axes[0, 1])

# --- DBSCAN ---
sc3 = axes[1, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_labels, cmap="Spectral", s=10)
axes[1, 0].set_title(f"DBSCAN (eps={eps:.3f})")
plt.colorbar(sc3, ax=axes[1, 0])

# --- GMM ---
sc4 = axes[1, 1].scatter(X_pca[:, 0], X_pca[:, 1], c=gmm_labels, cmap="coolwarm", s=10)
axes[1, 1].set_title("Gaussian Mixture Model")
plt.colorbar(sc4, ax=axes[1, 1])

plt.tight_layout()
vis_file = os.path.join(BASE_DIR, "11_clustering_visualization.png")
plt.savefig(vis_file, dpi=300)
plt.close()
print(f"   ✓ Cluster Visualization saved → {vis_file}")

# =============================================================================
# 10. SAVE RESULTS
# =============================================================================

result_df = df_cleaned.copy()
for col in df_clusters.columns:
    if col.startswith("Cluster_"):
        result_df[col] = np.nan
        result_df.loc[original_index, col] = df_clusters[col].values

output_file = os.path.join(BASE_DIR, "clustering_results.csv")
result_df.to_csv(output_file, index=False, encoding="utf-8-sig")
print(f"\n✓ Clustering results saved → {output_file}")

print("\n" + "=" * 80)
print("ADVANCED CLUSTERING ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 80)
