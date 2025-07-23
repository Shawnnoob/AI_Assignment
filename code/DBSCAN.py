import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # Data Visualization
import seaborn as sns # Data Visualization
from sklearn.cluster import DBSCAN # DBSCAN model
from sklearn.decomposition import PCA # Principal Component Analysis (PCA)
from sklearn.neighbors import NearestNeighbors

df = pd.read_csv("../input/normalized_df.csv", delimiter=',')

# Set min_samples
min_samples = 100

# Compute distances to the k-th nearest neighbor
neighbors = NearestNeighbors(n_neighbors=min_samples)
neighbors_fit = neighbors.fit(df)
distances, indices = neighbors_fit.kneighbors(df)

# Get the distance to the kth nearest neighbor
k_distances = np.sort(distances[:, -1])  # Last column is kth nearest

# Plot k-distance graph
plt.figure(figsize=(10, 5))
plt.plot(k_distances)
plt.title(f"k-Distance Graph (k = {min_samples - 1})")
plt.xlabel("Data Points sorted by distance")
plt.ylabel(f"{min_samples}-th Nearest Neighbor Distance")
plt.grid(True)
plt.savefig("../graphs/Nearest_Neighbour.png", dpi=300)

# ==================================================

# Based on the elbow, choose an appropriate eps
optimal_eps = 0.25

dbscan = DBSCAN(eps=optimal_eps, min_samples=min_samples)
dbscan.fit(df)

df['cluster'] = dbscan.labels_
print(df['cluster'].value_counts())

# Visualize with PCA
pca = PCA(n_components=2)
components = pca.fit_transform(df)
df['pca1'], df['pca2'] = components[:, 0], components[:, 1]

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='pca1', y='pca2', hue='cluster', palette='tab10')
plt.title("DBSCAN Clusters after Tuning eps")
plt.savefig("../graphs/DBSCAN_tuned.png", dpi=300)

# =====================================================

# dbscan = DBSCAN(eps=0.25, min_samples=20)
# dbscan.fit(df)
#
# df['cluster'] = dbscan.labels_
#
# print(df['cluster'].value_counts())
#
# # Reduce to 2D with PCA for visualization
# pca = PCA(n_components=2)
# components = pca.fit_transform(df)
#
# df['pca1'] = components[:, 0]
# df['pca2'] = components[:, 1]
#
# # Plot clusters
# plt.figure(figsize=(8, 6))
# sns.scatterplot(data=df, x='pca1', y='pca2', hue='cluster', palette='tab10', legend='full')
# plt.title("DBSCAN Clustering (visualized with PCA)")
# plt.xlabel("PCA 1")
# plt.ylabel("PCA 2")
# plt.legend(title="Cluster")
# plt.savefig("../graphs/DBSCAN.png", dpi=300)