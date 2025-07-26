import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Ensure output directories exist
os.makedirs("output", exist_ok=True)
os.makedirs("graphs", exist_ok=True)

# Load dataset
df = pd.read_csv("AI_Assignment/input/normalized_df.csv")

# Elbow method to find optimal K
sse = []
k_range = range(1, 11)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(df)
    sse.append(km.inertia_)

# Plot the Elbow
plt.figure(figsize=(8, 5))
plt.plot(k_range, sse, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('SSE (Inertia)')
plt.title('Elbow Method for Optimal K')
plt.grid(True)
plt.savefig("AI_Assignment/graphs/elbow_kmeans.png", dpi=300)
plt.close()

# Apply KMeans with chosen K based on elbow method
k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(df)

# Assign cluster labels
labels = kmeans.labels_
df['cluster'] = labels

# Evaluate clustering
score = silhouette_score(df.drop("cluster", axis=1), labels)
print(f'Silhouette Score: {score:.4f}')
print(f'Number of clusters: {k}')

# PCA for 2D visualization
pca = PCA(n_components=2)
components = pca.fit_transform(df.drop("cluster", axis=1))
df['pca1'], df['pca2'] = components[:, 0], components[:, 1]

print(df.head())
print(df.shape)

# Plot clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='pca1', y='pca2', hue='cluster', palette='tab10')
plt.title("K-Means Clustering Results (PCA-reduced)")
plt.savefig("AI_Assignment/graphs/kmeans_clusters.png", dpi=300)

