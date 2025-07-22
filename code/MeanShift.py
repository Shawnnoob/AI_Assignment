import pandas as pd
from sklearn.cluster import MeanShift
from sklearn.metrics import silhouette_score

# load the dataset
normalized_dt = pd.read_csv("AI_Assignment/input/normalized_df.csv")


# apply the meanshift algorithms
meanshift = MeanShift()
meanshift.fit(normalized_dt)

# numbering each cluster
labels = meanshift.labels_

print(labels[109:324])

# evaluate the score of the clustering
score = silhouette_score(normalized_dt, labels)
print(f'Silhouette Score: {score:.4f}')
print(f'Number of clusters found: {len(set(labels))}')
