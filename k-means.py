
# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load Wine dataset
wine = load_wine()
X = wine.data

print("Original Dataset Shape:", X.shape)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA to reduce features to 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Reduced Dataset Shape:", X_pca.shape)

# Display explained variance
print("Explained Variance Ratio:", pca.explained_variance_ratio_)
print("Total Variance Retained:",
      round(sum(pca.explained_variance_ratio_) * 100, 2), "%")

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_pca)

# Calculate silhouette score
score = silhouette_score(X_pca, clusters)
print("Silhouette Score:", round(score, 3))

# Visualize PCA clusters
plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=clusters,
    cmap='viridis',
    s=50
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("K-Means Clustering using PCA on Wine Dataset")
plt.colorbar(label="Cluster")
plt.show()

# Visualize actual wine classes for comparison
plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=wine.target,
    cmap='viridis',
    s=50
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Actual Wine Classes after PCA")
plt.colorbar(label="Actual Class")
plt.show()
