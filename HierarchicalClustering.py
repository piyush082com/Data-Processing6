
# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

# Load Iris dataset
iris = load_iris()
X = iris.data

print("Dataset Shape:", X.shape)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Hierarchical Clustering
hierarchical = AgglomerativeClustering(n_clusters=3, linkage='ward')
h_labels = hierarchical.fit_predict(X_scaled)

# Calculate Silhouette Score
h_score = silhouette_score(X_scaled, h_labels)

print("\nHierarchical Clustering")
print("Silhouette Score:", round(h_score, 3))

# Draw Dendrogram
plt.figure(figsize=(10, 6))
dendrogram(linkage(X_scaled, method='ward'))
plt.title("Dendrogram - Hierarchical Clustering")
plt.xlabel("Data Points")
plt.ylabel("Euclidean Distance")
plt.show()

# K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
k_labels = kmeans.fit_predict(X_scaled)

# Calculate Silhouette Score
k_score = silhouette_score(X_scaled, k_labels)

print("\nK-Means Clustering")
print("Silhouette Score:", round(k_score, 3))

# Compare both algorithms
print("\nPerformance Comparison")
print("Hierarchical Silhouette Score:", round(h_score, 3))
print("K-Means Silhouette Score:", round(k_score, 3))

# Visualize both clustering results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=h_labels, cmap='viridis')
plt.title("Hierarchical Clustering")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")

plt.subplot(1, 2, 2)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=k_labels, cmap='viridis')
plt.title("K-Means Clustering")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")

plt.tight_layout()
plt.show()
