
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Load UCI Wholesale Customers dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv"

df = pd.read_csv(url)

# Display dataset
print("Dataset:")
print(df.head())

print("\nDataset Shape:", df.shape)

# Select numerical features
X = df.drop(columns=["Channel", "Region"])

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to dataset
df["Cluster"] = clusters

# Calculate Silhouette Score
score = silhouette_score(X_scaled, clusters)

print("\nSilhouette Score:", round(score, 3))

# Display cluster counts
print("\nCluster Distribution:")
print(df["Cluster"].value_counts())

# Reduce dimensions to 2D using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Visualize clusters
plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=clusters,
    cmap="viridis",
    s=50
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("K-Means Clustering on Wholesale Customers Dataset")
plt.colorbar(label="Cluster")
plt.show()
