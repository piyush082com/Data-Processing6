
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report

# Load UCI Heart Disease dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

columns = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "target"
]

df = pd.read_csv(url, names=columns)

# Replace missing values
df.replace("?", np.nan, inplace=True)
df.dropna(inplace=True)

# Convert data into numeric format
df = df.astype(float)

# Convert target into binary classification
df["target"] = (df["target"] > 0).astype(int)

# Separate features and target
X = df.drop("target", axis=1)
y = df["target"]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Decision Tree Model
dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

# Random Forest Model
rf = RandomForestClassifier(
    n_estimators=100, max_depth=4, random_state=42
)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

# Evaluate Decision Tree
print("DECISION TREE RESULTS")
print("Accuracy:", accuracy_score(y_test, dt_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, dt_pred))
print(classification_report(y_test, dt_pred))

# Evaluate Random Forest
print("\nRANDOM FOREST RESULTS")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

# Compare Model Accuracy
dt_acc = accuracy_score(y_test, dt_pred)
rf_acc = accuracy_score(y_test, rf_pred)

plt.figure(figsize=(6, 4))
plt.bar(["Decision Tree", "Random Forest"],
        [dt_acc, rf_acc])
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.ylim(0, 1)
plt.show()

# Bias-Variance Tradeoff Analysis
print("\nBIAS-VARIANCE ANALYSIS")

print("Decision Tree Training Accuracy:",
      dt.score(X_train, y_train))
print("Decision Tree Testing Accuracy:",
      dt.score(X_test, y_test))

print("Random Forest Training Accuracy:",
      rf.score(X_train, y_train))
print("Random Forest Testing Accuracy:",
      rf.score(X_test, y_test))

# Feature Importance - Random Forest
importance = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

plt.figure(figsize=(9, 5))
importance.plot(kind="bar")
plt.title("Random Forest Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()

# Feature Importance - Decision Tree
plt.figure(figsize=(9, 5))
plt.bar(X.columns, dt.feature_importances_)
plt.title("Decision Tree Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
