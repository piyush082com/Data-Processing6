
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load UCI Auto MPG dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"

columns = [
    "mpg", "cylinders", "displacement", "horsepower",
    "weight", "acceleration", "model_year", "origin"
]

df = pd.read_csv(
    url,
    sep=r'\s+',
    names=columns,
    na_values='?'
)

# Handle missing values
df.dropna(inplace=True)

print("Dataset:")
print(df.head())
print("\nDataset Shape:", df.shape)

# Separate input features and target
X = df.drop("mpg", axis=1)
y = df["mpg"]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict fuel efficiency
y_pred = model.predict(X_test)

# Evaluate model performance
print("\nModel Evaluation")
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:",
      np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))

# Display feature coefficients
print("\nFeature Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(feature, ":", round(coef, 3))

# Visualize Actual vs Predicted MPG
plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual MPG")
plt.ylabel("Predicted MPG")
plt.title("Actual vs Predicted Fuel Efficiency")
plt.show()

# Visualize Feature Correlation
plt.figure(figsize=(9, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# Visualize Feature Influence
plt.figure(figsize=(9, 5))
plt.bar(X.columns, model.coef_)
plt.xlabel("Features")
plt.ylabel("Regression Coefficients")
plt.title("Feature Influence on Fuel Efficiency")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
