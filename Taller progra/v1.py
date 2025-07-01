import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Load data
data = pd.read_csv("./input/City_Types.csv")

# Convert 'Type' to binary (0: Residential, 1: Industrial)
data["Type"] = LabelEncoder().fit_transform(data["Type"])

# Feature Engineering
data["Total_Pollution"] = data[["CO", "NO2", "SO2", "PM2.5", "PM10"]].sum(axis=1)
data["NO2_SO2_Ratio"] = (data["NO2"] + 1e-6) / (
    data["SO2"] + 1e-6
)  # Avoid division by zero
data["O3_Ratio"] = data["O3"] / (
    data["PM2.5"] + 1e-6
)  # Ozone relative to fine particles

# Extract hour from datetime (pollution patterns vary by time)
data["Hour"] = pd.to_datetime(data["Date"]).dt.hour

# One-Hot Encoding for 'City' (categorical)
data = pd.get_dummies(data, columns=["City"])

# Drop original 'Date' (not needed for modeling)
X = data.drop(["Date", "Type"], axis=1)
y = data["Type"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Hyperparameter Tuning (GridSearchCV)
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [10, 15, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}

model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(model, param_grid, cv=5, scoring="accuracy")
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# Evaluate Model
y_pred = best_model.predict(X_test)
print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Residential", "Industrial"],
    yticklabels=["Residential", "Industrial"],
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Feature Importance
feature_importance = pd.DataFrame(
    {"Feature": X.columns, "Importance": best_model.feature_importances_}
).sort_values("Importance", ascending=False)

print("\nTop 10 Features:")
print(feature_importance.head(10))

# Plot Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feature_importance.head(10))
plt.title("Top 10 Important Features")
plt.show()
