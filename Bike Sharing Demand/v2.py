import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split

# 1. Load and prepare data
train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")


# 2. Feature engineering
def create_features(df):
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


train = create_features(train)
test = create_features(test)

# 3. Handle missing values
imputer = SimpleImputer(strategy="median")
numeric_features = ["temp", "atemp", "humidity", "windspeed"]
train[numeric_features] = imputer.fit_transform(train[numeric_features])
test[numeric_features] = imputer.transform(test[numeric_features])

# 4. Scale numeric features
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train[numeric_features])
test_scaled = scaler.transform(test[numeric_features])

# Add scaled features back to DataFrames
scaled_cols = [f"{col}_scaled" for col in numeric_features]
train = pd.concat(
    [train, pd.DataFrame(train_scaled, columns=scaled_cols, index=train.index)], axis=1
)
test = pd.concat(
    [test, pd.DataFrame(test_scaled, columns=scaled_cols, index=test.index)], axis=1
)

# 5. Temporal clustering with safe epsilon calculation
temporal_features = ["hour", "day_of_week"]

# Option 1: Simple time-based grouping (more reliable)
time_bins = [0, 6, 9, 16, 19, 24]  # Customize these bins
time_labels = ["Late_Night", "Morning_Rush", "Daytime", "Evening_Rush", "Night"]

train["temporal_cluster"] = pd.cut(train["hour"], bins=time_bins, labels=time_labels)
test["temporal_cluster"] = pd.cut(test["hour"], bins=time_bins, labels=time_labels)

# Convert to numeric codes for modeling
train["temporal_cluster"] = train["temporal_cluster"].cat.codes
test["temporal_cluster"] = test["temporal_cluster"].cat.codes

# 6. Prepare final features
features = (
    numeric_features
    + scaled_cols
    + ["hour", "day_of_week", "is_weekend", "temporal_cluster"]
)
X = train[features]
y = train["count"]

# 7. Train/validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Train Random Forest
rf = RandomForestRegressor(
    n_estimators=100, max_depth=10, min_samples_split=5, random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)

# 9. Evaluate
val_pred = rf.predict(X_val)
val_pred = np.maximum(0, val_pred)  # Ensure non-negative predictions
rmsle = np.sqrt(mean_squared_log_error(y_val, val_pred))
print(f"Validation RMSLE: {rmsle:.4f}")

# 10. Feature importance visualization
importances = rf.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), importances[sorted_idx], align="center")
plt.yticks(range(len(sorted_idx)), np.array(features)[sorted_idx])
plt.title("Feature Importance")
plt.show()

# 11. Predict on test set
test_pred = rf.predict(test[features])
test_pred = np.maximum(0, test_pred)  # Ensure non-negative predictions

# 12. Create submission
submission = pd.DataFrame({"datetime": test["datetime"], "count": test_pred})
submission.to_csv("submission_final.csv", index=False)
print("Submission file created successfully!")

# 13. Example predictions (like teacher's demo)
sample_data = pd.DataFrame(
    {
        "temp": [15, 30],
        "atemp": [16, 32],
        "humidity": [40, 80],
        "windspeed": [5, 15],
        "hour": [8, 18],
        "day_of_week": [1, 5],
        "is_weekend": [0, 1],
        "temporal_cluster": [2, 3],  # From our clustering
    }
)

# Add scaled features
sample_scaled = scaler.transform(sample_data[numeric_features])
sample_data[scaled_cols] = sample_scaled

print("\nSample predictions:")
sample_data["predicted_count"] = rf.predict(sample_data[features])
print(sample_data[["hour", "day_of_week", "predicted_count"]])
