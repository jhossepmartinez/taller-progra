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
import warnings

# Formato de los datos:
# datetime - hourly date + timestamp
# season -  1 = spring, 2 = summer, 3 = fall, 4 = winter
# holiday - whether the day is considered a holiday
# workingday - whether the day is neither a weekend nor holiday
# weather
# 1: Clear, Few clouds, Partly cloudy, Partly cloudy
# 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
# 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
# 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
# temp - temperature in Celsius
# atemp - "feels like" temperature in Celsius
# humidity - relative humidity
# windspeed - wind speed
# casual - number of non-registered user rentals initiated
# registered - number of registered user rentals initiated
# count - number of total rentals


# Load data
train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")


# Enrich data
def create_features(df):
    # Datetime features
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month

    # To make the model know that 23:00 is close to 00:00
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Easiers to work with weather
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["bad_weather"] = (df["weather"] >= 3).astype(int)

    return df


train = create_features(train)
test = create_features(test)


# Clustering, seems broken or i don't understand how it works because its creating a single cluster (visible with matplotlib)
def adaptive_clustering(df):
    # Use temporal features for clustering
    temporal_features = ["hour", "day_of_week"]
    df["cluster"] = -1  # Initialize as noise

    # gpt
    neighbors = NearestNeighbors(n_neighbors=5)
    neighbors.fit(df[temporal_features])
    dists, _ = neighbors.kneighbors(df[temporal_features])
    eps = np.percentile(dists[:, 4], 90)
    eps = max(eps, 1.0)

    # DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=5)
    clusters = dbscan.fit_predict(df[temporal_features])
    df["cluster"] = clusters

    return df, eps


train, eps = adaptive_clustering(train)
test["cluster"] = DBSCAN(eps=eps, min_samples=5).fit_predict(
    test[["hour", "day_of_week"]]
)

# Visualize clusters gpt (no cacho)
plt.figure(figsize=(12, 6))
for c in train["cluster"].unique():
    cluster_data = train[train["cluster"] == c]
    plt.scatter(
        cluster_data["hour"], cluster_data["day_of_week"], s=10, label=f"Cluster {c}"
    )
plt.title("Temporal Clustering (DBSCAN)")
plt.xlabel("Hour of Day")
plt.ylabel("Day of Week")
plt.legend()
plt.grid(True)
plt.savefig("temporal_clusters.png")
print("Saved cluster visualization")

# PROCESS DATA (add missing values)
# Handle missing values
numeric_features = ["temp", "atemp", "humidity", "windspeed"]
imputer = SimpleImputer(strategy="median")  # replace missing values with median
train[numeric_features] = imputer.fit_transform(
    train[numeric_features]
)  # add missing values
test[numeric_features] = imputer.transform(test[numeric_features])  # add missing values

features = numeric_features + [
    "hour_sin",
    "hour_cos",
    "cluster",
    "is_weekend",
    "bad_weather",
]

X = train[features]
y = np.log1p(train["count"])  # Log-transform target

# Train-test split 80/20
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Keep depth at 7 to avoid overfitting
rf = RandomForestRegressor(
    n_estimators=200, max_depth=7, min_samples_split=5, random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)

# Results ~.5 is good enough
val_pred = np.expm1(rf.predict(X_val))  # Reverse log transform
rmsle = np.sqrt(mean_squared_log_error(np.expm1(y_val), val_pred))
print(f"\nValidation RMSLE: {rmsle}")

# Feature importance (for future improvs)
plt.figure(figsize=(12, 6))
importances = rf.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
plt.barh(range(len(sorted_idx)), importances[sorted_idx], align="center")
plt.yticks(range(len(sorted_idx)), np.array(features)[sorted_idx])
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
print("Saved feature importance plot")
