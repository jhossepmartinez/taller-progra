import pandas as pd
import numpy as np
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
    return df


train = create_features(train)
test = create_features(test)

# 3. Define features and target
features = [
    "season",
    "holiday",
    "workingday",
    "weather",
    "temp",
    "atemp",
    "humidity",
    "windspeed",
    "hour",
    "day_of_week",
    "month",
]
target = "count"

X = train[features]
y = train[target]

# 4. Train-test split (for validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 6. Evaluate on validation set
val_pred = model.predict(X_val)
rmsle = np.sqrt(mean_squared_log_error(y_val, val_pred))
print(f"Validation RMSLE: {rmsle:.4f}")

# 7. Make predictions on test set
test_pred = model.predict(test[features])

# 8. Prepare submi
submission = pd.DataFrame({"datetime": test["datetime"], "count": test_pred})
submission.to_csv("submission.csv", index=False)
