import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

# Load data
train_data = pd.read_csv("./input/train.csv")
test_data = pd.read_csv("./input/test.csv")


# Feature Engineering
def preprocess_data(df):
    # Fill missing Age with median
    df["Age"] = df["Age"].fillna(df["Age"].median())

    # Fill missing Fare with median
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    # Extract titles from names (e.g., Mr, Mrs, Miss)
    df["Title"] = df["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)
    df["Title"] = df["Title"].replace(
        [
            "Lady",
            "Countess",
            "Capt",
            "Col",
            "Don",
            "Dr",
            "Major",
            "Rev",
            "Sir",
            "Jonkheer",
            "Dona",
        ],
        "Rare",
    )
    df["Title"] = df["Title"].replace("Mlle", "Miss")
    df["Title"] = df["Title"].replace("Ms", "Miss")
    df["Title"] = df["Title"].replace("Mme", "Mrs")

    # Create a new feature: Family Size
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    # Create a new feature: Is Alone
    df["IsAlone"] = 0
    df.loc[df["FamilySize"] == 1, "IsAlone"] = 1

    # Drop unnecessary columns
    df = df.drop(["Name", "Ticket", "Cabin", "PassengerId"], axis=1)

    return df


# Preprocess data
train_data = preprocess_data(train_data)
test_data = preprocess_data(test_data)

# One-Hot Encoding for categorical features
train_data = pd.get_dummies(train_data, columns=["Sex", "Embarked", "Title"])
test_data = pd.get_dummies(test_data, columns=["Sex", "Embarked", "Title"])

# Ensure both train and test have the same columns
missing_cols = set(train_data.columns) - set(test_data.columns)
for col in missing_cols:
    test_data[col] = 0
test_data = test_data[train_data.columns.drop("Survived")]

# Split data into features and target
X = train_data.drop(["Survived"], axis=1)
y = train_data["Survived"]

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning with GridSearchCV
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 10, 15],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(model, param_grid, cv=5, scoring="accuracy")
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Predict on validation set
val_predictions = best_model.predict(X_val)
print(f"Validation Accuracy: {accuracy_score(y_val, val_predictions)}")

# Predict on test set
test_predictions = best_model.predict(test_data)

# Save submission
output = pd.DataFrame(
    {
        "PassengerId": pd.read_csv("./input/test.csv")["PassengerId"],
        "Survived": test_predictions,
    }
)
output.to_csv("./output/submission.csv", index=False)
print("Your improved submission was successfully saved!")
