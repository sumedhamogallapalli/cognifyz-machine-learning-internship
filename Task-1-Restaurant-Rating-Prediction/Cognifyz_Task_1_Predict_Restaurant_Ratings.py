"""
Cognifyz Technologies — Machine Learning Internship
Task 1: Predict Restaurant Ratings

Place "Dataset .csv" in the same folder and run:
    python Cognifyz_Task_1_Predict_Restaurant_Ratings.py
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

DATA_PATH = "Dataset .csv"
RANDOM_STATE = 42

df = pd.read_csv(DATA_PATH)

# 0.0 is explicitly "Not rated" in this dataset, so it is not a numerical rating.
rated = df[df["Aggregate rating"] > 0].copy()
rated["Cuisines"] = rated["Cuisines"].fillna("Unknown")

# Deliberate leakage control:
# - Exclude target-derived Rating color / Rating text.
# - Exclude identifiers and free-text address/name fields.
FEATURES = [
    "Country Code", "City", "Locality", "Longitude", "Latitude",
    "Cuisines", "Average Cost for two", "Currency",
    "Has Table booking", "Has Online delivery", "Is delivering now",
    "Switch to order menu", "Price range", "Votes"
]

X = rated[FEATURES]
y = rated["Aggregate rating"]
CAT_FEATURES = [c for c in FEATURES if X[c].dtype == "object"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE
)

model = CatBoostRegressor(
    iterations=1200,
    depth=7,
    learning_rate=0.04,
    loss_function="RMSE",
    l2_leaf_reg=5,
    random_seed=RANDOM_STATE,
    verbose=False,
    allow_writing_files=False
)

model.fit(
    X_train, y_train,
    cat_features=CAT_FEATURES,
    eval_set=(X_test, y_test),
    early_stopping_rounds=80,
    verbose=False
)

pred = model.predict(X_test)

rmse = mean_squared_error(y_test, pred) ** 0.5
mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)

print("\n=== COGNIFYZ TASK 1 RESULTS ===")
print(f"Original rows: {len(df):,}")
print(f"Rated rows used: {len(rated):,}")
print(f"RMSE: {rmse:.3f}")
print(f"MAE : {mae:.3f}")
print(f"R²  : {r2:.3f}")
print(f"Within ±0.30: {np.mean(np.abs(y_test.values-pred) <= 0.30)*100:.1f}%")
print(f"Within ±0.50: {np.mean(np.abs(y_test.values-pred) <= 0.50)*100:.1f}%")

print("\nTop features:")
importance = pd.Series(model.get_feature_importance(), index=FEATURES)
print(importance.sort_values(ascending=False).head(10))
