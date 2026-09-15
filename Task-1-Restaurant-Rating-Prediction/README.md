# Task 1 — Predict Restaurant Ratings

## Objective

Build a machine learning regression model to predict restaurant aggregate ratings.

## Model

CatBoost Regression

## Key Steps

- Data cleaning
- Missing-value handling
- Feature selection
- Target leakage prevention
- Categorical feature handling
- Train/test splitting
- Regression modeling
- Model evaluation
- Feature importance analysis

## Evaluation Metrics

- RMSE
- MAE
- R²

## Key Insight

Restaurants marked as "Not rated" were not treated as genuine zero-star ratings. Target-derived fields such as Rating Text and Rating Color were excluded to prevent data leakage.

## Technologies

Python, Pandas, NumPy, Scikit-learn, CatBoost
