# Task 3 — Cuisine Classification

## Objective

Develop a machine learning model to classify restaurants according to their primary cuisine.

## Problem Formulation

Because the original dataset contains restaurants with multiple cuisines, the first listed cuisine was treated as the primary cuisine for this single-label classification benchmark.

Very rare cuisine classes were excluded from the supervised benchmark to support meaningful stratified evaluation.

## Model

TF-IDF + Logistic Regression

## Key Steps

- Data preprocessing
- Primary cuisine extraction
- Class filtering
- TF-IDF feature extraction
- Stratified train/test split
- Balanced class weights
- Multiclass classification
- Confusion-matrix analysis

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- Weighted F1
- Macro F1
- Top-3 accuracy

## Important Limitation

Cuisine classification is challenging with the available restaurant metadata because the dataset does not contain rich menu or review text. A production system could use semantic embeddings or true multi-label classification.

## Technologies

Python, Pandas, NumPy, Scikit-learn
