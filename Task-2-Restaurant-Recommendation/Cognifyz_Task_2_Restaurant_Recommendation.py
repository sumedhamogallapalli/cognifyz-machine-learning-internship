"""
Cognifyz Technologies — Machine Learning Internship
Task 2: Restaurant Recommendation

Content-based / preference-aware recommender.
Place "Dataset .csv" in the same folder.

Install:
    pip install pandas numpy scikit-learn

Run:
    python Cognifyz_Task_2_Restaurant_Recommendation.py
"""

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = "Dataset .csv"

df = pd.read_csv(DATA_PATH)
rec = df.copy()

# ---- Preprocess ----
for col in ["Cuisines","City","Locality"]:
    rec[col] = rec[col].fillna("Unknown").astype(str)

for col in ["Average Cost for two","Price range","Aggregate rating","Votes"]:
    rec[col] = pd.to_numeric(rec[col], errors="coerce")

rec["Price range"] = rec["Price range"].fillna(rec["Price range"].median())
rec["Aggregate rating"] = rec["Aggregate rating"].fillna(0)
rec["Votes"] = rec["Votes"].fillna(0)

def clean_text(x):
    x = re.sub(r"[^a-zA-Z0-9]+", " ", str(x).lower())
    return re.sub(r"\s+", " ", x).strip()

# TF-IDF content profile
rec["content_text"] = (
    rec["Cuisines"].map(clean_text) + " " +
    rec["City"].map(clean_text) + " " +
    rec["Locality"].map(clean_text)
)

vectorizer = TfidfVectorizer(
    stop_words="english", ngram_range=(1,2), min_df=2
)
tfidf = vectorizer.fit_transform(rec["content_text"])

# Normalization for numeric preference signals
scaler = MinMaxScaler()
num_cols = ["Price range","Average Cost for two","Aggregate rating","Votes"]
num_matrix = scaler.fit_transform(rec[num_cols])

def yes(v):
    return 1 if str(v).strip().lower() == "yes" else 0

service_matrix = np.column_stack([
    rec["Has Table booking"].map(yes),
    rec["Has Online delivery"].map(yes)
])

votes_log = np.log1p(rec["Votes"].values)
votes_norm = (votes_log-votes_log.min())/(votes_log.max()-votes_log.min()+1e-12)
rating_norm = np.clip(rec["Aggregate rating"].values/5.0,0,1)

def recommend(
    cuisine=None,
    city=None,
    locality=None,
    price_range=None,
    table_booking=None,
    online_delivery=None,
    top_n=10
):
    """Return ranked restaurants with an interpretable match score."""

    parts = []
    if cuisine: parts.append(clean_text(cuisine))
    if city: parts.append(clean_text(city))
    if locality: parts.append(clean_text(locality))

    query = " ".join(parts) if parts else "restaurant"
    qvec = vectorizer.transform([query])
    content_sim = cosine_similarity(qvec, tfidf).ravel()

    # Main content similarity + quality signals
    score = (
        0.65 * content_sim +
        0.10 * rating_norm +
        0.05 * votes_norm
    )

    # Price preference
    if price_range is not None:
        pr = float(price_range)
        score += 0.15 * (
            1 - np.minimum(np.abs(rec["Price range"].values-pr)/3.0, 1)
        )
    else:
        score += 0.15 * num_matrix[:,0]

    # Optional service preferences
    preferences = [
        ("Has Table booking", table_booking, 0),
        ("Has Online delivery", online_delivery, 1)
    ]
    for col, pref, service_idx in preferences:
        if pref is not None:
            p = 1 if str(pref).lower() in ("yes","y","true","1") else 0
            score += 0.025 * (service_matrix[:,service_idx] == p)

    # Avoid recommending unrated restaurants when enough rated choices exist.
    if (rec["Aggregate rating"] > 0).sum() >= top_n:
        score = np.where(rec["Aggregate rating"].values > 0, score, -np.inf)

    idx = np.argsort(-score)[:top_n]
    out = rec.iloc[idx].copy()
    out["Match Score"] = np.round(np.clip(score[idx],0,1)*100,1)

    out["Why Recommended"] = out.apply(
        lambda r:
        "Cuisine/location similarity"
        + (f" • Price range {int(r['Price range'])}" if price_range is not None else "")
        + (f" • Rating {r['Aggregate rating']:.1f}" if r["Aggregate rating"] > 0 else ""),
        axis=1
    )

    return out[[
        "Restaurant Name","City","Locality","Cuisines",
        "Price range","Aggregate rating","Votes",
        "Has Table booking","Has Online delivery",
        "Match Score","Why Recommended"
    ]]

if __name__ == "__main__":
    print("\n=== COGNIFYZ TASK 2: RESTAURANT RECOMMENDER ===")

    city = df["City"].dropna().value_counts().index[0]

    results = recommend(
        cuisine="North Indian",
        city=city,
        price_range=2,
        top_n=10
    )

    print("\nUser preferences:")
    print(f"  Cuisine     : North Indian")
    print(f"  City        : {city}")
    print(f"  Price range : 2")

    print("\nTop recommendations:")
    print(results.to_string(index=False))
