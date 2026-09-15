"""
Cognifyz Technologies — Machine Learning Internship
Task 3: Cuisine Classification
Install: pip install pandas numpy scikit-learn
Run: python Cognifyz_Task_3_Cuisine_Classification.py
"""
import re, warnings, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

df=pd.read_csv("Dataset .csv")
d=df[["Cuisines","City","Locality","Price range","Average Cost for two","Aggregate rating","Votes","Has Table booking","Has Online delivery"]].copy()
d["Cuisines"]=d["Cuisines"].fillna("Unknown").astype(str).str.strip()
d=d[d["Cuisines"].ne("")].copy()
d["Primary Cuisine"]=d["Cuisines"].str.split(",").str[0].str.strip()
d=d[d["Primary Cuisine"].ne("Unknown")].copy()

MIN_SAMPLES=30
counts=d["Primary Cuisine"].value_counts()
d=d[d["Primary Cuisine"].isin(counts[counts>=MIN_SAMPLES].index)].copy()

def clean_text(x):
    return re.sub(r"\s+"," ",re.sub(r"[^a-zA-Z0-9]+"," ",str(x).lower())).strip()

# Cuisines is the target source, so it MUST NOT be an input feature.
d["text"]=d["City"].fillna("Unknown").map(clean_text)+" "+d["Locality"].fillna("Unknown").map(clean_text)
d["text"] += " price_"+d["Price range"].fillna(0).astype(str)
d["text"] += " booking_"+d["Has Table booking"].fillna("No").map(clean_text)
d["text"] += " delivery_"+d["Has Online delivery"].fillna("No").map(clean_text)

X_train,X_test,y_train,y_test=train_test_split(
    d["text"],d["Primary Cuisine"],test_size=.20,random_state=42,stratify=d["Primary Cuisine"]
)

vec=TfidfVectorizer(stop_words="english",ngram_range=(1,2),min_df=2,max_features=20000)
Xtr=vec.fit_transform(X_train); Xte=vec.transform(X_test)

classes=np.unique(y_train)
weights=compute_class_weight(class_weight="balanced",classes=classes,y=y_train)
cw=dict(zip(classes,weights))

model=LogisticRegression(max_iter=1200,class_weight=cw,solver="saga",random_state=42)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model.fit(Xtr,y_train)

pred=model.predict(Xte)
print("\n=== COGNIFYZ TASK 3 RESULTS ===")
print(f"Usable records: {len(d):,}")
print(f"Primary-cuisine classes: {d['Primary Cuisine'].nunique()}")
print(f"Accuracy: {accuracy_score(y_test,pred):.3f}")
print(f"Weighted Precision: {precision_score(y_test,pred,average='weighted',zero_division=0):.3f}")
print(f"Weighted Recall: {recall_score(y_test,pred,average='weighted',zero_division=0):.3f}")
print(f"Weighted F1: {f1_score(y_test,pred,average='weighted',zero_division=0):.3f}")
print(f"Macro F1: {f1_score(y_test,pred,average='macro',zero_division=0):.3f}")
print("\nDetailed classification report:\n")
print(classification_report(y_test,pred,zero_division=0))
