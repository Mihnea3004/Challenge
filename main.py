import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import sklearn.feature_extraction.text as extract
import numpy

data = pd.read_csv("ml_insurance_challenge.csv", sep=",")
taxonomy = pd.read_csv("insurance_taxonomy - insurance_taxonomy.csv")

data["all"] = data[["description","business_tags","niche"]].fillna("").agg(" ".join, axis=1)

all_text = pd.concat([data["all"],taxonomy["label"]])

vectorizer = extract.TfidfVectorizer(stop_words="english")
matrix = vectorizer.fit_transform(all_text)

company_vectors = matrix[:len(data)]
taxonomy_vectors = matrix[len(data):]

similarities = cosine_similarity(company_vectors, taxonomy_vectors)

top_n = 100
rate = 0.8
labels = []
last = 0

for sim in similarities:
    top_indices = sim.argsort()[-top_n:][::-1]
    maximum = sim[top_indices[0]]
    filtered_indices = [idx for idx in top_indices if sim[idx] >= maximum * rate]
    matched_labels = taxonomy.iloc[filtered_indices]["label"].values
    labels.append(", ".join(matched_labels))

data["insurance_labels"] = labels

data.to_csv("insurance_challenge.csv", index=False)