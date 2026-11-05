import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

LABEL_MAP = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"
    df = pd.read_csv(url, header=None, names=["label", "title", "description"])
    df["label_name"] = df["label"].map(LABEL_MAP)
    return df.sample(1000, random_state=42).reset_index(drop=True)

class SimpleVectorDB:
    def __init__(self):
        self.embeddings = None
        self.metadata = None
        self.scale = 127.0

    def add(self, embeddings, metadata):
        embeddings = normalize(np.asarray(embeddings))
        self.embeddings = np.clip(np.round(embeddings * self.scale), -127, 127).astype(np.int8)
        self.metadata = list(metadata)

    def search(self, query_embedding, top_k=5):
        query_embedding = normalize(np.asarray(query_embedding).reshape(1, -1))[0]
        stored = self.embeddings.astype(np.float32) / self.scale
        scores = stored @ query_embedding
        idx = np.argsort(scores)[::-1][:top_k]
        return [(float(scores[i]), self.metadata[i]) for i in idx]

@st.cache_resource
def build_index(df):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=20000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df["title"] + " " + df["description"])
    svd = TruncatedSVD(n_components=80, random_state=42)
    emb = svd.fit_transform(X)
    db = SimpleVectorDB()
    db.add(emb, df[["title", "description", "label_name"]].to_dict("records"))
    return vectorizer, svd, db

st.title("AG News Vector Search")
df = load_data()
vectorizer, svd, db = build_index(df)

query = st.text_input("Search news articles", "AI companies release new chips")
top_k = st.slider("Number of results", 1, 10, 5)

if query:
    q = svd.transform(vectorizer.transform([query]))[0]
    results = db.search(q, top_k=top_k)
    for score, item in results:
        st.write(f"**{item['label_name']}** | score: {score:.3f}")
        st.write(item["title"])
        st.caption(item["description"])
        st.divider()
