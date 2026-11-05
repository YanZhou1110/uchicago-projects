import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_data
def load_data():
    path = Path("assignment1_outputs/news_articles.csv")
    if not path.exists():
        st.error("Dataset file not found. Run the notebook first to generate assignment1_outputs/news_articles.csv")
        st.stop()
    return pd.read_csv(path)

def custom_tokenize(text):
    text = str(text).lower()
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text)

def custom_stem(words):
    stems = []
    for word in words:
        stem = word
        if len(stem) > 5 and stem.endswith("ing"):
            stem = stem[:-3]
        elif len(stem) > 4 and stem.endswith("ied"):
            stem = stem[:-3] + "y"
        elif len(stem) > 4 and stem.endswith("ed"):
            stem = stem[:-2]
        elif len(stem) > 4 and stem.endswith("es"):
            stem = stem[:-2]
        elif len(stem) > 3 and stem.endswith("s"):
            stem = stem[:-1]
        if len(stem) > 2 and stem[-1] == stem[-2] and stem[-1] in "bcdfghjklmnpqrstvwxyz":
            stem = stem[:-1]
        stems.append(stem)
    return stems

class SimpleBM25:
    def __init__(self, tokenized_corpus, k1=1.5, b=0.75):
        self.corpus = tokenized_corpus
        self.k1 = k1
        self.b = b
        self.N = len(tokenized_corpus)
        self.doc_lens = [len(doc) for doc in tokenized_corpus]
        self.avgdl = sum(self.doc_lens) / max(self.N, 1)
        self.doc_freq = Counter()
        self.term_freqs = []
        for doc in tokenized_corpus:
            tf = Counter(doc)
            self.term_freqs.append(tf)
            for term in tf:
                self.doc_freq[term] += 1
        self.idf = {term: math.log(1 + (self.N - df + 0.5) / (df + 0.5)) for term, df in self.doc_freq.items()}

    def get_scores(self, query_tokens):
        scores = np.zeros(self.N)
        for i, doc_tf in enumerate(self.term_freqs):
            dl = self.doc_lens[i]
            score = 0.0
            for term in query_tokens:
                if term not in doc_tf:
                    continue
                tf = doc_tf[term]
                idf = self.idf.get(term, 0.0)
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                score += idf * (tf * (self.k1 + 1)) / denom
            scores[i] = score
        return scores

try:
    from gensim.models import Word2Vec
    GENSIM_AVAILABLE = True
except Exception:
    GENSIM_AVAILABLE = False

@st.cache_resource
def build_indexes(df):
    texts = df["text"].fillna("").tolist()
    tokenized = [custom_tokenize(t) for t in texts]
    vectorizer = TfidfVectorizer(max_features=1500, ngram_range=(1, 2), stop_words="english")
    tfidf = vectorizer.fit_transform(texts)
    bm25 = SimpleBM25([custom_stem(t) for t in tokenized])

    if GENSIM_AVAILABLE:
        model = Word2Vec(sentences=tokenized, vector_size=100, window=5, min_count=1, workers=1, sg=1, epochs=50, seed=42)
        def doc_vec(tokens):
            vecs = [model.wv[w] for w in tokens if w in model.wv]
            return np.mean(vecs, axis=0) if vecs else np.zeros(100)
        doc_vectors = np.vstack([doc_vec(doc) for doc in tokenized])
        lookup = model.wv
    else:
        vocab = sorted(set(w for doc in tokenized for w in doc))
        token_vectors = {}
        for token in vocab:
            seed = abs(hash(token)) % (2**32)
            rng = np.random.default_rng(seed)
            token_vectors[token] = rng.normal(size=100)
        def doc_vec(tokens):
            vecs = [token_vectors[w] for w in tokens if w in token_vectors]
            return np.mean(vecs, axis=0) if vecs else np.zeros(100)
        doc_vectors = np.vstack([doc_vec(doc) for doc in tokenized])
        lookup = token_vectors
    return tokenized, vectorizer, tfidf, bm25, lookup, doc_vectors

def cosine_scores(query_vector, matrix):
    q = query_vector.reshape(1, -1)
    denom = (np.linalg.norm(matrix, axis=1) * np.linalg.norm(q) + 1e-9)
    return matrix.dot(q.T).ravel() / denom

def search(df, method, query, tokenized, vectorizer, tfidf, bm25, lookup, doc_vectors, top_k=5):
    if method == "TF-IDF":
        q_vec = vectorizer.transform([query])
        scores = cosine_similarity(q_vec, tfidf).ravel()
    elif method == "BM25":
        scores = bm25.get_scores(custom_stem(custom_tokenize(query)))
    else:
        q_tokens = custom_tokenize(query)
        vecs = [lookup[w] for w in q_tokens if w in lookup]
        q_vec = np.mean(vecs, axis=0) if vecs else np.zeros(doc_vectors.shape[1])
        scores = cosine_scores(q_vec, doc_vectors)

    top_idx = scores.argsort()[::-1][:top_k]
    return pd.DataFrame({
        "Rank": range(1, top_k + 1),
        "Title": df.iloc[top_idx]["title"].values,
        "Source": df.iloc[top_idx]["source"].values,
        "Article URL": df.iloc[top_idx]["article_url"].values,
        "Score": scores[top_idx],
    })

def main():
    st.set_page_config(page_title="Assignment 1 News Search", layout="wide")
    st.title("News Search App")
    st.write("Keyword search over the Assignment 1 news dataset using TF-IDF, BM25, and Word2Vec.")
    df = load_data()
    tokenized, vectorizer, tfidf, bm25, lookup, doc_vectors = build_indexes(df)
    query = st.text_input("Enter your query", value="AI chip market")
    method = st.selectbox("Choose retrieval method", ["TF-IDF", "BM25", "Word2Vec"])
    top_k = st.slider("Number of results", min_value=3, max_value=10, value=5)
    if query:
        results = search(df, method, query, tokenized, vectorizer, tfidf, bm25, lookup, doc_vectors, top_k=top_k)
        st.subheader(f"Top {top_k} results using {method}")
        st.dataframe(results, use_container_width=True)
        st.markdown("### Notes on comparison")
        st.markdown("- TF-IDF is best for exact keyword overlap.")
        st.markdown("- BM25 often ranks short term-dense documents better.")
        st.markdown("- Word2Vec can surface semantically related documents.")

if __name__ == "__main__":
    main()
