import re
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title='News RAG Chatbot', layout='wide')
OUT_DIR = Path('assignment3_outputs')
st.title('News RAG Chatbot')
st.caption('Grounded answers with inline article citations')

@st.cache_data
def load_chunks():
    return pd.read_csv(OUT_DIR / 'article_chunks.csv')

@st.cache_resource
def load_model(name):
    return SentenceTransformer(name)

@st.cache_resource
def load_embeddings(name):
    safe = re.sub(r'[^A-Za-z0-9_.-]+', '__', name)
    return np.load(OUT_DIR / f'embeddings_{safe}.npy')

def prep(model_name, texts, is_query=False):
    if 'e5' in model_name.lower():
        return [('query: ' if is_query else 'passage: ') + t for t in texts]
    if 'bge' in model_name.lower() and is_query:
        return ['Represent this sentence for searching relevant passages: ' + t for t in texts]
    return texts

def first_sentence(text):
    return re.split(r'(?<=[.!?])\s+', str(text))[0]

model_name = st.sidebar.selectbox('Embedding model', [
    'sentence-transformers/all-MiniLM-L6-v2',
    'sentence-transformers/all-mpnet-base-v2',
    'sentence-transformers/multi-qa-MiniLM-L6-cos-v1',
    'BAAI/bge-small-en-v1.5',
    'intfloat/e5-small-v2',
])
top_k = st.sidebar.slider('Retrieved chunks', 1, 8, 4)
threshold = st.sidebar.slider('Minimum similarity', 0.0, 0.6, 0.15, 0.01)
chunks = load_chunks()
model = load_model(model_name)
emb = load_embeddings(model_name)

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.write(message['content'])

query = st.chat_input('Ask a question about the scraped news corpus')
if query:
    q = model.encode(prep(model_name, [query], True), normalize_embeddings=True)[0]
    scores = emb @ q
    idx = np.argsort(scores)[::-1][:top_k]
    hits = chunks.iloc[idx].copy()
    hits['score'] = scores[idx]
    if len(hits) == 0 or hits.score.max() < threshold:
        answer = 'I do not have enough information in the retrieved passages.'
    else:
        h = hits.iloc[0]
        answer = f"{first_sentence(h.text)} [{h.chunk_id} | {h.title}]"
    st.session_state.messages += [
        {'role': 'user', 'content': query},
        {'role': 'assistant', 'content': answer},
    ]
    with st.chat_message('user'):
        st.write(query)
    with st.chat_message('assistant'):
        st.write(answer)
        st.dataframe(hits[['chunk_id', 'title', 'source', 'score', 'text']], use_container_width=True)
