"""
Problem 2 Open-Question Chatbot App — SEC 10-K QA

This Streamlit app lets users ask NEW open-ended questions, not just select stored
evaluation questions.

It runs locally over your SEC corpus files.

Recommended folder:
    Final/
      problem2_open_chatbot_app.py
      03_problem2_package/
      05_data/
      ...

Run:
    cd 'data/Final'
    streamlit run problem2_open_chatbot_app.py

Optional with explicit corpus path:
    streamlit run problem2_open_chatbot_app.py -- --corpus_dir "./05_data/sec_corpus_clean"

Optional Ollama generation:
    ollama serve
    ollama pull qwen2.5
    streamlit run problem2_open_chatbot_app.py
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


SECTION_KEYWORDS = {
    "Item1": ["item1", "business"],
    "Item1A": ["item1a", "risk"],
    "Item7": ["item7", "management", "discussion", "mda", "md&a"],
    "Item8": ["item8", "financial"],
}


@dataclass
class Chunk:
    chunk_id: int
    ticker: str
    fiscal_year: str
    section: str
    source_file: str
    text: str
    score: float = 0.0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_dir", default=None, help="Folder containing SEC cleaned .txt files")
    parser.add_argument("--model", default="qwen2.5", help="Ollama model name, e.g. qwen2.5, llama3, mistral")
    args, _ = parser.parse_known_args()
    return args


def find_corpus_dir(user_dir: Optional[str] = None) -> Path:
    roots = []
    if user_dir:
        roots.append(Path(user_dir).expanduser())
    roots.extend([Path.cwd(), Path(__file__).resolve().parent])

    candidate_names = [
        "sec_corpus_clean",
        "example_sec_corpus_clean",
        "sec_corpus",
        "corpus",
    ]

    checked = []
    for root in roots:
        root = root.resolve()
        checked.append(str(root))

        if root.exists() and any(root.glob("*.txt")):
            return root

        for name in candidate_names:
            matches = [p for p in root.rglob(name) if p.is_dir()]
            for m in matches:
                if any(m.rglob("*.txt")):
                    return m

        txt_dirs = {}
        for f in root.rglob("*.txt"):
            txt_dirs[f.parent] = txt_dirs.get(f.parent, 0) + 1
        if txt_dirs:
            return max(txt_dirs, key=txt_dirs.get)

    raise FileNotFoundError(
        "Could not find a corpus folder with .txt files.\n"
        "Checked:\n- " + "\n- ".join(checked) + "\n\n"
        "Pass --corpus_dir explicitly, for example:\n"
        'streamlit run problem2_open_chatbot_app.py -- --corpus_dir "./05_data/sec_corpus_clean"'
    )


def infer_metadata(path: Path) -> Tuple[str, str, str]:
    name = path.stem
    upper = name.upper()

    ticker = "UNKNOWN"
    year = "UNKNOWN"
    section = "UNKNOWN"

    parts = re.split(r"[_\-\s]+", name)

    for p in parts:
        if re.fullmatch(r"[A-Z]{1,5}", p.upper()):
            ticker = p.upper()
            break

    year_match = re.search(r"(20[0-2][0-9])", name)
    if year_match:
        year = year_match.group(1)

    lower = name.lower().replace(" ", "")
    for sec, keys in SECTION_KEYWORDS.items():
        if any(k in lower for k in keys):
            section = sec
            break

    return ticker, year, section


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_words: int = 220, overlap_words: int = 50) -> List[str]:
    words = text.split()
    if not words:
        return []

    chunks = []
    step = max(1, chunk_words - overlap_words)
    for start in range(0, len(words), step):
        end = start + chunk_words
        part = " ".join(words[start:end]).strip()
        if len(part.split()) >= 30:
            chunks.append(part)
        if end >= len(words):
            break
    return chunks


@st.cache_data(show_spinner=False)
def load_chunks(corpus_dir_str: str, chunk_words: int, overlap_words: int) -> pd.DataFrame:
    corpus_dir = Path(corpus_dir_str)
    rows = []
    chunk_id = 0

    files = sorted(corpus_dir.rglob("*.txt"))
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        text = clean_text(text)
        if len(text) < 200:
            continue

        ticker, year, section = infer_metadata(path)

        for chunk in chunk_text(text, chunk_words=chunk_words, overlap_words=overlap_words):
            rows.append({
                "chunk_id": chunk_id,
                "ticker": ticker,
                "fiscal_year": year,
                "section": section,
                "source_file": str(path.relative_to(corpus_dir)),
                "text": chunk,
            })
            chunk_id += 1

    return pd.DataFrame(rows)


@st.cache_resource(show_spinner=False)
def build_vectorizer(texts_tuple: Tuple[str, ...]):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=50000,
        ngram_range=(1, 2),
        min_df=1,
    )
    matrix = vectorizer.fit_transform(list(texts_tuple))
    return vectorizer, matrix


def retrieve(df: pd.DataFrame, question: str, ticker: str, year: str, section: str, top_k: int) -> List[Chunk]:
    filtered = df.copy()

    if ticker != "All" and "ticker" in filtered:
        filtered = filtered[filtered["ticker"].astype(str) == ticker]
    if year != "All" and "fiscal_year" in filtered:
        filtered = filtered[filtered["fiscal_year"].astype(str) == year]
    if section != "All" and "section" in filtered:
        filtered = filtered[filtered["section"].astype(str) == section]

    if filtered.empty:
        return []

    texts = tuple(filtered["text"].astype(str).tolist())
    vectorizer, matrix = build_vectorizer(texts)
    q_vec = vectorizer.transform([question])
    scores = cosine_similarity(q_vec, matrix).ravel()

    filtered = filtered.copy()
    filtered["score"] = scores
    top = filtered.sort_values("score", ascending=False).head(top_k)

    chunks = []
    for _, row in top.iterrows():
        chunks.append(Chunk(
            chunk_id=int(row["chunk_id"]),
            ticker=str(row["ticker"]),
            fiscal_year=str(row["fiscal_year"]),
            section=str(row["section"]),
            source_file=str(row["source_file"]),
            text=str(row["text"]),
            score=float(row["score"]),
        ))
    return chunks


def build_prompt(question: str, chunks: List[Chunk]) -> str:
    context_blocks = []
    for i, c in enumerate(chunks, start=1):
        context_blocks.append(
            f"[Source {i} | {c.ticker} FY{c.fiscal_year} | {c.section} | {c.source_file} | score={c.score:.3f}]\n{c.text}"
        )

    context = "\n\n".join(context_blocks)

    return f"""You are an expert financial analyst answering questions about SEC 10-K filings.

Use ONLY the provided context. If the context does not contain enough evidence, say that the filing context is insufficient.

Question:
{question}

Retrieved context:
{context}

Answer with:
1. A concise answer.
2. Evidence bullets with source numbers.
3. A short caveat if evidence is incomplete.

Answer:"""


def call_ollama(prompt: str, model: str) -> Tuple[bool, str]:
    try:
        import requests
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 350,
                },
            },
            timeout=120,
        )
        if response.status_code != 200:
            return False, f"Ollama error {response.status_code}: {response.text[:500]}"
        data = response.json()
        return True, data.get("response", "").strip()
    except Exception as exc:
        return False, str(exc)


def extractive_fallback(question: str, chunks: List[Chunk]) -> str:
    if not chunks:
        return "I could not retrieve relevant filing context for this question."

    lines = [
        "Ollama generation is unavailable, so here is an extractive answer based on retrieved evidence:",
        "",
    ]
    for i, c in enumerate(chunks[:3], start=1):
        snippet = c.text[:650]
        if len(c.text) > 650:
            snippet += "..."
        lines.append(f"Evidence {i} ({c.ticker} FY{c.fiscal_year}, {c.section}, score={c.score:.3f}): {snippet}")
        lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()

    st.set_page_config(page_title="SEC 10-K QA Chatbot", layout="wide")
    st.title("SEC Filing Intelligence Platform — Open QA Chatbot")
    st.caption("Ask open-ended questions over SEC 10-K section files. This is a local RAG-style demo.")

    with st.sidebar:
        st.header("Setup")
        corpus_dir_input = st.text_input("Corpus folder", value=args.corpus_dir or "")
        model = st.text_input("Ollama model", value=args.model)
        chunk_words = st.slider("Chunk size (words)", 120, 500, 220, 20)
        overlap_words = st.slider("Chunk overlap (words)", 0, 120, 50, 10)
        top_k = st.slider("Top-K retrieved chunks", 1, 10, 5, 1)
        use_ollama = st.checkbox("Generate answer with Ollama", value=True)

    try:
        corpus_dir = find_corpus_dir(corpus_dir_input or None)
        df = load_chunks(str(corpus_dir), chunk_words, overlap_words)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    if df.empty:
        st.error("No text chunks were loaded. Check the corpus folder.")
        st.stop()

    st.sidebar.success(f"Loaded {len(df):,} chunks")
    st.sidebar.write(f"Corpus: `{corpus_dir}`")

    tickers = ["All"] + sorted(df["ticker"].dropna().astype(str).unique())
    years = ["All"] + sorted(df["fiscal_year"].dropna().astype(str).unique())
    sections = ["All"] + sorted(df["section"].dropna().astype(str).unique())

    st.subheader("Ask a question")

    c1, c2, c3 = st.columns(3)
    with c1:
        ticker = st.selectbox("Ticker filter", tickers)
    with c2:
        year = st.selectbox("Fiscal year filter", years)
    with c3:
        section = st.selectbox("Section filter", sections)

    default_q = "What are the main cybersecurity risks described in Apple's risk factors?"
    question = st.text_area("Your question", value=default_q, height=90)

    run = st.button("Run QA", type="primary")

    if run:
        if not question.strip():
            st.warning("Please enter a question.")
            st.stop()

        chunks = retrieve(df, question, ticker, year, section, top_k)

        if not chunks:
            st.warning("No chunks matched the selected filters. Try Ticker = All, Year = All, or Section = All.")
            st.stop()

        prompt = build_prompt(question, chunks)

        with st.spinner("Generating answer..."):
            if use_ollama:
                ok, answer = call_ollama(prompt, model)
                if not ok:
                    st.warning(
                        "Ollama generation failed. Showing extractive fallback instead. "
                        "Make sure `ollama serve` is running and the model is pulled."
                    )
                    st.code(answer)
                    answer = extractive_fallback(question, chunks)
            else:
                answer = extractive_fallback(question, chunks)

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Retrieved Evidence")
        for i, c in enumerate(chunks, start=1):
            with st.expander(
                f"Source {i}: {c.ticker} FY{c.fiscal_year} | {c.section} | score={c.score:.3f} | {c.source_file}",
                expanded=(i <= 2),
            ):
                st.write(c.text)

        with st.expander("Prompt sent to model"):
            st.code(prompt)

    else:
        st.info("Enter any question, choose optional filters, and click Run QA.")

    st.divider()
    st.subheader("Corpus Preview")
    preview = df[["ticker", "fiscal_year", "section", "source_file"]].drop_duplicates().head(50)
    st.dataframe(preview, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
