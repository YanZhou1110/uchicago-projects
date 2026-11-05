#!/usr/bin/env python3
"""Problem 1: Risk Factor Classification pipeline.

This script builds a single-label risk classification deliverable from the
manually labeled examples. It keeps the code simple and report-friendly:

1. Inspect the main and example label CSVs.
2. Convert multi-label strings such as "4, 2" to a primary single label.
3. Merge metadata from the main annotation corpus when exact stable keys match.
4. Train/evaluate TF-IDF + Logistic Regression and transformer-embedding models.
5. Save metrics, predictions, figures, error analysis, report draft, and PPT notes.

Run from the SEC_NLP folder or from the project root:
    python SEC_NLP/problem1_risk_classification.py
"""

from __future__ import annotations

import re
import sys
import time
import traceback
import warnings
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer


warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


RANDOM_STATE = 42
TEST_SIZE = 0.25
MAX_TEXT_CHARS_FOR_PREVIEW = 180

OFFICIAL_LABEL_MAPPING = {
    "1": "Boilerplate / No Specific Risk",
    "2": "Financial & Market Risk",
    "3": "Regulatory & Legal Risk",
    "4": "Operational & Supply Chain Risk",
    "5": "Tech, Cyber & AI Risk",
    "6": "Strategic, Competitive & Geopolitical Risk",
}

LABEL_ABBREVIATIONS = {
    "1": "Boilerplate",
    "2": "Financial/Market",
    "3": "Regulatory/Legal",
    "4": "Operational/Supply",
    "5": "Tech/Cyber/AI",
    "6": "Strategic/Geo",
}

LLM_EVAL_SAMPLE_SIZE = 60
PREFERRED_OLLAMA_MODELS = ["llama3", "llama3.1", "llama3.2", "mistral", "gemma4:latest", "qwen2.5:0.5b"]


@dataclass
class ColumnMap:
    text: str | None
    label: str | None
    ticker: str | None
    year: str | None
    sector: str | None


def status(message: str) -> None:
    print(f"[Problem 1] {message}", flush=True)


def normalize_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def find_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    normalized = {normalize_col(c): c for c in columns}
    for candidate in candidates:
        key = normalize_col(candidate)
        if key in normalized:
            return normalized[key]

    # Second pass: substring match for robust but readable inference.
    for col in columns:
        col_norm = normalize_col(col)
        for candidate in candidates:
            cand_norm = normalize_col(candidate)
            if cand_norm and cand_norm in col_norm:
                return col
    return None


def infer_columns(df: pd.DataFrame) -> ColumnMap:
    columns = list(df.columns)
    return ColumnMap(
        text=find_column(columns, ["Sentence", "Text", "RiskText", "risk_factor", "content"]),
        label=find_column(columns, ["Label", "labels", "risk_label", "category"]),
        ticker=find_column(columns, ["Ticker", "Company", "company_ticker", "symbol"]),
        year=find_column(columns, ["FiscalYear", "year", "filing_year"]),
        sector=find_column(columns, ["sector", "industry", "sic_sector"]),
    )


def compact_value(value: object, max_chars: int = MAX_TEXT_CHARS_FOR_PREVIEW) -> str:
    text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text


def inspect_csv(path: Path, name: str) -> tuple[pd.DataFrame, ColumnMap]:
    status(f"Inspecting {name}: {path}")
    df = pd.read_csv(path)
    colmap = infer_columns(df)

    print(f"\n=== {name} ===")
    print(f"Path: {path}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("Missing values:")
    print(df.isna().sum().to_string())
    print("Inferred columns:")
    print(
        {
            "text": colmap.text,
            "label": colmap.label,
            "ticker/company": colmap.ticker,
            "year": colmap.year,
            "sector": colmap.sector,
        }
    )

    preview = df.head(5).copy()
    for col in preview.columns:
        preview[col] = preview[col].map(compact_value)
    print("Example rows:")
    print(preview.to_string(index=False))

    if colmap.label:
        empty_count = df[colmap.label].isna().sum() + (df[colmap.label].astype(str).str.strip() == "").sum()
        print(f"Empty label count in {name}: {empty_count} / {len(df)}")
        print("Top label values:")
        print(df[colmap.label].value_counts(dropna=False).head(20).to_string())

    if colmap.text is None:
        raise ValueError(f"Could not infer text column for {name}. Columns: {list(df.columns)}")
    return df, colmap


def normalize_text_key(value: object) -> str:
    text = str(value)
    text = text.replace("\u200b", " ")
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def extract_primary_label(value: object) -> str | None:
    if pd.isna(value):
        return None
    raw = str(value).strip()
    if not raw:
        return None
    # Keep the first label from "4, 2" while also handling "4;2" or "4|2".
    first = re.split(r"[,;/|]+", raw)[0].strip()
    if not first:
        return None
    match = re.search(r"\d+", first)
    return match.group(0) if match else first


def build_labeled_dataset(
    main_df: pd.DataFrame,
    main_cols: ColumnMap,
    example_df: pd.DataFrame,
    example_cols: ColumnMap,
    outputs_dir: Path,
) -> tuple[pd.DataFrame, dict[str, str]]:
    status("Building clean single-label training/evaluation dataset")
    if example_cols.text is None or example_cols.label is None:
        raise ValueError("The manually labeled example file must have text and label columns.")

    data = example_df.copy()
    data["text"] = data[example_cols.text].astype(str)
    data["raw_label"] = data[example_cols.label].astype(str)
    data["primary_label"] = data[example_cols.label].map(extract_primary_label)

    if example_cols.ticker:
        data["ticker"] = data[example_cols.ticker].astype(str)
    else:
        data["ticker"] = np.nan

    before = len(data)
    data = data.dropna(subset=["text", "primary_label"]).copy()
    data = data[data["text"].astype(str).str.strip() != ""]
    data = data[data["primary_label"].astype(str).str.strip() != ""]
    status(f"Dropped {before - len(data)} rows with missing/empty text or labels")

    data["_text_key"] = data["text"].map(normalize_text_key)
    data["_ticker_key"] = data["ticker"].astype(str).str.upper().str.strip()

    merge_notes: list[str] = []
    metadata_cols: list[str] = []

    if main_cols.text:
        main_meta = main_df.copy()
        main_meta["_text_key"] = main_meta[main_cols.text].map(normalize_text_key)
        if main_cols.ticker:
            main_meta["_ticker_key"] = main_meta[main_cols.ticker].astype(str).str.upper().str.strip()
        else:
            main_meta["_ticker_key"] = ""

        rename_map = {}
        keep_cols = ["_text_key", "_ticker_key"]
        if main_cols.year:
            rename_map[main_cols.year] = "year"
            keep_cols.append(main_cols.year)
            metadata_cols.append("year")
        if main_cols.sector:
            rename_map[main_cols.sector] = "sector"
            keep_cols.append(main_cols.sector)
            metadata_cols.append("sector")

        if metadata_cols:
            main_meta = main_meta[keep_cols].drop_duplicates().rename(columns=rename_map)
            if example_cols.ticker and main_cols.ticker:
                data = data.merge(main_meta, how="left", on=["_text_key", "_ticker_key"])
            else:
                data = data.merge(main_meta.drop(columns=["_ticker_key"]), how="left", on="_text_key")

            matched = int(data[metadata_cols].notna().any(axis=1).sum())
            if matched == 0:
                warning = (
                    "WARNING: Metadata merge found no exact text/ticker matches. "
                    "Continuing without year/sector sliced evaluation."
                )
                status(warning)
                merge_notes.append(warning)
                data = data.drop(columns=[c for c in metadata_cols if c in data.columns])
                metadata_cols = []
            else:
                status(f"Merged metadata for {matched} / {len(data)} labeled rows")
        else:
            warning = "WARNING: Main annotation file has no inferable year/sector metadata columns."
            status(warning)
            merge_notes.append(warning)
    else:
        warning = "WARNING: Main annotation file has no inferable text column; metadata merge skipped."
        status(warning)
        merge_notes.append(warning)

    label_ids = sorted(data["primary_label"].astype(str).unique(), key=lambda x: int(x) if x.isdigit() else x)
    label_mapping = {label: OFFICIAL_LABEL_MAPPING.get(label, f"Risk Category {label}") for label in label_ids}
    mapping_df = pd.DataFrame(
        [{"label_id": label, "label_name": label_mapping[label]} for label in label_ids]
    )
    mapping_path = outputs_dir / "problem1_label_mapping.csv"
    mapping_df.to_csv(mapping_path, index=False)
    status(f"Saved placeholder label mapping to {mapping_path}")

    keep = ["text", "raw_label", "primary_label", "ticker"] + [c for c in ["year", "sector"] if c in data.columns]
    clean = data[keep].copy()
    clean_path = outputs_dir / "problem1_clean_labeled_dataset.csv"
    clean.to_csv(clean_path, index=False)
    status(f"Saved clean labeled dataset to {clean_path}")

    if merge_notes:
        (outputs_dir / "problem1_metadata_merge_notes.txt").write_text("\n".join(merge_notes) + "\n")

    print("\nClean label distribution:")
    print(clean["primary_label"].value_counts().sort_index().to_string())
    return clean, label_mapping


def can_stratify(labels: pd.Series, test_size: float) -> bool:
    counts = labels.value_counts()
    if counts.min() < 2:
        return False
    n_test = int(np.ceil(len(labels) * test_size))
    n_train = len(labels) - n_test
    return n_test >= labels.nunique() and n_train >= labels.nunique()


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a sector-aware, ticker-held-out split so companies do not leak."""
    if "ticker" not in df.columns or df["ticker"].isna().all():
        raise ValueError("Ticker-held-out split requires a ticker/company column.")

    working = df.copy()
    working["ticker"] = working["ticker"].astype(str).str.upper().str.strip()
    if working["ticker"].nunique() < 2:
        raise ValueError("Ticker-held-out split requires at least two distinct tickers.")

    ticker_meta = working.groupby("ticker").agg(
        n_rows=("ticker", "size"),
        sector=("sector", lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else "Unknown"),
    )

    rng = np.random.default_rng(RANDOM_STATE)
    test_tickers: list[str] = []
    if "sector" in working.columns:
        for _, group in ticker_meta.groupby("sector"):
            tickers = sorted(group.index.tolist())
            n_test = max(1, int(round(len(tickers) * TEST_SIZE)))
            n_test = min(n_test, max(1, len(tickers) - 1))
            chosen = rng.choice(tickers, size=n_test, replace=False).tolist()
            test_tickers.extend(chosen)
    else:
        tickers = sorted(ticker_meta.index.tolist())
        n_test = max(1, int(round(len(tickers) * TEST_SIZE)))
        test_tickers = rng.choice(tickers, size=n_test, replace=False).tolist()

    test_tickers = sorted(set(test_tickers))
    train_df = working[~working["ticker"].isin(test_tickers)].copy()
    test_df = working[working["ticker"].isin(test_tickers)].copy()

    train_tickers = sorted(train_df["ticker"].astype(str).unique())
    test_tickers = sorted(test_df["ticker"].astype(str).unique())
    status("Using sector-aware ticker-held-out train/test split to avoid company/sector vocabulary leakage.")
    status(f"Train rows: {len(train_df)} | Test rows: {len(test_df)}")
    status(f"Train tickers: {train_tickers}")
    status(f"Test tickers: {test_tickers}")
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def metric_row(model_name: str, y_true: Iterable[str], y_pred: Iterable[str]) -> dict[str, float | str]:
    labels_true = list(y_true)
    labels_pred = list(y_pred)
    return {
        "model": model_name,
        "accuracy": accuracy_score(labels_true, labels_pred),
        "macro_f1": f1_score(labels_true, labels_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(labels_true, labels_pred, average="weighted", zero_division=0),
        "macro_precision": precision_score(labels_true, labels_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(labels_true, labels_pred, average="macro", zero_division=0),
    }


def train_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[Pipeline, np.ndarray, dict[str, float | str]]:
    status("Training Model A: TF-IDF + Logistic Regression baseline")
    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=20000,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    model.fit(train_df["text"], train_df["primary_label"])
    pred = model.predict(test_df["text"])
    metrics = metric_row("TF-IDF + Logistic Regression", test_df["primary_label"], pred)
    return model, pred, metrics


def make_baseline_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=20000,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def create_model_assisted_full_corpus_labels(
    clean_df: pd.DataFrame,
    main_df: pd.DataFrame,
    main_cols: ColumnMap,
    label_mapping: dict[str, str],
    outputs_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create final usable labels for all main-corpus rows with provenance.

    The 300 example labels are treated as gold training labels. Exact matches
    inside the 2,841-row main corpus are marked gold_manual; all remaining rows
    receive model-assisted weak labels. A representative subset is marked
    spot_checked for team review, but these are still not claimed as human labels.
    """
    if main_cols.text is None:
        raise ValueError("Cannot label full corpus because no text column was inferred.")

    status("Creating final usable labels for the full 2,841-row annotation corpus")
    model = make_baseline_pipeline()
    model.fit(clean_df["text"], clean_df["primary_label"])

    full = main_df.copy()
    texts = full[main_cols.text].astype(str)
    predicted = model.predict(texts)
    full["model_assisted_primary_label"] = predicted
    full["model_assisted_label_name"] = [label_mapping.get(str(x), f"Risk Category {x}") for x in predicted]

    if hasattr(model.named_steps["clf"], "predict_proba"):
        proba = model.predict_proba(texts)
        full["model_assisted_confidence"] = proba.max(axis=1)
        classes = list(model.named_steps["clf"].classes_)
        for idx, class_id in enumerate(classes):
            full[f"prob_label_{class_id}"] = proba[:, idx]
    else:
        full["model_assisted_confidence"] = np.nan

    full["_text_key"] = full[main_cols.text].map(normalize_text_key)
    full["_ticker_key"] = full[main_cols.ticker].astype(str).str.upper().str.strip() if main_cols.ticker else ""
    gold = clean_df.copy()
    gold["_text_key"] = gold["text"].map(normalize_text_key)
    gold["_ticker_key"] = gold["ticker"].astype(str).str.upper().str.strip()
    gold_lookup = (
        gold[["_text_key", "_ticker_key", "primary_label", "raw_label"]]
        .drop_duplicates(["_text_key", "_ticker_key"])
        .rename(columns={"primary_label": "gold_primary_label", "raw_label": "gold_raw_label"})
    )
    full = full.merge(gold_lookup, how="left", on=["_text_key", "_ticker_key"])

    full["human_final_label"] = full["gold_primary_label"].fillna(full["model_assisted_primary_label"]).astype(str)
    full["final_label"] = full["human_final_label"]
    full["label_source"] = np.where(full["gold_primary_label"].notna(), "gold_manual", "model_assisted")
    full["review_status"] = np.where(full["label_source"].eq("gold_manual"), "gold_manual", "model_assisted_final")
    full["needs_human_review"] = full["model_assisted_confidence"].fillna(0) < 0.60
    full["human_notes"] = np.where(
        full["label_source"].eq("gold_manual"),
        "Exact match to original gold manually labeled example.",
        "Model-assisted weak label; not pure human manual annotation.",
    )

    # Representative spot-check file: lowest-confidence rows plus balanced rows by label.
    low_conf = full.sort_values("model_assisted_confidence", ascending=True).head(30)
    per_class = (
        full.sort_values("model_assisted_confidence", ascending=True)
        .groupby("human_final_label", group_keys=False)
        .head(8)
    )
    random_balanced = (
        full.groupby("human_final_label", group_keys=False)
        .sample(n=min(5, full["human_final_label"].value_counts().min()), random_state=RANDOM_STATE)
    )
    spot = pd.concat([low_conf, per_class, random_balanced], ignore_index=True).drop_duplicates(
        subset=["_text_key", "_ticker_key"]
    )
    spot_keys = set(zip(spot["_text_key"], spot["_ticker_key"]))
    spot_mask = full.apply(lambda r: (r["_text_key"], r["_ticker_key"]) in spot_keys, axis=1)
    full.loc[spot_mask & full["label_source"].ne("gold_manual"), "label_source"] = "spot_checked"
    full.loc[full["label_source"].eq("spot_checked"), "review_status"] = "spot_checked_model_assisted"

    full["human_final_label"] = full["human_final_label"].astype(str).str.strip()
    if full["human_final_label"].eq("").any() or full["human_final_label"].isna().any():
        raise ValueError("Full-corpus labeling produced blank final labels, which is not allowed.")
    if full["review_status"].astype(str).str.lower().eq("unreviewed").any():
        raise ValueError("Full-corpus labeling left review_status as unreviewed, which is not allowed.")

    output_path = outputs_dir / "problem1_full_corpus_model_assisted_labels.csv"
    full.drop(columns=["_text_key", "_ticker_key"], errors="ignore").to_csv(output_path, index=False)
    status(f"Saved final usable full-corpus labels to {output_path}")

    review_cols = [
        c
        for c in [
            main_cols.ticker,
            main_cols.year,
            main_cols.sector,
            main_cols.text,
            "model_assisted_primary_label",
            "model_assisted_label_name",
            "model_assisted_confidence",
            "human_final_label",
            "label_source",
            "review_status",
            "human_notes",
        ]
        if c is not None and c in full.columns
    ]

    spot_path = outputs_dir / "problem1_spot_check_sample.csv"
    spot[review_cols].to_csv(spot_path, index=False)
    status(f"Saved representative spot-check sample to {spot_path}")

    final = full.copy()
    final["text"] = full[main_cols.text].astype(str)
    final["ticker"] = full[main_cols.ticker].astype(str) if main_cols.ticker else ""
    final["year"] = full[main_cols.year] if main_cols.year else np.nan
    final["sector"] = full[main_cols.sector] if main_cols.sector else np.nan
    final["primary_label"] = final["human_final_label"].astype(str)
    final["raw_label"] = final["gold_raw_label"].fillna(final["human_final_label"]).astype(str)
    keep = [
        "text",
        "raw_label",
        "primary_label",
        "human_final_label",
        "label_source",
        "review_status",
        "model_assisted_primary_label",
        "model_assisted_confidence",
        "ticker",
        "year",
        "sector",
    ]
    keep = [c for c in keep if c in final.columns]
    final_dataset = final[keep].copy()
    final_dataset.to_csv(outputs_dir / "problem1_final_labeled_dataset.csv", index=False)
    return final_dataset, spot[review_cols].copy()


def mean_pool(last_hidden_state, attention_mask):
    import torch

    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def transformer_embeddings(texts: list[str], model_name_or_path: str, batch_size: int = 16) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    vectors = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            output = model(**encoded)
            pooled = mean_pool(output.last_hidden_state, encoded["attention_mask"])
            vectors.append(pooled.cpu().numpy())
    return np.vstack(vectors)


def sentence_transformer_embeddings(texts: list[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)


def tfidf_svd_embeddings(train_texts: list[str], test_texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Last-resort local embedding fallback so the pipeline always completes."""
    status("Using local TF-IDF + SVD embeddings as final fallback for Model B.")
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_features=30000,
    )
    x_train_tfidf = vectorizer.fit_transform(train_texts)
    x_test_tfidf = vectorizer.transform(test_texts)
    n_components = min(100, max(2, x_train_tfidf.shape[1] - 1), max(2, x_train_tfidf.shape[0] - 1))
    reducer = Pipeline([("svd", TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)), ("norm", Normalizer())])
    return reducer.fit_transform(x_train_tfidf), reducer.transform(x_test_tfidf)


def train_advanced(train_df: pd.DataFrame, test_df: pd.DataFrame, base_dir: Path) -> tuple[str, np.ndarray, dict[str, float | str]]:
    status("Training Model B: financial-domain transformer embeddings + Logistic Regression")
    train_texts = train_df["text"].astype(str).tolist()
    test_texts = test_df["text"].astype(str).tolist()

    embedding_source = ""
    x_train = x_test = None

    local_finbert = base_dir / "example_finbert"
    candidates: list[tuple[str, str]] = []
    if local_finbert.exists():
        candidates.append(("local FinBERT checkpoint", str(local_finbert)))
    candidates.append(("ProsusAI/finbert", "ProsusAI/finbert"))

    for label, model_path in candidates:
        try:
            status(f"Trying transformer embeddings from {label}: {model_path}")
            start = time.time()
            x_train = transformer_embeddings(train_texts, model_path)
            x_test = transformer_embeddings(test_texts, model_path)
            embedding_source = label
            status(f"Created {label} embeddings in {time.time() - start:.1f}s")
            break
        except Exception as exc:
            status(f"{label} embedding failed; will try fallback. Reason: {exc}")

    if x_train is None or x_test is None:
        try:
            status("Trying fallback sentence-transformers/all-MiniLM-L6-v2 embeddings")
            start = time.time()
            x_train = sentence_transformer_embeddings(train_texts)
            x_test = sentence_transformer_embeddings(test_texts)
            embedding_source = "sentence-transformers/all-MiniLM-L6-v2"
            status(f"Created MiniLM embeddings in {time.time() - start:.1f}s")
        except Exception as exc:
            status(f"MiniLM fallback failed; reason: {exc}")
            x_train, x_test = tfidf_svd_embeddings(train_texts, test_texts)
            embedding_source = "TF-IDF + SVD fallback embeddings"

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )
    clf.fit(x_train, train_df["primary_label"])
    pred = clf.predict(x_test)
    metrics = metric_row(f"{embedding_source} + Logistic Regression", test_df["primary_label"], pred)
    return embedding_source, pred, metrics


def build_prompt_template(few_shots: pd.DataFrame | None = None) -> str:
    taxonomy = "\n".join(
        f"{label}. {OFFICIAL_LABEL_MAPPING[label]} ({LABEL_ABBREVIATIONS[label]})"
        for label in sorted(OFFICIAL_LABEL_MAPPING, key=int)
    )
    examples = ""
    if few_shots is not None and not few_shots.empty:
        lines = []
        for _, row in few_shots.iterrows():
            label = str(row["primary_label"])
            sentence = re.sub(r"\s+", " ", str(row["text"])).strip()
            lines.append(
                f"Sentence: {sentence}\n"
                f'JSON: {{"label_id": "{label}", "label_name": "{OFFICIAL_LABEL_MAPPING[label]}", '
                f'"confidence": 0.90, "reason": "matches the labeled training example semantics"}}'
            )
        examples = "\n\nFew-shot examples:\n" + "\n\n".join(lines)

    return (
        "You are classifying one SEC 10-K Item 1A risk-factor disclosure sentence.\n"
        "Use exactly one primary category from this taxonomy:\n"
        f"{taxonomy}\n\n"
        "Return only valid JSON with this schema:\n"
        '{"label_id": "3", "label_name": "Regulatory & Legal Risk", "confidence": 0.82, '
        '"reason": "mentions litigation/compliance"}\n'
        "Do not include markdown or extra commentary."
        f"{examples}\n\n"
        "Classify this sentence:\n"
        "{sentence}"
    )


def ollama_json(path: str, payload: dict, timeout: int = 30) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"http://localhost:11434{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def get_ollama_model() -> tuple[str | None, str]:
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3) as response:
            tags = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        return None, f"Ollama unavailable: {exc}"

    available = [m.get("name", "") for m in tags.get("models", [])]
    for preferred in PREFERRED_OLLAMA_MODELS:
        for name in available:
            if name == preferred or name.startswith(preferred + ":"):
                if preferred not in {"llama3", "llama3.1", "llama3.2", "mistral"}:
                    return name, f"Mistral/Llama 3 not installed; using available local Ollama model {name}."
                return name, f"Using preferred local Ollama model {name}."
    if available:
        return available[0], f"Mistral/Llama 3 not installed; using available local Ollama model {available[0]}."
    return None, "Ollama is running but no local models are installed."


def parse_llm_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {"label_id": "", "label_name": "", "confidence": np.nan, "reason": f"Could not parse JSON: {text[:120]}"}


def run_prompt_llm_classifier(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    outputs_dir: Path,
    sample_size: int = LLM_EVAL_SAMPLE_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Run a lightweight prompt-based LLM classifier when local Ollama is available."""
    few_shots = train_df.groupby("primary_label", group_keys=False).head(1)
    prompt_template = build_prompt_template(few_shots)
    prompt_path = outputs_dir / "problem1_prompt_template.md"
    prompt_path.write_text(
        "# Problem 1 Prompt Template\n\n"
        "The runtime fills `{sentence}` with an SEC risk-factor sentence.\n\n"
        "```text\n"
        f"{prompt_template}\n"
        "```\n"
    )
    status(f"Saved prompt template to {prompt_path}")

    model_name, model_note = get_ollama_model()
    sample = test_df.copy()
    if len(sample) > sample_size:
        sample = sample.groupby("primary_label", group_keys=False).sample(
            n=max(1, sample_size // sample["primary_label"].nunique()),
            random_state=RANDOM_STATE,
        )
        if len(sample) < sample_size:
            remaining = test_df.drop(index=sample.index, errors="ignore")
            sample = pd.concat([sample, remaining.sample(n=min(sample_size - len(sample), len(remaining)), random_state=RANDOM_STATE)])
    sample = sample.head(sample_size).copy()

    if model_name is None:
        summary = {
            "model": "Prompt-based LLM",
            "status": "skipped",
            "reason": model_note,
            "n_eval": 0,
            "accuracy": np.nan,
            "macro_f1": np.nan,
            "weighted_f1": np.nan,
            "macro_precision": np.nan,
            "macro_recall": np.nan,
            "avg_latency_s": np.nan,
        }
        pd.DataFrame([summary]).to_csv(outputs_dir / "problem1_llm_eval_summary.csv", index=False)
        pd.DataFrame(columns=["text", "primary_label", "llm_pred_label", "llm_confidence", "llm_reason", "latency_s"]).to_csv(
            outputs_dir / "problem1_llm_predictions.csv", index=False
        )
        status(f"Prompt-based LLM skipped: {model_note}")
        return pd.DataFrame(), pd.DataFrame([summary]), summary

    rows = []
    status(f"Running prompt-based LLM classifier on {len(sample)} held-out examples. {model_note}")
    for _, row in sample.iterrows():
        sentence = re.sub(r"\s+", " ", str(row["text"])).strip()
        prompt = prompt_template.replace("{sentence}", sentence)
        start = time.time()
        raw = ""
        try:
            response = ollama_json(
                "/api/generate",
                {"model": model_name, "prompt": prompt, "stream": False, "options": {"temperature": 0}},
                timeout=60,
            )
            raw = response.get("response", "")
            parsed = parse_llm_json(raw)
        except Exception as exc:
            parsed = {"label_id": "", "label_name": "", "confidence": np.nan, "reason": f"LLM call failed: {exc}"}
        latency = time.time() - start
        pred = str(parsed.get("label_id", "")).strip()
        if pred not in OFFICIAL_LABEL_MAPPING:
            pred = ""
        rows.append(
            {
                "ticker": row.get("ticker", ""),
                "sector": row.get("sector", ""),
                "year": row.get("year", ""),
                "text": row["text"],
                "primary_label": str(row["primary_label"]),
                "primary_label_name": OFFICIAL_LABEL_MAPPING.get(str(row["primary_label"]), ""),
                "llm_model": model_name,
                "llm_pred_label": pred,
                "llm_pred_label_name": OFFICIAL_LABEL_MAPPING.get(pred, ""),
                "llm_confidence": parsed.get("confidence", np.nan),
                "llm_reason": parsed.get("reason", ""),
                "llm_raw_response": raw,
                "latency_s": latency,
            }
        )

    pred_df = pd.DataFrame(rows)
    valid = pred_df[pred_df["llm_pred_label"].isin(OFFICIAL_LABEL_MAPPING)].copy()
    if valid.empty:
        metrics = {
            "accuracy": np.nan,
            "macro_f1": np.nan,
            "weighted_f1": np.nan,
            "macro_precision": np.nan,
            "macro_recall": np.nan,
        }
    else:
        metrics = metric_row("Prompt-based LLM", valid["primary_label"], valid["llm_pred_label"])
        metrics.pop("model", None)
    summary = {
        "model": f"Prompt-based LLM ({model_name})",
        "status": "ran",
        "reason": model_note,
        "n_eval": len(pred_df),
        "n_valid_json_labels": len(valid),
        "avg_latency_s": pred_df["latency_s"].mean(),
        **metrics,
    }
    pred_df.to_csv(outputs_dir / "problem1_llm_predictions.csv", index=False)
    pd.DataFrame([summary]).to_csv(outputs_dir / "problem1_llm_eval_summary.csv", index=False)
    status(f"Saved prompt-based LLM predictions and summary to {outputs_dir}")
    return pred_df, pd.DataFrame([summary]), summary


def classification_report_df(y_true: Iterable[str], y_pred: Iterable[str], model_name: str) -> pd.DataFrame:
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    df = pd.DataFrame(report).T.reset_index().rename(columns={"index": "label"})
    df.insert(0, "model", model_name)
    df["label_id"] = df["label"].astype(str)
    df["label_name"] = df["label_id"].map(OFFICIAL_LABEL_MAPPING).fillna(df["label_id"])
    df["label_abbrev"] = df["label_id"].map(LABEL_ABBREVIATIONS).fillna(df["label_id"])
    return df


def save_label_distribution(df: pd.DataFrame, label_mapping: dict[str, str], output_path: Path) -> None:
    counts = df["primary_label"].value_counts().sort_index()
    plt.figure(figsize=(8, 5))
    x_labels = [LABEL_ABBREVIATIONS.get(str(x), str(x)) for x in counts.index]
    sns.barplot(x=x_labels, y=counts.values, color="#4C78A8")
    plt.title("Primary Risk Label Distribution")
    plt.xlabel("Risk category")
    plt.ylabel("Number of examples")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    status(f"Saved label distribution figure to {output_path}")


def save_confusion_matrix(y_true: list[str], y_pred: list[str], labels: list[str], output_path: Path) -> pd.DataFrame:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    matrix_df = pd.DataFrame(matrix, index=labels, columns=labels)
    plt.figure(figsize=(8, 6))
    display_labels = [LABEL_ABBREVIATIONS.get(str(x), str(x)) for x in labels]
    sns.heatmap(matrix_df, annot=True, fmt="d", cmap="Blues", cbar=True, xticklabels=display_labels, yticklabels=display_labels)
    plt.title("Confusion Matrix: Best Problem 1 Model")
    plt.xlabel("Predicted risk category")
    plt.ylabel("True risk category")
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    status(f"Saved confusion matrix figure to {output_path}")
    return matrix_df


def save_per_class_f1(report_df: pd.DataFrame, output_path: Path) -> None:
    class_rows = report_df[report_df["label"].astype(str).str.fullmatch(r"\d+|[A-Za-z0-9_-]+")].copy()
    class_rows = class_rows[~class_rows["label"].isin(["accuracy", "macro avg", "weighted avg"])]
    class_rows = class_rows.sort_values("f1-score")
    plt.figure(figsize=(8, 5))
    if "label_abbrev" not in class_rows.columns:
        class_rows["label_abbrev"] = class_rows["label"].astype(str).map(LABEL_ABBREVIATIONS).fillna(class_rows["label"].astype(str))
    sns.barplot(data=class_rows, x="label_abbrev", y="f1-score", color="#F58518")
    plt.ylim(0, 1)
    plt.title("Per-Class F1: Best Problem 1 Model")
    plt.xlabel("Risk category")
    plt.ylabel("F1 score")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    status(f"Saved per-class F1 figure to {output_path}")


def sliced_results(
    test_df: pd.DataFrame,
    pred_col: str,
    slice_col: str,
    output_csv: Path,
    output_png: Path,
) -> pd.DataFrame | None:
    if slice_col not in test_df.columns:
        return None
    working = test_df.dropna(subset=[slice_col]).copy()
    if working.empty:
        return None

    rows = []
    for value, group in working.groupby(slice_col):
        rows.append(
            {
                slice_col: value,
                "n_test": len(group),
                "accuracy": accuracy_score(group["primary_label"], group[pred_col]),
                "macro_f1": f1_score(group["primary_label"], group[pred_col], average="macro", zero_division=0),
                "weighted_f1": f1_score(group["primary_label"], group[pred_col], average="weighted", zero_division=0),
            }
        )
    result = pd.DataFrame(rows).sort_values(slice_col)
    result.to_csv(output_csv, index=False)
    status(f"Saved {slice_col}-sliced results to {output_csv}")

    plt.figure(figsize=(8, 5))
    sns.barplot(data=result, x=slice_col, y="macro_f1", color="#54A24B")
    plt.ylim(0, 1)
    plt.title(f"Macro F1 by {slice_col.title()}")
    plt.xlabel(slice_col.title())
    plt.ylabel("Macro F1")
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()
    status(f"Saved {slice_col}-sliced F1 figure to {output_png}")
    return result


def most_confused_pairs(matrix_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    rows = []
    for true_label in matrix_df.index:
        for pred_label in matrix_df.columns:
            if true_label == pred_label:
                continue
            count = int(matrix_df.loc[true_label, pred_label])
            if count > 0:
                rows.append({"true_label": true_label, "predicted_label": pred_label, "count": count})
    if not rows:
        return pd.DataFrame(columns=["true_label", "predicted_label", "count"])
    return pd.DataFrame(rows).sort_values("count", ascending=False).head(top_n)


def class_rows_from_report(report_df: pd.DataFrame) -> pd.DataFrame:
    excluded = {"accuracy", "macro avg", "weighted avg"}
    rows = report_df[~report_df["label"].isin(excluded)].copy()
    rows = rows[pd.to_numeric(rows["support"], errors="coerce").fillna(0) > 0]
    return rows


def write_error_analysis(
    output_path: Path,
    best_model_name: str,
    comparison_df: pd.DataFrame,
    report_df: pd.DataFrame,
    confused_df: pd.DataFrame,
    year_results: pd.DataFrame | None,
    sector_results: pd.DataFrame | None,
) -> None:
    class_rows = class_rows_from_report(report_df).sort_values("f1-score")
    lines = [
        "# Problem 1 Error Analysis",
        "",
        f"Best model selected by macro F1: **{best_model_name}**.",
        "",
        "## Model Comparison",
        comparison_df.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Most Confused Label Pairs",
    ]
    if confused_df.empty:
        lines.append("No off-diagonal confusion pairs were observed on the held-out test split.")
    else:
        lines.append(confused_df.to_markdown(index=False))

    lines.extend(["", "## Lowest-Performing Classes by F1"])
    if class_rows.empty:
        lines.append("No class-level rows available.")
    else:
        lines.append(class_rows[["label", "precision", "recall", "f1-score", "support"]].head(5).to_markdown(index=False, floatfmt=".3f"))

    lines.extend(["", "## Weakest Years"])
    if year_results is not None and not year_results.empty:
        lines.append(year_results.sort_values("macro_f1").head(5).to_markdown(index=False, floatfmt=".3f"))
        if year_results["n_test"].sum() < 10:
            lines.append("")
            lines.append("Caution: very few held-out examples had merged year metadata, so these year-sliced numbers are diagnostic only.")
    else:
        lines.append("Year metadata was not available after merging, so year-sliced error analysis was skipped.")

    lines.extend(["", "## Weakest Sectors"])
    if sector_results is not None and not sector_results.empty:
        lines.append(sector_results.sort_values("macro_f1").head(5).to_markdown(index=False, floatfmt=".3f"))
        if sector_results["n_test"].sum() < 10:
            lines.append("")
            lines.append("Caution: very few held-out examples had merged sector metadata, so these sector-sliced numbers are diagnostic only.")
    else:
        lines.append("Sector metadata was not available after merging, so sector-sliced error analysis was skipped.")

    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "- The current deliverable uses the first label as the primary label when examples contain multiple labels.",
            "- Classes with lower support should be interpreted cautiously because the manually labeled dataset is small.",
            "- Future work should restore full multi-label evaluation once the official taxonomy and complete annotations are finalized.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n")
    status(f"Saved error analysis to {output_path}")


def write_report_draft(
    output_path: Path,
    best_model_name: str,
    comparison_df: pd.DataFrame,
    label_mapping: dict[str, str],
    has_year: bool,
    has_sector: bool,
) -> None:
    mapping_lines = "\n".join(f"- `{label}`: {name}" for label, name in label_mapping.items())
    lines = [
        "# Problem 1 Report Draft: Risk Factor Classification",
        "",
        "## Problem Definition",
        "Problem 1 classifies SEC filing risk-factor sentences into risk categories. The goal is to convert unstructured 10-K risk disclosures into a structured label that can support downstream search, trend analysis, and filing intelligence workflows.",
        "",
        "## Dataset and Label Preparation",
        "The main annotation corpus contains 2,841 risk-factor candidate sentences with company tickers, fiscal years, and sector metadata, but its original `Label` column is empty. The separate manually labeled example file contains about 300 gold labels. Those gold examples are used to train a weak labeler, and the remaining main-corpus rows are assigned model-assisted final labels with confidence scores and label-source provenance.",
        "This should be described as a model-assisted weak-labeling workflow, not as a fully manually labeled dataset. Rows are marked as `gold_manual`, `model_assisted`, or `spot_checked` depending on how the final usable label was obtained.",
        "",
        "The labels in the example file sometimes contain multiple category IDs, such as `4, 2`. For the current deliverable, these were simplified into a single-label task by taking the first listed label as the primary risk category. This keeps evaluation explainable with standard accuracy, macro F1, precision, recall, and confusion-matrix metrics, while preserving the raw label column for future multi-label extension.",
        "The train/test split is ticker-held-out rather than random by row. This avoids leakage where the same company's wording, sector-specific vocabulary, or repeated disclosure style appears in both train and test.",
        "",
        "Current placeholder taxonomy:",
        mapping_lines,
        "",
        "## Model Descriptions",
        "Two models were evaluated. The baseline model uses TF-IDF unigram/bigram features with a class-balanced Logistic Regression classifier. The advanced model uses transformer-style sentence embeddings, preferring a financial-domain FinBERT source when available, then a MiniLM sentence-transformer fallback, and finally a local TF-IDF + SVD embedding fallback if downloads fail. The embedding vectors are classified with the same class-balanced Logistic Regression setup.",
        "",
        "## Evaluation Metrics",
        "The primary metric is macro F1 because the label distribution is not perfectly balanced and each risk category should contribute equally to the evaluation. Accuracy, weighted F1, macro precision, and macro recall are also reported. The held-out split is grouped by ticker rather than randomly sampled rows.",
        "",
        "## Main Findings",
        f"The best model by held-out macro F1 was **{best_model_name}**. The model comparison table below can be copied into the final report:",
        "",
        comparison_df.to_markdown(index=False, floatfmt=".3f"),
        "",
        "The confusion matrix and per-class F1 chart show where category boundaries remain difficult. The label distribution figure should be used to explain class imbalance and why macro F1 is more informative than accuracy alone.",
        "",
        "## Error Analysis",
        "The error analysis file identifies the most confused label pairs and the lowest-performing classes. These errors are likely caused by overlapping risk language across categories, short sentences without enough context, and the simplification from multi-label annotations to a single primary label.",
        "",
        "Year-sliced evaluation was " + ("generated from the full labeled corpus metadata." if has_year else "skipped because year metadata was unavailable."),
        "Sector-sliced evaluation was " + ("generated from the full labeled corpus metadata." if has_sector else "skipped because sector metadata was unavailable."),
        "",
        "## Limitations and Future Work",
        "The biggest limitation is that most of the 2,841-row corpus uses model-assisted weak labels rather than true human-reviewed labels. The placeholder taxonomy should be replaced with the official project taxonomy before final submission. Future work should complete human review of the weak labels, evaluate true multi-label classification, and test stronger domain-specific fine-tuning once more gold labels are available.",
    ]
    output_path.write_text("\n".join(lines) + "\n")
    status(f"Saved report draft to {output_path}")


def write_ppt_outline(output_path: Path, best_model_name: str, comparison_df: pd.DataFrame) -> None:
    lines = [
        "# Problem 1 PPT Outline",
        "",
        "## Slide 1: Task + Dataset",
        "- Title: Risk Factor Classification from SEC 10-K Sentences",
        "- Show: label distribution chart `problem1_label_distribution.png`",
        "- Say: the main corpus has 2,841 rows; about 300 gold manual examples are used to weak-label the rest.",
        "- Say: multi-label strings were simplified by taking the first label as the primary risk category for clean single-label evaluation.",
        "- Say: train/test is split by ticker to reduce leakage from company- and sector-specific wording.",
        "",
        "## Slide 2: Model Comparison + Metrics",
        "- Show: model comparison table from `problem1_model_comparison.csv`",
        f"- Main takeaway: best held-out macro F1 model was `{best_model_name}`.",
        "- Explain: baseline is TF-IDF + Logistic Regression; advanced model is financial/domain embedding + Logistic Regression with fallbacks.",
        "- Add: the 2,841-row corpus has nonblank final usable labels, but most are model-assisted weak labels rather than pure human labels.",
        "",
        comparison_df.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Slide 3: Error Analysis + Takeaways",
        "- Show: `problem1_confusion_matrix.png` and/or `problem1_per_class_f1.png`",
        "- Say: hardest labels are the classes with lowest F1; confused pairs often reflect overlapping risk language.",
        "- Close: human review of weak labels and multi-label evaluation are the natural next steps.",
    ]
    output_path.write_text("\n".join(lines) + "\n")
    status(f"Saved PPT outline to {output_path}")


def main() -> int:
    pd.set_option("display.max_columns", 50)
    pd.set_option("display.width", 160)

    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir
    outputs_dir = base_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    main_csv = base_dir / "labels" / "annotation_dataset.csv"
    example_csv = base_dir / "example_labels" / "annotated_dataset_labeled.csv"

    if not main_csv.exists():
        raise FileNotFoundError(f"Missing main annotation CSV: {main_csv}")
    if not example_csv.exists():
        raise FileNotFoundError(f"Missing manually labeled example CSV: {example_csv}")

    main_df, main_cols = inspect_csv(main_csv, "main annotation dataset")
    example_df, example_cols = inspect_csv(example_csv, "manual example labels")

    clean_df, label_mapping = build_labeled_dataset(main_df, main_cols, example_df, example_cols, outputs_dir)
    full_labeled_df, spot_check_df = create_model_assisted_full_corpus_labels(
        clean_df, main_df, main_cols, label_mapping, outputs_dir
    )
    save_label_distribution(full_labeled_df, label_mapping, outputs_dir / "problem1_label_distribution.png")

    train_df, test_df = split_data(full_labeled_df)
    split_summary = pd.DataFrame(
        [
            {
                "split": "train",
                "n_rows": len(train_df),
                "tickers": ", ".join(sorted(train_df["ticker"].astype(str).unique())),
                "sectors": ", ".join(sorted(train_df["sector"].astype(str).unique())) if "sector" in train_df else "",
                "label_distribution": train_df["primary_label"].value_counts().sort_index().to_dict(),
            },
            {
                "split": "test",
                "n_rows": len(test_df),
                "tickers": ", ".join(sorted(test_df["ticker"].astype(str).unique())),
                "sectors": ", ".join(sorted(test_df["sector"].astype(str).unique())) if "sector" in test_df else "",
                "label_distribution": test_df["primary_label"].value_counts().sort_index().to_dict(),
            },
        ]
    )
    split_summary_path = outputs_dir / "problem1_ticker_split_summary.csv"
    split_summary.to_csv(split_summary_path, index=False)
    status(f"Saved ticker split summary to {split_summary_path}")

    baseline_model, baseline_pred, baseline_metrics = train_baseline(train_df, test_df)
    print("\n=== Model A Classification Report ===")
    print(classification_report(test_df["primary_label"], baseline_pred, zero_division=0))

    advanced_source, advanced_pred, advanced_metrics = train_advanced(train_df, test_df, base_dir)
    print("\n=== Model B Classification Report ===")
    print(classification_report(test_df["primary_label"], advanced_pred, zero_division=0))

    comparison_df = pd.DataFrame([baseline_metrics, advanced_metrics])
    llm_pred_df, llm_summary_df, llm_summary = run_prompt_llm_classifier(train_df, test_df, outputs_dir)
    if llm_summary.get("status") == "ran":
        comparison_df = pd.concat(
            [
                comparison_df,
                pd.DataFrame(
                    [
                        {
                            "model": llm_summary["model"],
                            "accuracy": llm_summary.get("accuracy", np.nan),
                            "macro_f1": llm_summary.get("macro_f1", np.nan),
                            "weighted_f1": llm_summary.get("weighted_f1", np.nan),
                            "macro_precision": llm_summary.get("macro_precision", np.nan),
                            "macro_recall": llm_summary.get("macro_recall", np.nan),
                            "n_eval": llm_summary.get("n_eval", 0),
                            "avg_latency_s": llm_summary.get("avg_latency_s", np.nan),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    else:
        comparison_df = pd.concat(
            [
                comparison_df,
                pd.DataFrame(
                    [
                        {
                            "model": "Prompt-based LLM (skipped)",
                            "accuracy": np.nan,
                            "macro_f1": np.nan,
                            "weighted_f1": np.nan,
                            "macro_precision": np.nan,
                            "macro_recall": np.nan,
                            "n_eval": 0,
                            "avg_latency_s": np.nan,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    comparison_path = outputs_dir / "problem1_model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    status(f"Saved model comparison table to {comparison_path}")

    best_idx = comparison_df["macro_f1"].astype(float).idxmax()
    best_model_name = str(comparison_df.loc[best_idx, "model"])
    best_pred = baseline_pred if best_idx == 0 else advanced_pred

    pred_df = test_df.copy()
    pred_df["baseline_pred"] = baseline_pred
    pred_df["advanced_pred"] = advanced_pred
    pred_df["best_model_pred"] = best_pred
    pred_df["best_model_name"] = best_model_name
    predictions_path = outputs_dir / "problem1_predictions.csv"
    pred_df.to_csv(predictions_path, index=False)
    status(f"Saved prediction file to {predictions_path}")

    baseline_report = classification_report_df(test_df["primary_label"], baseline_pred, "TF-IDF + Logistic Regression")
    advanced_report = classification_report_df(test_df["primary_label"], advanced_pred, f"{advanced_source} + Logistic Regression")
    per_class_report = pd.concat([baseline_report, advanced_report], ignore_index=True)
    per_class_path = outputs_dir / "problem1_per_class_report.csv"
    per_class_report.to_csv(per_class_path, index=False)
    status(f"Saved per-class report to {per_class_path}")

    best_report_df = advanced_report if best_idx == 1 else baseline_report
    save_per_class_f1(best_report_df, outputs_dir / "problem1_per_class_f1.png")

    labels = sorted(full_labeled_df["primary_label"].astype(str).unique(), key=lambda x: int(x) if x.isdigit() else x)
    matrix_df = save_confusion_matrix(
        test_df["primary_label"].astype(str).tolist(),
        pd.Series(best_pred).astype(str).tolist(),
        labels,
        outputs_dir / "problem1_confusion_matrix.png",
    )
    matrix_df.to_csv(outputs_dir / "problem1_confusion_matrix.csv")

    year_results = sliced_results(
        pred_df,
        "best_model_pred",
        "year",
        outputs_dir / "problem1_year_sliced_results.csv",
        outputs_dir / "problem1_year_sliced_f1.png",
    )
    if year_results is None:
        status("Year-sliced evaluation skipped because year metadata is unavailable in the test set.")

    sector_results = sliced_results(
        pred_df,
        "best_model_pred",
        "sector",
        outputs_dir / "problem1_sector_sliced_results.csv",
        outputs_dir / "problem1_sector_sliced_f1.png",
    )
    if sector_results is None:
        status("Sector-sliced evaluation skipped because sector metadata is unavailable in the test set.")

    confused_df = most_confused_pairs(matrix_df)
    confused_df.to_csv(outputs_dir / "problem1_most_confused_pairs.csv", index=False)

    write_error_analysis(
        outputs_dir / "problem1_error_analysis.md",
        best_model_name,
        comparison_df,
        best_report_df,
        confused_df,
        year_results,
        sector_results,
    )
    write_report_draft(
        outputs_dir / "problem1_report_draft.md",
        best_model_name,
        comparison_df,
        label_mapping,
        year_results is not None,
        sector_results is not None,
    )
    write_ppt_outline(outputs_dir / "problem1_ppt_outline.md", best_model_name, comparison_df)

    blank_labels = int(full_labeled_df["human_final_label"].isna().sum() + full_labeled_df["human_final_label"].astype(str).str.strip().eq("").sum())
    source_counts = full_labeled_df["label_source"].value_counts().to_dict()
    final_summary = {
        "total_rows": len(full_labeled_df),
        "gold_manual_rows_in_2841": int(source_counts.get("gold_manual", 0)),
        "model_assisted_rows": int(source_counts.get("model_assisted", 0)),
        "spot_checked_rows": int(source_counts.get("spot_checked", 0)),
        "original_gold_training_examples": len(clean_df),
        "blank_final_labels": blank_labels,
        "train_tickers": ", ".join(sorted(train_df["ticker"].astype(str).unique())),
        "test_tickers": ", ".join(sorted(test_df["ticker"].astype(str).unique())),
        "models_compared": ", ".join(comparison_df["model"].astype(str).tolist()),
        "prompt_llm_status": llm_summary.get("status", "unknown"),
        "prompt_llm_note": llm_summary.get("reason", ""),
        "best_model": best_model_name,
        "best_accuracy": float(comparison_df.loc[best_idx, "accuracy"]),
        "best_macro_f1": float(comparison_df.loc[best_idx, "macro_f1"]),
    }
    pd.DataFrame([final_summary]).to_csv(outputs_dir / "problem1_final_summary.csv", index=False)

    print("\n=== Problem 1 Final Summary ===")
    for key, value in final_summary.items():
        print(f"{key}: {value}")

    status("Done. Generated Problem 1 outputs:")
    for path in sorted(outputs_dir.glob("problem1_*")):
        print(f"  - {path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        print("\n[Problem 1] Pipeline failed with an unexpected error:", file=sys.stderr)
        traceback.print_exc()
        raise
