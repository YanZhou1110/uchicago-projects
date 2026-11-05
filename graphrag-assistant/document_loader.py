from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
from pypdf import PdfReader


TEXT_COLUMNS = ["text", "content", "article", "body", "abstract", "description", "summary"]
TITLE_COLUMNS = ["title", "headline", "name"]


@dataclass
class DocumentChunk:
    chunk_id: str
    file_name: str
    source_type: str
    text: str
    title: str = ""
    row_id: Optional[int] = None


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_pdf_text(file_path: Path) -> str:
    reader = PdfReader(str(file_path))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return clean_text("\n".join(pages))


def extract_txt_text(file_path: Path) -> str:
    try:
        return clean_text(file_path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return clean_text(file_path.read_text(encoding="latin-1"))


def choose_csv_text_column(df: pd.DataFrame) -> str:
    lower_to_original = {c.lower().strip(): c for c in df.columns}
    for col in TEXT_COLUMNS:
        if col in lower_to_original:
            return lower_to_original[col]

    object_cols = [c for c in df.columns if df[c].dtype == "object"]
    if not object_cols:
        raise ValueError("No text-like column found in CSV. Please include a column such as text, content, article, body, or title.")

    lengths = {}
    for col in object_cols:
        lengths[col] = df[col].fillna("").astype(str).str.len().mean()
    return max(lengths, key=lengths.get)


def choose_csv_title_column(df: pd.DataFrame) -> Optional[str]:
    lower_to_original = {c.lower().strip(): c for c in df.columns}
    for col in TITLE_COLUMNS:
        if col in lower_to_original:
            return lower_to_original[col]
    return None


def split_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    text = clean_text(text)
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current = ""

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if len(current) + len(sent) + 1 <= chunk_size:
            current = f"{current} {sent}".strip()
        else:
            if current:
                chunks.append(current)
            if overlap > 0 and current:
                current = current[-overlap:] + " " + sent
            else:
                current = sent

    if current:
        chunks.append(current)

    return chunks


def load_file_to_chunks(file_path: Path, chunk_size: int = 900, overlap: int = 150, max_csv_rows: int = 100) -> List[DocumentChunk]:
    suffix = file_path.suffix.lower()
    chunks: List[DocumentChunk] = []

    if suffix == ".pdf":
        text = extract_pdf_text(file_path)
        for idx, chunk_text in enumerate(split_text(text, chunk_size, overlap)):
            chunks.append(DocumentChunk(
                chunk_id=f"{file_path.stem}_chunk_{idx}",
                file_name=file_path.name,
                source_type="pdf",
                text=chunk_text,
                title=file_path.stem,
            ))

    elif suffix in [".txt", ".md"]:
        text = extract_txt_text(file_path)
        for idx, chunk_text in enumerate(split_text(text, chunk_size, overlap)):
            chunks.append(DocumentChunk(
                chunk_id=f"{file_path.stem}_chunk_{idx}",
                file_name=file_path.name,
                source_type="text",
                text=chunk_text,
                title=file_path.stem,
            ))

    elif suffix == ".csv":
        df = pd.read_csv(file_path)
        text_col = choose_csv_text_column(df)
        title_col = choose_csv_title_column(df)
        df = df.head(max_csv_rows)

        for row_idx, row in df.iterrows():
            title = str(row.get(title_col, "")) if title_col else ""
            body = str(row.get(text_col, ""))
            combined = clean_text(f"{title}. {body}" if title else body)
            if not combined:
                continue
            for local_idx, chunk_text in enumerate(split_text(combined, chunk_size, overlap)):
                chunks.append(DocumentChunk(
                    chunk_id=f"{file_path.stem}_row_{row_idx}_chunk_{local_idx}",
                    file_name=file_path.name,
                    source_type="csv",
                    text=chunk_text,
                    title=title,
                    row_id=int(row_idx),
                ))
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    return chunks


def load_all_files(file_paths: Iterable[Path], chunk_size: int = 900, overlap: int = 150, max_csv_rows: int = 100) -> List[DocumentChunk]:
    all_chunks: List[DocumentChunk] = []
    for path in file_paths:
        all_chunks.extend(load_file_to_chunks(path, chunk_size, overlap, max_csv_rows))
    return all_chunks


def save_chunks(chunks: List[DocumentChunk], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps([asdict(c) for c in chunks], indent=2, ensure_ascii=False), encoding="utf-8")


def load_saved_chunks(path: Path) -> List[DocumentChunk]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [DocumentChunk(**item) for item in data]
