from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import pandas as pd

from document_loader import DocumentChunk


@dataclass
class EntityMention:
    name: str
    entity_type: str
    chunk_id: str
    file_name: str


@dataclass
class Triple:
    subject: str
    relation: str
    object: str
    chunk_id: str
    file_name: str
    evidence: str


ORG_HINTS = ["University", "Institute", "Company", "Corporation", "Inc", "LLC", "Agency", "Department", "Laboratory", "Labs", "Bank"]
TECH_HINTS = ["AI", "LLM", "RAG", "GraphRAG", "LightRAG", "NLP", "Transformer", "BERT", "GPT", "model", "algorithm", "dataset", "knowledge graph"]
RELATION_KEYWORDS = {
    "uses": ["uses", "used", "using", "applies", "applied", "leverages", "built with", "based on"],
    "improves": ["improves", "improved", "increases", "enhances", "boosts", "outperforms"],
    "reduces": ["reduces", "reduced", "decreases", "lowers", "mitigates"],
    "contains": ["contains", "includes", "consists of", "has", "provides"],
    "compares_with": ["compares", "compared with", "versus", "vs", "against"],
    "causes_or_influences": ["causes", "leads to", "affects", "influences", "drives"],
}


def _try_load_spacy():
    try:
        import spacy
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            return None
    except Exception:
        return None


NLP = _try_load_spacy()


def normalize_entity(name: str) -> str:
    name = re.sub(r"\s+", " ", name).strip(" .,:;()[]{}\"'")
    return name


def guess_entity_type(name: str) -> str:
    upper = name.upper()
    if upper in {"AI", "LLM", "RAG", "NLP", "API", "PDF", "CSV"}:
        return "TECH"
    if any(h.lower() in name.lower() for h in ORG_HINTS):
        return "ORG"
    if any(h.lower() in name.lower() for h in TECH_HINTS):
        return "TECH"
    if re.search(r"\b(19|20)\d{2}\b", name):
        return "DATE"
    return "CONCEPT"


def extract_entities_from_text(text: str, chunk: DocumentChunk) -> List[EntityMention]:
    mentions: List[EntityMention] = []

    if NLP is not None:
        doc = NLP(text[:8000])
        for ent in doc.ents:
            cleaned = normalize_entity(ent.text)
            if len(cleaned) >= 2:
                mentions.append(EntityMention(cleaned, ent.label_, chunk.chunk_id, chunk.file_name))

    acronym_matches = re.findall(r"\b[A-Z][A-Z0-9]{1,8}\b", text)
    phrase_matches = re.findall(r"\b(?:[A-Z][a-zA-Z0-9\-]+\s+){0,4}[A-Z][a-zA-Z0-9\-]+\b", text)
    keyword_matches = []
    for hint in TECH_HINTS:
        if re.search(rf"\b{re.escape(hint)}\b", text, flags=re.IGNORECASE):
            keyword_matches.append(hint)

    for raw in acronym_matches + phrase_matches + keyword_matches:
        cleaned = normalize_entity(raw)
        if len(cleaned) < 2 or cleaned.lower() in {"the", "this", "that", "figure", "table", "section"}:
            continue
        mentions.append(EntityMention(cleaned, guess_entity_type(cleaned), chunk.chunk_id, chunk.file_name))

    dedup = {}
    for m in mentions:
        key = (m.name.lower(), m.chunk_id)
        if key not in dedup:
            dedup[key] = m
    return list(dedup.values())


def infer_relation(sentence: str) -> str:
    lower = sentence.lower()
    for relation, keywords in RELATION_KEYWORDS.items():
        if any(k in lower for k in keywords):
            return relation
    return "co_occurs_with"


def extract_triples_from_chunk(chunk: DocumentChunk, entities: List[EntityMention], max_pairs_per_chunk: int = 18) -> List[Triple]:
    triples: List[Triple] = []
    if len(entities) < 2:
        return triples

    sentences = re.split(r"(?<=[.!?])\s+", chunk.text)
    entity_names = list({e.name for e in entities})

    for sent in sentences:
        sent_clean = sent.strip()
        if len(sent_clean) < 20:
            continue
        present = [name for name in entity_names if re.search(rf"\b{re.escape(name)}\b", sent_clean, flags=re.IGNORECASE)]
        if len(present) < 2:
            continue
        relation = infer_relation(sent_clean)
        for i in range(min(len(present), 5)):
            for j in range(i + 1, min(len(present), 5)):
                if present[i].lower() == present[j].lower():
                    continue
                triples.append(Triple(
                    subject=present[i],
                    relation=relation,
                    object=present[j],
                    chunk_id=chunk.chunk_id,
                    file_name=chunk.file_name,
                    evidence=sent_clean[:350],
                ))
                if len(triples) >= max_pairs_per_chunk:
                    return triples
    return triples


def build_graph(chunks: List[DocumentChunk]) -> Tuple[nx.Graph, List[EntityMention], List[Triple]]:
    all_entities: List[EntityMention] = []
    all_triples: List[Triple] = []
    graph = nx.Graph()

    for chunk in chunks:
        entities = extract_entities_from_text(chunk.text, chunk)
        all_entities.extend(entities)
        triples = extract_triples_from_chunk(chunk, entities)
        all_triples.extend(triples)

    entity_type_counter: Dict[str, Counter] = defaultdict(Counter)
    entity_sources: Dict[str, set] = defaultdict(set)
    for ent in all_entities:
        key = ent.name
        entity_type_counter[key][ent.entity_type] += 1
        entity_sources[key].add(ent.file_name)

    for name, type_counts in entity_type_counter.items():
        graph.add_node(
            name,
            label=name,
            entity_type=type_counts.most_common(1)[0][0],
            mention_count=sum(type_counts.values()),
            sources=", ".join(sorted(entity_sources[name])),
        )

    edge_counter = Counter()
    edge_evidence: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)
    edge_chunks: Dict[Tuple[str, str, str], set] = defaultdict(set)

    for tri in all_triples:
        if tri.subject not in graph or tri.object not in graph:
            continue
        a, b = sorted([tri.subject, tri.object])
        key = (a, b, tri.relation)
        edge_counter[key] += 1
        edge_evidence[key].append(tri.evidence)
        edge_chunks[key].add(tri.chunk_id)

    for (a, b, relation), weight in edge_counter.items():
        graph.add_edge(
            a,
            b,
            relation=relation,
            weight=int(weight),
            evidence=" | ".join(edge_evidence[(a, b, relation)][:3]),
            chunks=", ".join(sorted(edge_chunks[(a, b, relation)])),
        )

    return graph, all_entities, all_triples


def graph_stats(graph: nx.Graph, chunks: List[DocumentChunk], triples: List[Triple]) -> Dict[str, int]:
    return {
        "documents": len({c.file_name for c in chunks}),
        "chunks": len(chunks),
        "entities": graph.number_of_nodes(),
        "relationships": graph.number_of_edges(),
        "connected_components": nx.number_connected_components(graph) if graph.number_of_nodes() else 0,
        "extracted_triples": len(triples),
    }


def save_graph_outputs(graph: nx.Graph, entities: List[EntityMention], triples: List[Triple], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    graph_json = nx.node_link_data(graph)
    (output_dir / "knowledge_graph.json").write_text(json.dumps(graph_json, indent=2, ensure_ascii=False), encoding="utf-8")
    nx.write_graphml(graph, output_dir / "knowledge_graph.graphml")

    pd.DataFrame([asdict(e) for e in entities]).to_csv(output_dir / "entities.csv", index=False)
    pd.DataFrame([asdict(t) for t in triples]).to_csv(output_dir / "triples.csv", index=False)


def load_graph(graph_path: Path) -> nx.Graph:
    if not graph_path.exists():
        return nx.Graph()
    data = json.loads(graph_path.read_text(encoding="utf-8"))
    return nx.node_link_graph(data)
