from __future__ import annotations

import os
import re
from dataclasses import asdict
from typing import Dict, List, Tuple

import networkx as nx
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from document_loader import DocumentChunk

load_dotenv()


class GraphRAGEngine:
    """
    LightRAG-compatible fallback pipeline.

    Instead of depending on a specific LightRAG server setup, this class follows the same idea:
    1. retrieve relevant text chunks,
    2. retrieve nearby graph entities and relationships,
    3. generate a grounded answer with evidence.

    If OPENAI_API_KEY is available and USE_OPENAI=1, it can use an API LLM for final wording.
    Otherwise it returns an extractive answer, so the app still runs locally for class submission.
    """

    def __init__(self, chunks: List[DocumentChunk], graph: nx.Graph):
        self.chunks = chunks
        self.graph = graph
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=6000)
        self.chunk_texts = [c.text for c in chunks]
        self.matrix = self.vectorizer.fit_transform(self.chunk_texts) if self.chunk_texts else None

    def retrieve_chunks(self, question: str, top_k: int = 4) -> List[Tuple[DocumentChunk, float]]:
        if self.matrix is None or not self.chunk_texts:
            return []
        q_vec = self.vectorizer.transform([question])
        scores = cosine_similarity(q_vec, self.matrix).flatten()
        ranked = scores.argsort()[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in ranked if scores[i] > 0]

    def retrieve_graph_context(self, question: str, max_entities: int = 8, max_edges: int = 12) -> Dict[str, List[Dict]]:
        q_lower = question.lower()
        matched = []
        for node, data in self.graph.nodes(data=True):
            node_lower = str(node).lower()
            if node_lower in q_lower or any(tok in node_lower for tok in re.findall(r"[a-zA-Z]{4,}", q_lower)):
                matched.append((node, data))

        if not matched:
            degree_ranked = sorted(self.graph.degree, key=lambda x: x[1], reverse=True)[:max_entities]
            matched = [(n, self.graph.nodes[n]) for n, _ in degree_ranked]

        entities = []
        for node, data in matched[:max_entities]:
            entities.append({
                "name": node,
                "type": data.get("entity_type", "UNKNOWN"),
                "mention_count": data.get("mention_count", 1),
                "sources": data.get("sources", ""),
            })

        entity_names = {e[0] for e in matched[:max_entities]}
        edges = []
        for u, v, data in self.graph.edges(data=True):
            if u in entity_names or v in entity_names:
                edges.append({
                    "subject": u,
                    "relation": data.get("relation", "related_to"),
                    "object": v,
                    "weight": data.get("weight", 1),
                    "evidence": data.get("evidence", ""),
                })
            if len(edges) >= max_edges:
                break

        return {"entities": entities, "relationships": edges}

    def _build_prompt(self, question: str, chunks: List[Tuple[DocumentChunk, float]], graph_context: Dict[str, List[Dict]]) -> str:
        chunk_lines = []
        for chunk, score in chunks:
            chunk_lines.append(f"Source: {chunk.file_name}, chunk: {chunk.chunk_id}, score: {score:.3f}\n{chunk.text[:900]}")

        edge_lines = []
        for edge in graph_context["relationships"]:
            edge_lines.append(f"{edge['subject']} --{edge['relation']}--> {edge['object']} | evidence: {edge.get('evidence', '')[:250]}")

        return f"""Answer the question using only the provided evidence.
Question: {question}

Relevant source chunks:
{chr(10).join(chunk_lines)}

Relevant knowledge graph relationships:
{chr(10).join(edge_lines)}

Write a concise answer. Include sources and mention uncertainty if evidence is weak.
"""

    def _try_openai_answer(self, prompt: str) -> str | None:
        if os.getenv("USE_OPENAI", "0") != "1" or not os.getenv("OPENAI_API_KEY"):
            return None
        try:
            from openai import OpenAI
            client = OpenAI()
            response = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            return response.choices[0].message.content
        except Exception as exc:
            return f"OpenAI generation failed, so fallback answer is used. Error: {exc}"

    def answer(self, question: str, top_k: int = 4) -> Dict:
        retrieved_chunks = self.retrieve_chunks(question, top_k=top_k)
        graph_context = self.retrieve_graph_context(question)
        prompt = self._build_prompt(question, retrieved_chunks, graph_context)
        llm_answer = self._try_openai_answer(prompt)

        if llm_answer and not llm_answer.startswith("OpenAI generation failed"):
            answer_text = llm_answer
        else:
            answer_text = self._fallback_answer(question, retrieved_chunks, graph_context)
            if llm_answer:
                answer_text = f"{answer_text}\n\nNote: {llm_answer}"

        return {
            "answer": answer_text,
            "entities": graph_context["entities"],
            "relationships": graph_context["relationships"],
            "source_chunks": [
                {"chunk": asdict(chunk), "score": score}
                for chunk, score in retrieved_chunks
            ],
        }

    def _fallback_answer(self, question: str, chunks: List[Tuple[DocumentChunk, float]], graph_context: Dict[str, List[Dict]]) -> str:
        if not chunks and not graph_context["relationships"]:
            return "I could not find enough evidence in the uploaded documents to answer this question."

        best_chunk = chunks[0][0].text if chunks else ""
        best_sentences = re.split(r"(?<=[.!?])\s+", best_chunk)
        selected = " ".join(best_sentences[:3]).strip()

        relationship_text = ""
        if graph_context["relationships"]:
            edge = graph_context["relationships"][0]
            relationship_text = f" The knowledge graph also connects {edge['subject']} to {edge['object']} through the relation '{edge['relation']}'."

        source_text = ""
        if chunks:
            source_text = " Sources: " + "; ".join([f"{c.file_name} / {c.chunk_id}" for c, _ in chunks[:3]]) + "."

        return f"Based on the retrieved evidence, {selected}{relationship_text}{source_text}"
