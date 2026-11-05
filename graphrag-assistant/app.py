from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

from document_loader import load_all_files, load_saved_chunks, save_chunks
from graph_builder import build_graph, graph_stats, load_graph, save_graph_outputs
from rag_engine import GraphRAGEngine


BASE_DIR = Path(__file__).parent
STORAGE_DIR = BASE_DIR / "storage"
UPLOAD_DIR = STORAGE_DIR / "uploads"
CHUNKS_PATH = STORAGE_DIR / "chunks.json"
GRAPH_PATH = STORAGE_DIR / "knowledge_graph.json"
TRIPLES_PATH = STORAGE_DIR / "triples.csv"
ENTITIES_PATH = STORAGE_DIR / "entities.csv"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="GraphRAG Streamlit App", layout="wide")


def load_existing_state():
    chunks = load_saved_chunks(CHUNKS_PATH)
    graph = load_graph(GRAPH_PATH)
    return chunks, graph


def save_uploaded_files(uploaded_files):
    saved_paths = []
    for uploaded in uploaded_files:
        out_path = UPLOAD_DIR / uploaded.name
        out_path.write_bytes(uploaded.getbuffer())
        saved_paths.append(out_path)
    return saved_paths


def make_pyvis_graph(graph: nx.Graph, max_nodes: int = 120) -> str:
    if graph.number_of_nodes() == 0:
        return ""

    degree_ranked = sorted(graph.degree, key=lambda x: x[1], reverse=True)
    selected_nodes = {node for node, _ in degree_ranked[:max_nodes]}
    subgraph = graph.subgraph(selected_nodes).copy()

    color_map = {
        "PERSON": "#f2c6c2",
        "ORG": "#b7c9b7",
        "GPE": "#b9c7dc",
        "DATE": "#ded0a6",
        "TECH": "#c9bddf",
        "CONCEPT": "#d8d8d8",
    }

    net = Network(height="620px", width="100%", bgcolor="#ffffff", font_color="#333333")
    net.force_atlas_2based(gravity=-45, central_gravity=0.01, spring_length=130, spring_strength=0.08)

    for node, data in subgraph.nodes(data=True):
        entity_type = data.get("entity_type", "CONCEPT")
        count = int(data.get("mention_count", 1))
        net.add_node(
            node,
            label=str(node)[:32],
            title=f"Entity: {node}<br>Type: {entity_type}<br>Mentions: {count}<br>Sources: {data.get('sources', '')}",
            color=color_map.get(entity_type, "#d8d8d8"),
            size=12 + min(count, 20),
        )

    for u, v, data in subgraph.edges(data=True):
        net.add_edge(
            u,
            v,
            label=data.get("relation", "related_to"),
            title=f"Relation: {data.get('relation', '')}<br>Evidence: {data.get('evidence', '')[:400]}",
            value=int(data.get("weight", 1)),
        )

    html_path = STORAGE_DIR / "graph_visualization.html"
    net.write_html(str(html_path), notebook=False)
    return html_path.read_text(encoding="utf-8")


if "messages" not in st.session_state:
    st.session_state.messages = []

if "chunks" not in st.session_state or "graph" not in st.session_state:
    st.session_state.chunks, st.session_state.graph = load_existing_state()

st.title("GraphRAG Knowledge Graph Chatbot")
st.caption("Upload papers or article datasets, build a knowledge graph, and query it with a LightRAG-style chat workflow.")

with st.sidebar:
    st.header("1. Upload data")
    uploaded_files = st.file_uploader(
        "Upload PDFs, TXT/MD files, or CSV article datasets",
        type=["pdf", "txt", "md", "csv"],
        accept_multiple_files=True,
    )

    chunk_size = st.slider("Chunk size", 400, 1800, 900, 100)
    overlap = st.slider("Chunk overlap", 0, 400, 150, 50)
    max_csv_rows = st.slider("Max CSV rows to process", 10, 500, 100, 10)

    if uploaded_files:
        st.write("Uploaded files:")
        for f in uploaded_files:
            st.write(f"- {f.name}")

    build_clicked = st.button("Build knowledge graph", type="primary")

    if st.button("Load saved graph"):
        st.session_state.chunks, st.session_state.graph = load_existing_state()
        st.success("Loaded saved chunks and graph from storage/.")

    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.success("Chat history cleared.")

if build_clicked:
    if not uploaded_files:
        st.warning("Please upload at least one file first.")
    else:
        with st.spinner("Processing documents and extracting graph relationships..."):
            paths = save_uploaded_files(uploaded_files)
            chunks = load_all_files(paths, chunk_size=chunk_size, overlap=overlap, max_csv_rows=max_csv_rows)
            graph, entities, triples = build_graph(chunks)
            save_chunks(chunks, CHUNKS_PATH)
            save_graph_outputs(graph, entities, triples, STORAGE_DIR)
            st.session_state.chunks = chunks
            st.session_state.graph = graph
        st.success("Knowledge graph built and saved in storage/.")

chunks = st.session_state.chunks
main_graph = st.session_state.graph

stats = graph_stats(main_graph, chunks, []) if chunks else {
    "documents": 0,
    "chunks": 0,
    "entities": 0,
    "relationships": 0,
    "connected_components": 0,
    "extracted_triples": 0,
}
if TRIPLES_PATH.exists():
    try:
        stats["extracted_triples"] = len(pd.read_csv(TRIPLES_PATH))
    except Exception:
        pass

metric_cols = st.columns(5)
metric_cols[0].metric("Documents", stats["documents"])
metric_cols[1].metric("Chunks", stats["chunks"])
metric_cols[2].metric("Entities", stats["entities"])
metric_cols[3].metric("Relationships", stats["relationships"])
metric_cols[4].metric("Components", stats["connected_components"])

kg_tab, table_tab, chat_tab, storage_tab = st.tabs(["Knowledge Graph", "Entities & Relationships", "GraphRAG Chat", "Local Storage"])

with kg_tab:
    st.subheader("Knowledge graph visualization")
    if main_graph.number_of_nodes() == 0:
        st.info("Build or load a graph first.")
    else:
        html = make_pyvis_graph(main_graph)
        components.html(html, height=650, scrolling=True)

with table_tab:
    st.subheader("Extracted entities and relationships")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Entities**")
        if ENTITIES_PATH.exists():
            st.dataframe(pd.read_csv(ENTITIES_PATH).head(300), use_container_width=True)
        else:
            st.info("No entities saved yet.")
    with col2:
        st.markdown("**Relationships / triples**")
        if TRIPLES_PATH.exists():
            st.dataframe(pd.read_csv(TRIPLES_PATH).head(300), use_container_width=True)
        else:
            st.info("No triples saved yet.")

with chat_tab:
    st.subheader("Ask questions about the uploaded documents")
    st.write("The app retrieves source chunks and graph relationships, then answers with evidence. OpenAI is optional; without an API key it uses a local extractive fallback.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a question about the uploaded papers/articles...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            if not chunks or main_graph.number_of_nodes() == 0:
                answer_text = "Please build or load a knowledge graph before asking questions."
                st.markdown(answer_text)
            else:
                engine = GraphRAGEngine(chunks, main_graph)
                result = engine.answer(question, top_k=4)
                answer_text = result["answer"]
                st.markdown(answer_text)

                with st.expander("Evidence used by GraphRAG"):
                    st.markdown("**Related entities**")
                    st.json(result["entities"][:8])
                    st.markdown("**Related relationships**")
                    st.json(result["relationships"][:8])
                    st.markdown("**Source chunks**")
                    for item in result["source_chunks"][:4]:
                        c = item["chunk"]
                        st.write(f"- {c['file_name']} / {c['chunk_id']} / score={item['score']:.3f}")
                        st.caption(c["text"][:600])

            st.session_state.messages.append({"role": "assistant", "content": answer_text})

with storage_tab:
    st.subheader("Files saved locally")
    storage_files = sorted([p for p in STORAGE_DIR.glob("**/*") if p.is_file()])
    if not storage_files:
        st.info("No storage files yet.")
    else:
        for p in storage_files:
            st.write(f"- {p.relative_to(BASE_DIR)}")

    st.markdown("**Saved graph JSON preview**")
    if GRAPH_PATH.exists():
        data = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
        st.json({"nodes": len(data.get("nodes", [])), "links": len(data.get("links", []))})
    else:
        st.info("No graph JSON saved yet.")
