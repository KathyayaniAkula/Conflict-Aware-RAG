from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from config.config import SETTINGS, RESULTS_DIR
from modules.baseline_rag import BaselineRAG
from modules.llm_generator import LLMGenerator
from modules.proposed_rag import ProposedRAG
from modules.retriever import VectorStoreIndex


def load_or_init_vectorstore() -> VectorStoreIndex:
    store = VectorStoreIndex(collection_name="research_chunks", persist_directory=Path("vectorstore"))
    if store.count() == 0:
        from scripts.ingest_data import build_vectorstore
        store = build_vectorstore()
    return store


st.set_page_config(page_title=SETTINGS.project_title, layout="wide")
st.title(SETTINGS.project_title)

with st.sidebar:
    st.header("System config")
    query = st.text_input("Enter query", "How can retrieval systems improve transparency and reduce duplicate evidence?")
    if st.button("Run pipeline"):
        vectorstore = load_or_init_vectorstore()
        llm = LLMGenerator(model_name=SETTINGS.ollama_model)
        baseline = BaselineRAG(vectorstore, llm)
        proposed = ProposedRAG(vectorstore, llm)

        with st.spinner("Running baseline and conflict-aware pipeline..."):
            baseline_result = baseline.run(query, top_k=SETTINGS.top_k)
            proposed_result = proposed.run(query, top_k=SETTINGS.top_k)

        st.session_state["baseline_result"] = baseline_result
        st.session_state["proposed_result"] = proposed_result

if "baseline_result" in st.session_state:
    baseline_result = st.session_state["baseline_result"]
    proposed_result = st.session_state["proposed_result"]

    tab1, tab2 = st.tabs(["Baseline RAG", "Proposed Conflict-Aware RAG"])

    with tab1:
        st.subheader("Final answer")
        st.write(baseline_result["answer"])
        st.subheader("Retrieved documents")
        for item in baseline_result["retrieval"]:
            st.markdown(f"- {item.get('title', 'Unknown')} :: {item.get('source', 'Unknown')} :: {item.get('score', 0)}")

    with tab2:
        st.subheader("Final answer")
        st.write(proposed_result["answer"])

        st.subheader("Confidence Score")
        st.write(proposed_result["confidence"]["confidence_score"])
        st.json(proposed_result["confidence"]["breakdown"])

        st.subheader("Selected evidence")
        for item in proposed_result["selected_evidence"]:
            st.markdown(f"- {item.get('title', 'Unknown')} | {item.get('source', 'Unknown')} | Final score: {item.get('final_score')}")

        st.subheader("Duplicate detection")
        st.json(proposed_result["duplicate_removal"])

        st.subheader("Credibility and recency")
        st.json({"credibility": proposed_result["credibility_scores"], "recency": proposed_result["recency_scores"]})

        st.subheader("Contradiction pairs")
        st.json(proposed_result["contradictions"])

        st.subheader("Reranking details")
        st.json(proposed_result["reranked"])

        st.subheader("Citations")
        st.json(proposed_result["citations"])
