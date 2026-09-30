from __future__ import annotations

from modules.baseline_rag import BaselineRAG
from modules.llm_generator import LLMGenerator
from modules.proposed_rag import ProposedRAG
from scripts.ingest_data import build_vectorstore


def test_end_to_end_pipeline_runs():
    vectorstore = build_vectorstore()
    llm = LLMGenerator(model_name="llama3.2")
    baseline = BaselineRAG(vectorstore, llm)
    proposed = ProposedRAG(vectorstore, llm)
    query = "How can retrieval systems improve transparency and reduce duplicate evidence?"
    baseline_result = baseline.run(query, top_k=5)
    proposed_result = proposed.run(query, top_k=8)
    assert isinstance(baseline_result["answer"], str)
    assert isinstance(proposed_result["answer"], str)
    assert len(proposed_result["selected_evidence"]) > 0
