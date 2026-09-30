from __future__ import annotations

from typing import Any, Dict, List

from modules.llm_generator import LLMGenerator
from modules.retriever import VectorStoreIndex


class BaselineRAG:
    def __init__(self, vectorstore: VectorStoreIndex, llm: LLMGenerator):
        self.vectorstore = vectorstore
        self.llm = llm

    def run(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        retrieval = self.vectorstore.query(query, top_k=top_k)
        answer = self.llm.generate_answer(query, retrieval)
        return {
            "query": query,
            "retrieval": retrieval,
            "answer": answer,
            "pipeline": ["query", "retrieve", "llm", "answer"],
        }
