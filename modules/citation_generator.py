from __future__ import annotations

from typing import Any, Dict, List


class CitationGenerator:
    def __init__(self):
        pass

    def generate(self, evidence: List[Dict[str, Any]], answer: str) -> List[Dict[str, Any]]:
        citations = []
        for index, item in enumerate(evidence, start=1):
            citations.append({
                "citation_id": f"[{index}]",
                "document_title": item.get("title") or item.get("metadata", {}).get("title") or "Unknown document",
                "source": item.get("source") or item.get("metadata", {}).get("source") or "Unknown source",
                "chunk_id": item.get("chunk_id"),
                "relevant_excerpt": item.get("text", "")[:250],
                "publication_date": item.get("publication_date") or item.get("metadata", {}).get("publication_date"),
                "retrieval_score": item.get("score"),
                "final_score": item.get("final_score"),
            })
        return citations
