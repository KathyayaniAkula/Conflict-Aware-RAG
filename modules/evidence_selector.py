from __future__ import annotations

from typing import Any, Dict, List


class EvidenceSelector:
    def __init__(self, evidence_count: int = 4):
        self.evidence_count = evidence_count

    def select(self, reranked_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        selected = []
        seen_docs = set()
        for item in reranked_results:
            doc_id = item.get("document_id")
            if doc_id in seen_docs:
                continue
            selected.append(item)
            seen_docs.add(doc_id)
            if len(selected) >= self.evidence_count:
                break
        if len(selected) < self.evidence_count:
            for item in reranked_results:
                if item not in selected:
                    selected.append(item)
                if len(selected) >= self.evidence_count:
                    break
        return selected
