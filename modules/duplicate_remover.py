from __future__ import annotations

from typing import Any, Dict, List, Tuple

from modules.embeddings import EmbeddingModel


class DuplicateRemover:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", threshold: float = 0.92):
        self.model = EmbeddingModel(model_name)
        self.threshold = threshold

    def detect_duplicates(self, results: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        kept = []
        removed = []
        duplicate_records = []
        seen = []

        for item in results:
            text = item.get("text", "")
            embedding = self.model.encode([text])[0]
            is_duplicate = False
            for prior in seen:
                prior_embedding = prior["embedding"]
                similarity = self.model.similarity(embedding, prior_embedding)
                if similarity >= self.threshold:
                    is_duplicate = True
                    duplicate_records.append({
                        "duplicate_of": prior["chunk_id"],
                        "duplicate_candidate": item.get("chunk_id"),
                        "similarity": similarity,
                        "document_id": item.get("document_id"),
                        "title": item.get("title"),
                    })
                    removed.append(item)
                    break
            if not is_duplicate:
                kept.append(item)
                seen.append({"chunk_id": item.get("chunk_id"), "embedding": embedding})

        return kept, removed, duplicate_records

    def filter_duplicates(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        kept, removed, duplicates = self.detect_duplicates(results)
        return {
            "kept": kept,
            "removed": removed,
            "duplicate_records": duplicates,
            "count_removed": len(removed),
        }
