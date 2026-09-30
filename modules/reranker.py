from __future__ import annotations

from typing import Any, Dict, List


class ConflictAwareReranker:
    def __init__(self, weights: Dict[str, float] | None = None):
        self.weights = weights or {"relevance": 0.45, "credibility": 0.25, "recency": 0.2, "contradiction": 0.1}

    def compute_penalty(self, contradiction_pairs: List[Dict[str, Any]], item_id: str) -> float:
        penalty = 0.0
        for pair in contradiction_pairs:
            if pair.get("evidence_a") == item_id or pair.get("evidence_b") == item_id:
                penalty += pair.get("confidence", 0.0) * self.weights["contradiction"]
        return min(1.0, penalty)

    def rerank(self, results: List[Dict[str, Any]], contradictions: List[Dict[str, Any]] | None = None) -> List[Dict[str, Any]]:
        contradictions = contradictions or []
        reranked = []
        for idx, item in enumerate(results, start=1):
            item = dict(item)
            relevance = float(item.get("score", 0.0))
            credibility = float(item.get("credibility_score", 0.0))
            recency = float(item.get("recency_score", 0.0))
            penalty = self.compute_penalty(contradictions, item.get("chunk_id", ""))
            final_score = (
                self.weights["relevance"] * relevance
                + self.weights["credibility"] * credibility
                + self.weights["recency"] * recency
                - self.weights["contradiction"] * penalty
            )
            item["original_rank"] = idx
            item["relevance_score"] = round(relevance, 4)
            item["credibility_score"] = round(credibility, 4)
            item["recency_score"] = round(recency, 4)
            item["contradiction_penalty"] = round(penalty, 4)
            item["final_score"] = round(final_score, 4)
            reranked.append(item)
        reranked.sort(key=lambda x: x["final_score"], reverse=True)
        for rank, item in enumerate(reranked, start=1):
            item["final_rank"] = rank
        return reranked
