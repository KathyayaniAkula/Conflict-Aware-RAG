from __future__ import annotations

from typing import Any, Dict, List


class ConfidenceEstimator:
    def __init__(self):
        pass

    def estimate(self, results: List[Dict[str, Any]], evidence: List[Dict[str, Any]], contradictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not results:
            return {"confidence_score": 0.0, "breakdown": {"retrieval_quality": 0.0, "evidence_coverage": 0.0, "credibility": 0.0, "recency": 0.0, "contradiction": 0.0, "diversity": 0.0}} 

        avg_relevance = sum(float(item.get("relevance_score", 0.0)) for item in results) / len(results)
        avg_credibility = sum(float(item.get("credibility_score", 0.0)) for item in results) / len(results)
        avg_recency = sum(float(item.get("recency_score", 0.0)) for item in results) / len(results)
        evidence_coverage = min(1.0, len(evidence) / max(len(results), 1))
        diversity = min(1.0, len({item.get("document_id") for item in results}) / max(len(results), 1))
        contradiction_penalty = min(1.0, len(contradictions) / max(len(results), 1))

        confidence = (
            0.35 * avg_relevance +
            0.25 * avg_credibility +
            0.15 * avg_recency +
            0.15 * evidence_coverage +
            0.10 * diversity -
            0.10 * contradiction_penalty
        )
        confidence = max(0.0, min(1.0, confidence))

        breakdown = {
            "retrieval_quality": round(avg_relevance, 4),
            "evidence_coverage": round(evidence_coverage, 4),
            "credibility": round(avg_credibility, 4),
            "recency": round(avg_recency, 4),
            "contradiction": round(1.0 - contradiction_penalty, 4),
            "diversity": round(diversity, 4),
        }

        return {
            "confidence_score": round(confidence, 4),
            "breakdown": breakdown,
        }
