from __future__ import annotations

from modules.confidence_estimator import ConfidenceEstimator


def test_confidence_estimator_returns_normalized_score():
    estimator = ConfidenceEstimator()
    result = estimator.estimate([
        {"relevance_score": 0.9, "credibility_score": 0.8, "recency_score": 0.7},
        {"relevance_score": 0.7, "credibility_score": 0.6, "recency_score": 0.6},
    ], [{"document_id": "d1"}, {"document_id": "d2"}], [])
    assert 0.0 <= result["confidence_score"] <= 1.0
    assert "breakdown" in result
