from __future__ import annotations

from modules.reranker import ConflictAwareReranker


def test_reranking_updates_scores():
    reranker = ConflictAwareReranker({"relevance": 0.5, "credibility": 0.2, "recency": 0.2, "contradiction": 0.1})
    results = [{
        "chunk_id": "a",
        "score": 0.9,
        "credibility_score": 0.8,
        "recency_score": 0.7,
    }, {
        "chunk_id": "b",
        "score": 0.6,
        "credibility_score": 0.6,
        "recency_score": 0.5,
    }]
    reranked = reranker.rerank(results, [])
    assert all("final_score" in item for item in reranked)
    assert reranked[0]["final_score"] >= reranked[1]["final_score"]
