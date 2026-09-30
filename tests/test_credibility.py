from __future__ import annotations

from modules.credibility_scorer import CredibilityScorer


def test_credibility_scores_in_range():
    scorer = CredibilityScorer()
    result = scorer.score_document({"source_type": "official", "domain": "research", "original_url": "https://example.org", "publication_date": "2024-01-01"})
    assert 0.0 <= result["credibility_score"] <= 1.0
    assert result["credibility_score"] > 0.5
