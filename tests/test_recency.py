from __future__ import annotations

from modules.recency_ranker import RecencyRanker


def test_recency_score_is_normalized():
    scorer = RecencyRanker(decay_lambda=0.05)
    score_recent = scorer.score_date("2025-01-01")
    score_old = scorer.score_date("2010-01-01")
    assert 0.0 <= score_recent <= 1.0
    assert 0.0 <= score_old <= 1.0
    assert score_recent > score_old
