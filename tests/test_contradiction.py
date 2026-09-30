from __future__ import annotations

from modules.contradiction_detector import ContradictionDetector


def test_contradiction_detector_runs_and_returns_labels():
    detector = ContradictionDetector(model_name="cross-encoder/nli-deberta-v3-base", threshold=0.75)
    result = detector.detect_pair(
        "Dense retrieval consistently outperforms sparse retrieval across all tasks.",
        "Dense retrieval does not consistently outperform sparse retrieval; performance depends on the dataset and evidence quality."
    )
    assert "label" in result
    assert "confidence" in result
    assert result["label"] in {"contradiction", "entailment", "neutral"}
