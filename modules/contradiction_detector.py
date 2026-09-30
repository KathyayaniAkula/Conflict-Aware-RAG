from __future__ import annotations

import re
from typing import Any, Dict, List

try:
    from sentence_transformers import CrossEncoder
except ImportError:  # pragma: no cover
    CrossEncoder = None


class ContradictionDetector:
    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base", threshold: float = 0.75):
        self.model_name = model_name
        self.threshold = threshold
        self.model = None
        if CrossEncoder is not None:
            try:
                self.model = CrossEncoder(model_name)
            except Exception:
                self.model = None

    def _heuristic_label(self, text_a: str, text_b: str) -> Dict[str, Any]:
        a = text_a.lower()
        b = text_b.lower()
        overlap = len(set(re.findall(r"\w+", a)) & set(re.findall(r"\w+", b)))
        negations = ["not", "never", "no", "cannot", "does not", "doesn't", "never"]
        a_has_negation = any(term in a for term in negations)
        b_has_negation = any(term in b for term in negations)

        if a_has_negation != b_has_negation and overlap > 0:
            label = "contradiction"
            confidence = 0.72
        elif overlap > 0 and (a in b or b in a):
            label = "entailment"
            confidence = 0.68
        else:
            label = "neutral"
            confidence = 0.55

        return {
            "label": label,
            "confidence": float(confidence),
            "is_contradiction": label == "contradiction" and confidence >= self.threshold,
            "scores": {"contradiction": confidence if label == "contradiction" else 0.25, "entailment": confidence if label == "entailment" else 0.2, "neutral": confidence if label == "neutral" else 0.15},
            "method": "heuristic-fallback",
        }

    def detect_pair(self, text_a: str, text_b: str) -> Dict[str, Any]:
        if self.model is not None:
            try:
                scores = self.model.predict([(text_a, text_b)], show_progress_bar=False)[0]
                labels = ["contradiction", "entailment", "neutral"]
                label_idx = int(scores.argmax())
                label = labels[label_idx]
                confidence = float(scores[label_idx])
                return {
                    "label": label,
                    "confidence": float(confidence),
                    "is_contradiction": label == "contradiction" and confidence >= self.threshold,
                    "scores": {label_name: float(score) for label_name, score in zip(labels, scores)},
                    "method": "cross-encoder",
                }
            except Exception:
                pass
        return self._heuristic_label(text_a, text_b)

    def detect_conflicts(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        pairs = []
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                left = results[i]
                right = results[j]
                if left.get("document_id") == right.get("document_id"):
                    continue
                outcome = self.detect_pair(left.get("text", ""), right.get("text", ""))
                if outcome["is_contradiction"]:
                    pairs.append({
                        "evidence_a": left.get("chunk_id"),
                        "evidence_b": right.get("chunk_id"),
                        "label": outcome["label"],
                        "confidence": outcome["confidence"],
                        "status": "contradiction",
                        "pair": (left, right),
                        "method": outcome.get("method", "heuristic"),
                    })
        return pairs
