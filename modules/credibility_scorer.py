from __future__ import annotations

from typing import Any, Dict, List


class CredibilityScorer:
    def __init__(self, rules: Dict[str, Any] | None = None):
        self.rules = rules or {
            "source_type_weights": {"official": 1.0, "academic": 0.95, "news": 0.8, "blog": 0.6, "unknown": 0.5},
            "domain_bonus": {"ai": 0.1, "ml": 0.1, "research": 0.1, "science": 0.08, "education": 0.05, "unknown": 0.0},
            "has_url_bonus": 0.05,
            "has_publication_bonus": 0.05,
        }

    def score_document(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        source_type = str(metadata.get("source_type", "unknown")).lower()
        source_type_weight = self.rules["source_type_weights"].get(source_type, 0.5)
        domain = str(metadata.get("domain", "unknown")).lower()
        domain_bonus = self.rules["domain_bonus"].get(domain, 0.0)
        has_url = 1.0 if metadata.get("original_url") or metadata.get("source") else 0.0
        has_publication = 1.0 if metadata.get("publication_date") else 0.0

        base = source_type_weight + domain_bonus + (self.rules["has_url_bonus"] * has_url) + (self.rules["has_publication_bonus"] * has_publication)
        score = max(0.0, min(1.0, base / 1.4))

        return {
            "credibility_score": round(score, 4),
            "components": {
                "source_type_weight": round(source_type_weight, 4),
                "domain_bonus": round(domain_bonus, 4),
                "has_url_bonus": round(self.rules["has_url_bonus"] * has_url, 4),
                "has_publication_bonus": round(self.rules["has_publication_bonus"] * has_publication, 4),
            },
            "source_type": source_type,
            "domain": domain,
        }

    def score_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        scored = []
        for item in results:
            metadata = item.get("metadata", {}) or {}
            score_info = self.score_document(metadata)
            item = dict(item)
            item["credibility_score"] = score_info["credibility_score"]
            item["credibility_components"] = score_info["components"]
            scored.append(item)
        return scored
