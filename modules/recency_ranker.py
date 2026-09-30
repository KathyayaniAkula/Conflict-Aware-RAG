from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional


class RecencyRanker:
    def __init__(self, decay_lambda: float = 0.05):
        self.decay_lambda = decay_lambda

    def parse_date(self, value: Any) -> Optional[datetime]:
        if value in (None, "", "unknown"):
            return None
        text = str(value).strip()
        if not text:
            return None
        for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%Y"]:
            try:
                return datetime.strptime(text[:len(fmt)], fmt)
            except ValueError:
                continue
        match = re.search(r"(\d{4})-(\d{2})-(\d{2})", text)
        if match:
            try:
                return datetime.strptime(match.group(0), "%Y-%m-%d")
            except ValueError:
                pass
        return None

    def score_date(self, publication_date: Any) -> float:
        parsed = self.parse_date(publication_date)
        if parsed is None:
            return 0.5
        now = datetime.now()
        age_days = max((now - parsed).days, 0)
        return float(max(0.0, min(1.0, 1.0 / (1.0 + self.decay_lambda * age_days))))

    def score_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        scored = []
        for item in results:
            item = dict(item)
            value = item.get("publication_date") or item.get("metadata", {}).get("publication_date")
            score = self.score_date(value)
            item["recency_score"] = round(score, 4)
            item["publication_date"] = value
            scored.append(item)
        return scored
