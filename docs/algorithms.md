# Algorithms

## Duplicate filtering

Duplicate detection uses embedding cosine similarity between candidate chunks. If the similarity is above the configured threshold, the later item is treated as a duplicate and excluded from final evidence selection.

## Credibility scoring

The credibility score is a heuristic with the form:

score = (source_type_weight + domain_bonus + url_bonus + publication_bonus) / 1.4

The result is clipped to [0, 1].

## Recency scoring

Recency is computed as:

score = 1 / (1 + lambda * age_in_days)

The score is normalized to [0, 1] with missing dates treated as neutral score 0.5.

## Contradiction detection

The contradiction model scores each evidence pair as contradiction, entailment, or neutral. A pair is labeled a contradiction only when the contradiction probability is above the threshold.

## Conflict-aware reranking

FinalScore = w1 * RelevanceScore + w2 * CredibilityScore + w3 * RecencyScore - w4 * ContradictionPenalty

The implementation exposes these weights through configuration and allows transparent inspection of each component.

## Confidence estimation

The confidence score is computed from actual retrieval and evidence signals, including average relevance, credibility, recency, evidence coverage, diversity, and contradiction pressure. The result is normalized to [0, 1].
