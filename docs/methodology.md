# Methodology

The project studies the effect of transparent conflict-aware retrieval on answer quality and evidence selection. The baseline uses standard retrieval followed by LLM generation without additional filtering or reranking. The proposed pipeline improves evidence selection using explicit duplicate suppression, credibility estimation, recency awareness, contradiction detection, and reranking.

## Prediction goal

The system is not a legal decision engine. It is a general-purpose comparative research pipeline for retrieval transparency.

## Rules

- duplicate detection uses semantic similarity with an embedding threshold
- credibility is calculated from source type, domain, URL presence, and publication date presence using a transparent heuristic
- recency is based on a deterministic decay formula over age in days
- contradiction detection uses a local NLI model when available
- the reranker is transparent and configurable
- confidence is estimated from actual retrieval and evidence signals rather than a raw LLM self-report

## Experimental framing

Ablation configurations are used to compare:

- A: Baseline RAG
- B: Baseline + Duplicate Removal
- C: B + Credibility + Recency
- D: C + Contradiction Detection
- E: Full Proposed Conflict-Aware RAG
