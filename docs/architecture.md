# Architecture

The system is a local-first research pipeline implemented as a modular Python project. It compares a baseline retrieval pipeline with a proposed conflict-aware retrieval and re-ranking pipeline.

## High-level flow

Baseline:

Query -> Retrieval -> LLM -> Answer

Proposed:

Query -> Retrieval -> Duplicate removal -> Credibility scoring -> Recency scoring -> Contradiction detection -> Conflict-aware re-ranking -> Evidence selection -> LLM -> Confidence estimation -> Citations -> Transparent answer

## Modules

- document_loader.py: loads text, PDF, DOCX, CSV, and JSON data and preserves metadata
- text_cleaner.py: normalizes whitespace and noise
- text_splitter.py: chunking with configurable size and overlap
- embeddings.py: local semantic embedding model wrapper
- retriever.py: ChromaDB-backed retrieval
- duplicate_remover.py: semantic duplicate filtering using embeddings
- credibility_scorer.py: deterministic source credibility heuristic
- recency_ranker.py: date-based recency scoring
- contradiction_detector.py: contradiction detection via a cross-encoder NLI model
- reranker.py: proposed formula combining relevance, credibility, recency, and contradiction penalty
- evidence_selector.py: selects useful and diverse evidence
- llm_generator.py: Ollama local LLM integration
- confidence_estimator.py: transparent scoring from system evidence
- citation_generator.py: citation traceability

## Data flow

Documents are loaded and chunked, then indexed into ChromaDB with metadata preserved. Retrieval returns chunk-level matches. The proposed pipeline applies duplicate removal, scoring, contradiction analysis, and reranking before evidence selection and answer generation.
