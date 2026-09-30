from __future__ import annotations

from typing import Any, Dict, List

from config.config import SETTINGS
from modules.citation_generator import CitationGenerator
from modules.credibility_scorer import CredibilityScorer
from modules.duplicate_remover import DuplicateRemover
from modules.evidence_selector import EvidenceSelector
from modules.confidence_estimator import ConfidenceEstimator
from modules.contradiction_detector import ContradictionDetector
from modules.llm_generator import LLMGenerator
from modules.recency_ranker import RecencyRanker
from modules.reranker import ConflictAwareReranker
from modules.retriever import VectorStoreIndex


class ProposedRAG:
    def __init__(self, vectorstore: VectorStoreIndex, llm: LLMGenerator):
        self.vectorstore = vectorstore
        self.llm = llm
        self.duplicate_remover = DuplicateRemover(threshold=SETTINGS.duplicate_threshold)
        self.credibility_scorer = CredibilityScorer(SETTINGS.credibility_rules)
        self.recency_ranker = RecencyRanker(decay_lambda=SETTINGS.recency_lambda)
        self.contradiction_detector = ContradictionDetector(threshold=SETTINGS.contradiction_threshold)
        self.reranker = ConflictAwareReranker(SETTINGS.rerank_weights)
        self.evidence_selector = EvidenceSelector(SETTINGS.evidence_count)
        self.confidence_estimator = ConfidenceEstimator()
        self.citation_generator = CitationGenerator()

    def run(self, query: str, top_k: int = 8) -> Dict[str, Any]:
        retrieval = self.vectorstore.query(query, top_k=top_k)
        duplicate_state = self.duplicate_remover.filter_duplicates(retrieval)
        kept = duplicate_state.get("kept", retrieval)
        removed = duplicate_state.get("removed", [])
        duplicates = duplicate_state.get("duplicate_records", [])
        scored = self.credibility_scorer.score_results(kept)
        recency_scored = self.recency_ranker.score_results(scored)
        contradictions = self.contradiction_detector.detect_conflicts(recency_scored)
        reranked = self.reranker.rerank(recency_scored, contradictions)
        evidence = self.evidence_selector.select(reranked)
        answer = self.llm.generate_answer(query, evidence)
        confidence = self.confidence_estimator.estimate(reranked, evidence, contradictions)
        citations = self.citation_generator.generate(evidence, answer)

        return {
            "query": query,
            "retrieval": retrieval,
            "duplicate_removal": {
                "kept": kept,
                "removed": removed,
                "duplicate_records": duplicates,
                "count_removed": duplicate_state.get("count_removed", len(removed)),
            },
            "credibility_scores": scored,
            "recency_scores": recency_scored,
            "contradictions": contradictions,
            "reranked": reranked,
            "selected_evidence": evidence,
            "answer": answer,
            "confidence": confidence,
            "citations": citations,
            "pipeline": ["query", "retrieve", "duplicate_removal", "credibility", "recency", "contradiction", "rerank", "evidence", "llm", "confidence", "citations"],
        }
