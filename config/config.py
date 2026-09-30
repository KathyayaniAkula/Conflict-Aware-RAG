from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
PROCESSED_DIR = DATA_DIR / "processed"
EVALUATION_DIR = DATA_DIR / "evaluation"
VECTORSTORE_DIR = ROOT_DIR / "vectorstore"
RESULTS_DIR = ROOT_DIR / "results"
MODELS_DIR = ROOT_DIR / "models"


@dataclass
class Settings:
    project_title: str = "Transparent Legal AI using Enhanced Conflict-Aware Retrieval and Re-ranking"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    nli_model_name: str = "cross-encoder/nli-deberta-v3-base"
    ollama_model: str = "llama3.2"
    top_k: int = 8
    chunk_size: int = 500
    chunk_overlap: int = 80
    duplicate_threshold: float = 0.92
    contradiction_threshold: float = 0.75
    evidence_count: int = 4
    random_seed: int = 42
    rerank_weights: dict = field(default_factory=lambda: {"relevance": 0.45, "credibility": 0.25, "recency": 0.2, "contradiction": 0.1})
    credibility_rules: dict = field(default_factory=lambda: {
        "source_type_weights": {"official": 1.0, "academic": 0.95, "news": 0.8, "blog": 0.6, "unknown": 0.5},
        "domain_bonus": {"ai": 0.1, "ml": 0.1, "research": 0.1, "science": 0.08, "education": 0.05, "unknown": 0.0},
        "has_url_bonus": 0.05,
        "has_publication_bonus": 0.05,
    })
    recency_lambda: float = 0.05
    dataset_version: str = "v1"


SETTINGS = Settings()


def ensure_directories() -> None:
    for path in [DOCUMENTS_DIR, PROCESSED_DIR, EVALUATION_DIR, VECTORSTORE_DIR, RESULTS_DIR, MODELS_DIR, RESULTS_DIR / "raw", RESULTS_DIR / "tables", RESULTS_DIR / "plots"]:
        path.mkdir(parents=True, exist_ok=True)


ensure_directories()
