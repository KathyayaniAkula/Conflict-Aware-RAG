from __future__ import annotations

import json
from pathlib import Path

from config.config import DATA_DIR, DOCUMENTS_DIR, EVALUATION_DIR, SETTINGS
from modules.document_loader import load_documents_from_directory
from modules.text_cleaner import clean_text
from modules.text_splitter import chunk_documents
from modules.retriever import VectorStoreIndex


def build_seed_dataset() -> None:
    dataset_dir = DATA_DIR / "documents"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    docs = [
        {
            "document_id": "doc_ai_1",
            "title": "AI Research Overview",
            "source": "Synthetic Research Institute",
            "source_type": "academic",
            "publication_date": "2024-06-10",
            "domain": "ai",
            "original_url": "https://example.org/ai-overview",
            "text": "Transformer models improve retrieval performance by using attention mechanisms and efficient embeddings. Recent AI research shows strong benefits from dense retrieval and contextualized ranking."
        },
        {
            "document_id": "doc_ai_2",
            "title": "AI Research Overview Duplicate",
            "source": "Synthetic Research Institute",
            "source_type": "academic",
            "publication_date": "2024-06-15",
            "domain": "ai",
            "original_url": "https://example.org/ai-overview-dup",
            "text": "Transformer models improve retrieval performance by using attention mechanisms and efficient embeddings. Recent AI research shows strong benefits from dense retrieval and contextualized ranking."
        },
        {
            "document_id": "doc_ml_1",
            "title": "Machine Learning Findings",
            "source": "Open Tech Journal",
            "source_type": "news",
            "publication_date": "2023-10-01",
            "domain": "ml",
            "original_url": "https://example.org/ml-findings",
            "text": "Dense retrievers are effective but may be sensitive to noisy data and duplicate evidence. Multi-stage retrieval pipelines can reduce redundancy and improve answer transparency."
        },
        {
            "document_id": "doc_research_older",
            "title": "Research Method Prior Study",
            "source": "Institute Archive",
            "source_type": "official",
            "publication_date": "2019-05-11",
            "domain": "research",
            "original_url": "https://example.org/older-study",
            "text": "Traditional vector search systems rely on lexical overlap and do not strongly account for contradiction detection. They often struggle when multiple sources disagree."
        },
        {
            "document_id": "doc_research_new",
            "title": "Modern Research Method",
            "source": "Department Bulletin",
            "source_type": "official",
            "publication_date": "2025-01-15",
            "domain": "research",
            "original_url": "https://example.org/new-study",
            "text": "Conflict-aware reranking improves evidence selection by penalizing contradictory claims while preserving credible and recent results. Dense retrieval systems benefit from transparency."
        },
        {
            "document_id": "doc_incorrect",
            "title": "Low quality note",
            "source": "Personal Blog",
            "source_type": "blog",
            "publication_date": "2021-09-10",
            "domain": "unknown",
            "original_url": "https://example.org/blog-note",
            "text": "AI systems always succeed without any uncertainty and never require evidence. Retrieval is just a single step with no need for transparency."
        },
        {
            "document_id": "doc_contra_1",
            "title": "Contradictory Position A",
            "source": "Research Notebook",
            "source_type": "academic",
            "publication_date": "2024-03-22",
            "domain": "science",
            "original_url": "https://example.org/contra-a",
            "text": "Dense retrieval consistently outperforms sparse retrieval for all general research queries. The improvement is robust across all tasks."
        },
        {
            "document_id": "doc_contra_2",
            "title": "Contradictory Position B",
            "source": "Benchmark Review",
            "source_type": "news",
            "publication_date": "2024-06-01",
            "domain": "science",
            "original_url": "https://example.org/contra-b",
            "text": "Dense retrieval does not consistently outperform sparse retrieval across all tasks; performance depends on the dataset, query, and evidence quality."
        },
    ]

    dataset_path = DATA_DIR / "evaluation" / "controlled_dataset.json"
    with dataset_path.open("w", encoding="utf-8") as handle:
        json.dump({
            "name": "Controlled Evaluation Data",
            "documents": docs,
            "queries": [
                "How can retrieval systems improve transparency and reduce duplicate evidence?",
                "What factors matter when selecting evidence in dense retrieval pipelines?",
                "Does dense retrieval consistently outperform sparse retrieval?",
                "What are the trade-offs between credibility and recency in evidence ranking?"
            ],
            "metadata": {
                "notes": "Controlled synthetic evaluation dataset intended for general research experiments."
            }
        }, handle, indent=2)


def build_vectorstore() -> VectorStoreIndex:
    doc_path = DATA_DIR / "evaluation" / "controlled_dataset.json"
    if not doc_path.exists():
        build_seed_dataset()
    documents = load_documents_from_directory(DATA_DIR / "evaluation")
    cleaned = []
    for doc in documents:
        doc.text = clean_text(doc.text)
        cleaned.append(doc.__dict__)
    chunks = chunk_documents(cleaned, SETTINGS.chunk_size, SETTINGS.chunk_overlap)
    store = VectorStoreIndex(collection_name="research_chunks", persist_directory=DATA_DIR / "processed")
    store.add_documents(chunks)
    return store


def main() -> None:
    build_seed_dataset()
    build_vectorstore()
    print("Seed dataset and vectorstore created successfully.")


if __name__ == "__main__":
    main()
