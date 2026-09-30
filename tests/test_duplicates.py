from __future__ import annotations

from modules.duplicate_remover import DuplicateRemover


def test_duplicate_remover_identifies_redundant_items():
    items = [{
        "chunk_id": "a",
        "document_id": "d1",
        "title": "Alpha",
        "text": "Transformer models improve retrieval performance by using attention mechanisms.",
    }, {
        "chunk_id": "b",
        "document_id": "d2",
        "title": "Beta",
        "text": "Transformer models improve retrieval performance by using attention mechanisms.",
    }, {
        "chunk_id": "c",
        "document_id": "d3",
        "title": "Gamma",
        "text": "Different topic about uncertainty in retrieval pipelines.",
    }]
    remover = DuplicateRemover(threshold=0.92)
    kept, removed, duplicates = remover.detect_duplicates(items)
    assert len(kept) >= 1
    assert len(removed) >= 1 or len(duplicates) >= 1
