from __future__ import annotations

from config.config import SETTINGS
from scripts.ingest_data import build_vectorstore


def test_retrieval_returns_results():
    store = build_vectorstore()
    results = store.query("How can retrieval systems improve transparency and reduce duplicate evidence?", top_k=3)
    assert len(results) > 0
    assert all("text" in item for item in results)
