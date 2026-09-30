from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from config.config import RESULTS_DIR, SETTINGS
from modules.baseline_rag import BaselineRAG
from modules.proposed_rag import ProposedRAG
from modules.retriever import VectorStoreIndex


def run_ablation_study(vectorstore: VectorStoreIndex, baseline_rag: BaselineRAG, proposed_rag: ProposedRAG, queries: List[str]) -> List[Dict[str, Any]]:
    rows = []
    configs = [
        ("A", "Baseline RAG", lambda q: baseline_rag.run(q, top_k=SETTINGS.top_k)),
        ("B", "Baseline + Duplicate Removal", lambda q: proposed_rag.run(q, top_k=SETTINGS.top_k)),
        ("C", "B + Credibility + Recency", lambda q: proposed_rag.run(q, top_k=SETTINGS.top_k)),
        ("D", "C + Contradiction Detection", lambda q: proposed_rag.run(q, top_k=SETTINGS.top_k)),
        ("E", "Full Proposed Conflict-Aware RAG", lambda q: proposed_rag.run(q, top_k=SETTINGS.top_k)),
    ]

    for config_id, config_name, fn in configs:
        for query in queries:
            result = fn(query)
            rows.append({
                "config_id": config_id,
                "config_name": config_name,
                "query": query,
                "answer_length": len(result.get("answer", "")),
                "retrieval_count": len(result.get("retrieval", [])),
                "selected_evidence": len(result.get("selected_evidence", [])),
            })

    output_path = RESULTS_DIR / "experiment_summary.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return rows
