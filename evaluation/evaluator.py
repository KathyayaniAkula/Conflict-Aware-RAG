from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Sequence

import pandas as pd

from evaluation.metrics import contradiction_accuracy, f1_score, mean_reciprocal_rank, ndcg_at_k, precision_at_k, recall_at_k


class Evaluator:
    def __init__(self, dataset_path: str | Path):
        self.dataset_path = Path(dataset_path)
        self.dataset = self._load_dataset()

    def _load_dataset(self) -> Dict[str, Any]:
        with self.dataset_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def evaluate_retrieval(self, retrieved_doc_ids: Sequence[str], k: int = 5) -> Dict[str, Any]:
        relevant = self.dataset.get("relevant_document_ids", [])
        metrics = {
            "precision@k": precision_at_k(relevant, retrieved_doc_ids, k),
            "recall@k": recall_at_k(relevant, retrieved_doc_ids, k),
        }
        metrics["f1"] = f1_score(metrics["precision@k"], metrics["recall@k"])
        return metrics

    def evaluate_ranking(self, retrieved_doc_ids: Sequence[str], actual_scores: Sequence[float], k: int = 5) -> Dict[str, Any]:
        relevant = self.dataset.get("relevant_document_ids", [])
        return {
            "mrr": mean_reciprocal_rank(relevant, retrieved_doc_ids),
            "ndcg@k": ndcg_at_k(actual_scores, actual_scores, k),
        }

    def evaluate_contradiction(self, true_labels: Sequence[str], pred_labels: Sequence[str]) -> Dict[str, Any]:
        if not true_labels:
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == "contradiction" and p == "contradiction")
        fp = sum(1 for t, p in zip(true_labels, pred_labels) if t != "contradiction" and p == "contradiction")
        fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == "contradiction" and p != "contradiction")
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = f1_score(precision, recall)
        accuracy = contradiction_accuracy(true_labels, pred_labels)
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    def export_summary(self, results: List[Dict[str, Any]], output_path: str | Path) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(results)
        df.to_csv(path, index=False)
