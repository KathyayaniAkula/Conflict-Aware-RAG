from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np


def precision_at_k(relevant: Sequence[str], retrieved: Sequence[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(top_k)


def recall_at_k(relevant: Sequence[str], retrieved: Sequence[str], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def dcg_at_k(scores: Sequence[float], k: int) -> float:
    if k <= 0:
        return 0.0
    rel = scores[:k]
    dcg = 0.0
    for i, score in enumerate(rel, start=1):
        dcg += (2**score - 1) / np.log2(i + 1)
    return dcg


def ndcg_at_k(actual: Sequence[float], predicted: Sequence[float], k: int) -> float:
    if k <= 0:
        return 0.0
    ideal = sorted(actual, reverse=True)[:k]
    dcg_pred = dcg_at_k(predicted[:k], k)
    dcg_ideal = dcg_at_k(ideal, k)
    if dcg_ideal == 0:
        return 0.0
    return dcg_pred / dcg_ideal


def mean_reciprocal_rank(relevant: Sequence[str], retrieved: Sequence[str]) -> float:
    for index, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / index
    return 0.0


def contradiction_accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    if not y_true:
        return 0.0
    matches = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    return matches / len(y_true)
