from __future__ import annotations

from typing import List, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        embeddings = self.model.encode(list(texts), show_progress_bar=False, convert_to_numpy=True)
        return np.asarray(embeddings, dtype=np.float32)

    def similarity(self, a: Sequence[float], b: Sequence[float]) -> float:
        a_vec = np.asarray(a, dtype=np.float32)
        b_vec = np.asarray(b, dtype=np.float32)
        a_norm = np.linalg.norm(a_vec)
        b_norm = np.linalg.norm(b_vec)
        if a_norm == 0 or b_norm == 0:
            return 0.0
        return float(np.dot(a_vec, b_vec) / (a_norm * b_norm))

    def encode_query(self, query: str) -> np.ndarray:
        return self.encode([query])[0]

    def info(self) -> dict:
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.model.get_sentence_embedding_dimension(),
        }
