from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb

from config.config import SETTINGS, VECTORSTORE_DIR
from modules.embeddings import EmbeddingModel


class VectorStoreIndex:
    def __init__(self, collection_name: str = "research_chunks", persist_directory: Optional[str | Path] = None):
        self.persist_directory = Path(persist_directory or VECTORSTORE_DIR)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.embedding_model = EmbeddingModel(SETTINGS.embedding_model_name)

    def add_documents(self, chunks: List[Dict[str, Any]]) -> None:
        if not chunks:
            return
        ids = [chunk["chunk_id"] for chunk in chunks]
        texts = [chunk.get("text", "") for chunk in chunks]
        embeddings = self.embedding_model.encode(texts)
        metadatas = []
        for chunk in chunks:
            metadata = {k: str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v for k, v in chunk.items() if k not in {"text", "embedding"}}
            metadatas.append(metadata)
        self.collection.add(ids=ids, embeddings=embeddings.tolist(), documents=texts, metadatas=metadatas)

    def query(self, query: str, top_k: int = 5, where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if not query.strip():
            return []
        q_embedding = self.embedding_model.encode_query(query)
        results = self.collection.query(
            query_embeddings=[q_embedding.tolist()],
            n_results=top_k,
            where=where,
        )
        items: List[Dict[str, Any]] = []
        for idx, doc_id in enumerate(results.get("ids", [[]])[0]):
            metadata = results.get("metadatas", [[]])[0][idx] if results.get("metadatas") else {}
            score = results.get("distances", [[]])[0][idx] if results.get("distances") else 0.0
            items.append({
                "chunk_id": doc_id,
                "document_id": metadata.get("document_id", ""),
                "title": metadata.get("title", ""),
                "source": metadata.get("source", ""),
                "source_type": metadata.get("source_type", "unknown"),
                "publication_date": metadata.get("publication_date", ""),
                "text": results.get("documents", [[]])[0][idx],
                "score": float(score),
                "metadata": metadata,
            })
        return items

    def count(self) -> int:
        return self.collection.count()

    def get_collection_info(self) -> Dict[str, Any]:
        return {
            "collection_name": self.collection.name,
            "count": self.collection.count(),
            "embedding_model": SETTINGS.embedding_model_name,
        }

    def persist_metadata(self, path: str | Path) -> None:
        out = self.collection.get(include=["metadatas", "documents", "ids"]) 
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump({"ids": out["ids"], "documents": out["documents"], "metadatas": out["metadatas"]}, file, indent=2)
