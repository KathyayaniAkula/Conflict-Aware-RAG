from __future__ import annotations

from typing import List, Dict, Any


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 80) -> List[str]:
    if not text:
        return []
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0
    step = max(1, chunk_size - chunk_overlap)
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start += step
    return chunks


def chunk_documents(documents: List[Dict[str, Any]], chunk_size: int, chunk_overlap: int) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    for document in documents:
        text = document.get("text", "")
        for idx, chunk_text_value in enumerate(chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap), start=1):
            chunk = dict(document)
            chunk["chunk_id"] = f"{document.get('document_id', 'doc')}_chunk_{idx}"
            chunk["text"] = chunk_text_value
            chunk["chunk_index"] = idx
            chunks.append(chunk)
    return chunks
