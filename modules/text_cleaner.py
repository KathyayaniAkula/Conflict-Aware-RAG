from __future__ import annotations

import re
from typing import List


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def remove_noise(text: str) -> str:
    cleaned = text.replace("\xa0", " ")
    cleaned = re.sub(r"[\t\r\f\v]+", " ", cleaned)
    return normalize_whitespace(cleaned)


def clean_text(text: str) -> str:
    cleaned = remove_noise(text)
    cleaned = cleaned.replace("\n", " ")
    return normalize_whitespace(cleaned)


def normalize_documents(documents: List[dict]) -> List[dict]:
    for item in documents:
        if "text" in item:
            item["text"] = clean_text(item["text"])
    return documents
