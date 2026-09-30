from __future__ import annotations

import json
import subprocess
from typing import Any, Dict, List


class OllamaClient:
    def __init__(self, model_name: str = "llama3.2"):
        self.model_name = model_name

    def is_available(self) -> bool:
        try:
            result = subprocess.run(["ollama", "--version"], capture_output=True, text=True, check=False)
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def generate(self, prompt: str) -> str:
        if not self.is_available():
            return (
                "Ollama is not installed or unavailable in this environment. "
                "Install Ollama for Windows, then run: 'ollama pull llama3.2' and retry. "
                "No generated answer was produced because the required local LLM dependency is missing."
            )
        command = ["ollama", "run", self.model_name, prompt]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            return (
                "Ollama generation failed. The local model may not be available. "
                f"Details: {result.stderr.strip() or result.stdout.strip()}"
            )
        return result.stdout.strip()


class LLMGenerator:
    def __init__(self, model_name: str = "llama3.2"):
        self.client = OllamaClient(model_name)

    def build_prompt(self, query: str, evidence: List[Dict[str, Any]]) -> str:
        context_lines = []
        for idx, item in enumerate(evidence, start=1):
            context_lines.append(f"Evidence {idx}:\nTitle: {item.get('title', 'Unknown')}\nSource: {item.get('source', 'Unknown')}\nText: {item.get('text', '')}\n")
        evidence_text = "\n\n".join(context_lines) or "No evidence available."
        return (
            "You are answering using only the evidence supplied below. "
            "If the evidence is insufficient, explicitly state that the evidence is insufficient. "
            "Do not invent citations or sources. Do not claim certainty beyond the evidence.\n\n"
            f"User query: {query}\n\nEvidence:\n{evidence_text}\n\n"
            "Provide a transparent answer that notes uncertainty and conflicting evidence when relevant."
        )

    def generate_answer(self, query: str, evidence: List[Dict[str, Any]]) -> str:
        if not evidence:
            return "Evidence is insufficient to answer this query from the available retrieval set."
        prompt = self.build_prompt(query, evidence)
        return self.client.generate(prompt)
