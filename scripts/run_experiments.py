from __future__ import annotations

from scripts.ingest_data import build_vectorstore
from modules.baseline_rag import BaselineRAG
from modules.llm_generator import LLMGenerator
from modules.proposed_rag import ProposedRAG
from evaluation.experiments import run_ablation_study


def main() -> None:
    vectorstore = build_vectorstore()
    llm = LLMGenerator(model_name="llama3.2")
    baseline = BaselineRAG(vectorstore, llm)
    proposed = ProposedRAG(vectorstore, llm)
    queries = [
        "How can retrieval systems improve transparency and reduce duplicate evidence?",
        "What factors matter when selecting evidence in dense retrieval pipelines?",
        "Does dense retrieval consistently outperform sparse retrieval?",
        "What are the trade-offs between credibility and recency in evidence ranking?",
    ]
    rows = run_ablation_study(vectorstore, baseline, proposed, queries)
    print(rows)


if __name__ == "__main__":
    main()
