from __future__ import annotations

from modules.llm_generator import LLMGenerator
from modules.proposed_rag import ProposedRAG
from modules.retriever import VectorStoreIndex
from scripts.ingest_data import build_vectorstore


def main() -> None:
    vectorstore = build_vectorstore()
    llm = LLMGenerator(model_name="llama3.2")
    system = ProposedRAG(vectorstore, llm)
    query = "How can retrieval systems improve transparency and reduce duplicate evidence?"
    result = system.run(query, top_k=8)
    print(result)


if __name__ == "__main__":
    main()
