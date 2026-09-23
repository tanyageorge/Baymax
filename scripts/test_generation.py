"""
Milestone 2: full RAG pipeline test — retrieve chunks, then generate a
grounded answer using an LLM.

Prerequisite: LLM_PROVIDER + the matching API key must be set in
backend/.env (see app/llm_client.py) before this will work.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.llm_client import generate_answer
from test_retrieval import build_index


def answer_query(index, query: str, top_k: int = 3) -> None:
    retriever = index.as_retriever(similarity_top_k=top_k)
    results = retriever.retrieve(query)

    retrieved_chunks = [
        {
            "source": node.node.metadata.get("file_name", "unknown"),
            "text": node.node.get_text(),
        }
        for node in results
    ]

    print(f"\n=== Query: {query!r} ===")
    print(f"Retrieved from: {[c['source'] for c in retrieved_chunks]}")

    answer = generate_answer(query, retrieved_chunks)
    print(f"\nBaymax says:\n{answer}\n")
    print("-" * 80)


if __name__ == "__main__":
    index = build_index()

    test_queries = [
        "my kid has a fever and won't stop crying",
        "I got stung by a bee and my throat feels tight",
    ]

    for q in test_queries:
        answer_query(index, q)