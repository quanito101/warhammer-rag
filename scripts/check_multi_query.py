"""Compares retrieval for the agent's own multi-query phrasings with the
original question alone. No LLM requests.

Run with: uv run python scripts/check_multi_query.py
"""

from project_rag.retriever.text_retriever import TextRetriever

retriever = TextRetriever()

cases = {
    "scoring, agent's own queries from the real run": [
        "how to score victory points primary secondary mission objectives",
        "controlling objective markers objective control OC page 58",
        "Victory point progressive objective secondary end of the battle",
    ],
    "scoring, the original failing phrasing alone": [
        "What are the different ways to score points?",
    ],
    "scoring, ambiguous term plus its spelled-out form": [
        "What are the different ways to score points?",
        "victory points scored in a mission",
    ],
}

for label, queries in cases.items():
    print(f"\n{label}")
    for rank, chunk in enumerate(retriever.retrieve_with_queries(queries, k=5), start=1):
        print(f"  {rank}. chunk {chunk.chunk_id} page {chunk.page_no}: {chunk.text[:70]!r}")
