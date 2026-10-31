"""Prints the text retriever results for a few sample questions.

Run with: uv run python scripts/check_text_retriever.py
"""

from project_rag.retriever.text_retriever import TextRetriever

retriever = TextRetriever()

queries = [
    "what is the rule for grenades?",
    "how can I use a stratagem?",
    "what are the different ways to score points?",
]

for query in queries:
    print(f"\n{'=' * 80}\nQuery: {query}\n{'=' * 80}")
    for result in retriever.retrieve(query, k=3):
        print(
            f"\n[chunk {result.chunk_id}] type={result.chunk_type} "
            f"page={result.page_no} heading={result.heading!r} "
            f"rrf_score={result.rrf_score:.4f}"
        )
        print(result.text[:300])
