"""Shows where the Assault chunk (chunk 97, page 25) ranks for q03 in each
retrieval channel and after fusion. No LLM requests.

Run with: uv run python scripts/diagnose_q03_ranking.py
"""

from project_rag.retriever.text_retriever import TextRetriever

TARGET = 97
AGENT_QUERIES = [
    "unit shoot after advance advance move",
    "advance shooting phase eligible to shoot",
    "advanced can it shoot assault",
]
QUESTION = "Can a unit shoot in the turn it Advanced?"

retriever = TextRetriever()
total = len(retriever.chunks)


def position(ranking: list[int]) -> str:
    return str(ranking.index(TARGET) + 1) if TARGET in ranking else f">{len(ranking)}"


print(f"Rank of chunk {TARGET} per query (1 = best)\n")
print(f"{'BM25':>6} {'vector':>7}  query")
for query in [*AGENT_QUERIES, QUESTION]:
    bm25 = retriever._bm25_ranking(query, total)
    vector = retriever._vector_ranking(query, total)
    print(f"{position(bm25):>6} {position(vector):>7}  {query}")

print("\nFused rank of chunk", TARGET)
for label, queries in (
    ("agent's three queries", AGENT_QUERIES),
    ("question alone", [QUESTION]),
    ("question plus agent's queries", [QUESTION, *AGENT_QUERIES]),
):
    fused = [c.chunk_id for c in retriever._fused_chunks(queries, total, 20)]
    print(f"  {position(fused):>5}  {label}")
