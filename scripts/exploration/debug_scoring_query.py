import json

from project_rag.retriever.text_retriever import TextRetriever, _tokenize

chunks = json.loads(open("data/processed/chunks.json").read())

print("--- Chunks mentioning 'score'/'scoring'/'victory point'/'objective marker' ---")
for chunk in chunks:
    text_lower = chunk["text"].lower()
    if any(term in text_lower for term in ("score", "scoring", "victory point", "objective marker")):
        print(f"\n[chunk {chunk['chunk_id']}] type={chunk['chunk_type']} page={chunk['page_no']} heading={chunk['heading']!r}")
        print(chunk["text"][:400])

retriever = TextRetriever()
query = "what are the different ways to score points?"
bm25_ranking = retriever._bm25_ranking(query, 201)
vector_ranking = retriever._vector_ranking(query, 201)

print("\n--- Where scoring-related chunks rank in each channel ---")
scoring_ids = [c["chunk_id"] for c in chunks if "score" in c["text"].lower() or "victory point" in c["text"].lower()]
for chunk_id in scoring_ids:
    bm25_rank = bm25_ranking.index(chunk_id) if chunk_id in bm25_ranking else None
    vector_rank = vector_ranking.index(chunk_id) if chunk_id in vector_ranking else None
    print(f"chunk {chunk_id}: bm25_rank={bm25_rank}, vector_rank={vector_rank}")

print("\n\n--- Chunks explaining HOW to use a stratagem (procedure, not a specific card) ---")
for chunk in chunks:
    text_lower = chunk["text"].lower()
    if chunk["chunk_type"] != "stratagem" and "command point" in text_lower:
        print(f"\n[chunk {chunk['chunk_id']}] type={chunk['chunk_type']} page={chunk['page_no']} heading={chunk['heading']!r}")
        print(chunk["text"][:400])
