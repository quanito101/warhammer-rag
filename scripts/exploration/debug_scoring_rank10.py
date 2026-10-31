from project_rag.retriever.text_retriever import TextRetriever

retriever = TextRetriever()
query = "what are the different ways to score points?"

for result in retriever.retrieve(query, k=10):
    print(f"[chunk {result.chunk_id}] type={result.chunk_type} page={result.page_no} rrf_score={result.rrf_score:.4f}")
    print(result.text[:150].replace(chr(10), " "))
    print()
