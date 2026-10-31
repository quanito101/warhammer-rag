from project_rag.llm.gemini_client import GeminiClient
from project_rag.retriever.text_retriever import TextRetriever

retriever = TextRetriever()
llm_client = GeminiClient()
query = "what are the different ways to score points?"

print("--- Without query fusion (baseline, same as before) ---")
for result in retriever.retrieve(query, k=5):
    print(f"[chunk {result.chunk_id}] page={result.page_no} rrf_score={result.rrf_score:.4f}")

print("\n--- With query fusion ---")
for result in retriever.retrieve_with_query_fusion(query, llm_client=llm_client, k=5):
    print(f"[chunk {result.chunk_id}] page={result.page_no} rrf_score={result.rrf_score:.4f}")
    print(result.text[:200].replace(chr(10), " "))
    print()
