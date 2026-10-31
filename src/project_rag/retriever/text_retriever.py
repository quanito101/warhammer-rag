"""
Retrieves relevant Core Rules chunks for a query, combining two retrieval
channels per the course's taught toolkit (lexical + embedding) via
Reciprocal Rank Fusion (RRF), rather than relying on either channel alone.

Why both channels: BM25 (lexical) is strong on exact terminology match
(e.g. "Fire Overwatch", "Oath of Moment") but blind to paraphrase; vector
search (Sentence-BERT embeddings via ChromaDB) is strong on semantic
paraphrase ("how do I get extra attacks") but can miss a rare exact term
in a larger corpus. RRF (score(D) = sum over retrievers of
1/(rank_i(D) + k)) combines both rankings without needing their raw scores
to be on the same scale -- see docs/retrieval.md for the full writeup and
the retriever comparison this choice is meant to support.

This module only handles the Core Rules text chunks (data/processed/
chunks.json). The SQLite datasheet retriever is separate
(retriever/sql_retriever.py); routing between the two is handled one level
up (chatbot/router.py).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from rank_bm25 import BM25Okapi

from project_rag.chunker.chunker import load_chunks
from project_rag.llm.gemini_client import GeminiClient

DEFAULT_CHUNKS_PATH = Path("data/processed/chunks.json")
DEFAULT_CHROMA_DIR = Path("data/processed/chroma_db")
COLLECTION_NAME = "core_rules_chunks"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
RRF_K = 60

QUERY_REWRITE_PROMPT = """Given a question about Warhammer 40,000 rules, write {num_variants} alternative ways to ask the same question, using different but related terminology (synonyms, related game concepts, more specific rules vocabulary). Output exactly {num_variants} lines, one phrasing per line, no numbering and no extra commentary.

Question: {query}
"""

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def chunks_fingerprint(chunks: list[dict]) -> str:
    """Hash of the chunk ids and texts, stored with the Chroma collection so
    that an index built from older chunk text is detected and rebuilt.
    """
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(f"{chunk['chunk_id']}\x00{chunk['text']}\x00".encode())
    return digest.hexdigest()


def _tokenize(text: str) -> list[str]:
    """Lowercases and splits on runs of letters/digits, dropping all
    punctuation. BM25Okapi expects pre-tokenized text; this is deliberately
    simple (no stemming, no stopword removal) since the corpus is small
    (201 chunks) and exact-term matching (unit/stratagem names) is the
    point of having a lexical channel at all -- stemming risks collapsing
    distinctions BM25 exists here to catch.
    """
    return _TOKEN_PATTERN.findall(text.lower())


def parse_query_variants(llm_response: str, expected_count: int) -> list[str]:
    """Splits an LLM's rewritten-query response into individual query
    strings, one per non-empty line, truncated to `expected_count`. Kept
    as a standalone function so the parsing itself is unit-testable
    without a real LLM call.
    """
    lines = [line.strip() for line in llm_response.strip().splitlines() if line.strip()]
    return lines[:expected_count]


@dataclass
class RetrievedChunk:
    chunk_id: int
    text: str
    page_no: int | None
    heading: str | None
    chunk_type: str
    rrf_score: float


def reciprocal_rank_fusion(
    rankings: list[list[int]], k: int = RRF_K
) -> list[tuple[int, float]]:
    """Combines several ranked lists of chunk_ids into one ranking.

    Each list is assumed already sorted best-first. A chunk_id's score is
    the sum, over every list it appears in, of 1/(rank + k) where rank is
    its 0-based position in that list; a chunk_id absent from a list
    contributes 0 for that list, not a penalty. Returns (chunk_id, score)
    pairs sorted by score descending. This is the formula given in the
    course material (score(D) = sum_i 1/(rank_i(D) + k)), with k following
    the same default used there.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rank + k)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


class TextRetriever:
    """Loads the Core Rules chunks once and answers retrieve() calls by
    combining BM25 and vector search through reciprocal_rank_fusion.

    Building the BM25 index and the Chroma collection both happen in
    __init__ rather than lazily, since both need the full chunk list up
    front and neither is meant to be rebuilt per query. The Chroma
    collection is persisted to `chroma_dir` so embeddings are computed once
    across runs, not once per TextRetriever instantiation.
    """

    def __init__(
        self,
        chunks_path: Path | str = DEFAULT_CHUNKS_PATH,
        chroma_dir: Path | str = DEFAULT_CHROMA_DIR,
    ) -> None:
        self.chunks = load_chunks(chunks_path)
        self._chunks_by_id = {c["chunk_id"]: c for c in self.chunks}

        tokenized = [_tokenize(c["text"]) for c in self.chunks]
        self._bm25 = BM25Okapi(tokenized)

        client = chromadb.PersistentClient(path=str(chroma_dir))
        embedding_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
        fingerprint = chunks_fingerprint(self.chunks)
        self._collection = client.get_or_create_collection(
            name=COLLECTION_NAME, embedding_function=embedding_fn, metadata={"fingerprint": fingerprint}
        )
        if (self._collection.metadata or {}).get("fingerprint") != fingerprint:
            client.delete_collection(COLLECTION_NAME)
            self._collection = client.create_collection(
                name=COLLECTION_NAME, embedding_function=embedding_fn, metadata={"fingerprint": fingerprint}
            )
        if self._collection.count() != len(self.chunks):
            self._collection.upsert(
                ids=[str(c["chunk_id"]) for c in self.chunks],
                documents=[c["text"] for c in self.chunks],
            )

    def _bm25_ranking(self, query: str, k: int) -> list[int]:
        """Returns up to k chunk_ids ranked by BM25 score, best first."""
        scores = self._bm25.get_scores(_tokenize(query))
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.chunks[i]["chunk_id"] for i in ranked_indices[:k]]

    def _vector_ranking(self, query: str, k: int) -> list[int]:
        """Returns up to k chunk_ids ranked by embedding similarity, best
        first, via the Chroma collection's own nearest-neighbour search.
        """
        result = self._collection.query(query_texts=[query], n_results=k)
        return [int(chunk_id) for chunk_id in result["ids"][0]]

    def _fused_chunks(
        self, queries: list[str], k: int, fetch_k: int
    ) -> list[RetrievedChunk]:
        """BM25 and vector rankings for every query, fused by one RRF call,
        truncated to the top `k` chunks.
        """
        rankings = []
        for query in queries:
            rankings.append(self._bm25_ranking(query, fetch_k))
            rankings.append(self._vector_ranking(query, fetch_k))
        return [
            RetrievedChunk(
                chunk_id=chunk_id,
                text=self._chunks_by_id[chunk_id]["text"],
                page_no=self._chunks_by_id[chunk_id]["page_no"],
                heading=self._chunks_by_id[chunk_id]["heading"],
                chunk_type=self._chunks_by_id[chunk_id]["chunk_type"],
                rrf_score=score,
            )
            for chunk_id, score in reciprocal_rank_fusion(rankings)[:k]
        ]

    def retrieve(self, query: str, k: int = 5, fetch_k: int = 20) -> list[RetrievedChunk]:
        """Returns the top-k chunks for `query`, ranked by RRF over the
        BM25 and vector channels.

        `fetch_k` is how many candidates each channel contributes to the
        fusion before truncating to `k` -- wider than `k` itself so a
        chunk ranked just outside one channel's top-k but strong in the
        other channel still has a chance to surface.
        """
        return self._fused_chunks([query], k, fetch_k)

    def retrieve_with_queries(
        self, queries: list[str], k: int = 5, fetch_k: int = 20
    ) -> list[RetrievedChunk]:
        """Fusion over several phrasings of one question, supplied by the
        caller. The router agent writes the phrasings itself in its tool
        call, which gives the RAG Fusion benefit (see
        retrieve_with_query_fusion) without a separate LLM request.
        """
        return self._fused_chunks(queries, k, fetch_k)

    def retrieve_with_query_fusion(
        self,
        query: str,
        llm_client: GeminiClient,
        k: int = 5,
        fetch_k: int = 20,
        num_variants: int = 2,
    ) -> list[RetrievedChunk]:
        """RAG Fusion: asks the LLM for `num_variants` alternative
        phrasings of `query`, retrieves both BM25 and vector rankings for
        the original query AND every variant, and fuses all of those
        rankings together with one reciprocal_rank_fusion call, rather
        than retrieving once for the literal query alone.

        Motivated by a confirmed case (see docs/retrieval.md) where the
        literal query's vocabulary collided with a dominant but unrelated
        sense of the same words elsewhere in the corpus ("points" as
        army-building currency vs. victory points scored in a mission),
        causing retrieve() to miss the correct chunk even within its top
        10. A differently-worded variant can retrieve the right chunk
        even when the original phrasing can't -- this is the course's RAG
        Fusion pattern (course 3, slide 5), applied here specifically
        because we have a confirmed failure case it addresses, not
        speculatively.
        """
        prompt = QUERY_REWRITE_PROMPT.format(num_variants=num_variants, query=query)
        variants = parse_query_variants(llm_client.generate(prompt), num_variants)
        return self._fused_chunks([query, *variants], k, fetch_k)
