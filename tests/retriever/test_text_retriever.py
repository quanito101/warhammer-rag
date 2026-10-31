"""
Unit tests for the RRF combination logic in retriever/text_retriever.py.

Only reciprocal_rank_fusion is tested here, not TextRetriever itself:
TextRetriever needs the real chunk file and downloads an embedding model
on first use, which makes it an integration concern
(scripts/check_text_retriever.py) rather than a fast, dependency-free
unit test.
"""

from project_rag.retriever.text_retriever import TextRetriever, chunks_fingerprint, reciprocal_rank_fusion


def test_agreement_across_channels_ranks_highest():
    bm25_ranking = [10, 20, 30]
    vector_ranking = [20, 10, 40]
    fused = reciprocal_rank_fusion([bm25_ranking, vector_ranking])
    assert fused[0][0] in {10, 20}


def test_chunk_only_in_one_channel_still_included():
    bm25_ranking = [1, 2, 3]
    vector_ranking = [4, 5, 6]
    fused = reciprocal_rank_fusion([bm25_ranking, vector_ranking])
    fused_ids = {chunk_id for chunk_id, _ in fused}
    assert fused_ids == {1, 2, 3, 4, 5, 6}


def test_top_rank_in_both_channels_outscores_top_rank_in_one():
    agrees_on_top = reciprocal_rank_fusion([[1, 2], [1, 3]])
    only_one_channel = reciprocal_rank_fusion([[1, 2], [9, 3]])
    score_when_both_agree = dict(agrees_on_top)[1]
    score_when_only_one = dict(only_one_channel)[1]
    assert score_when_both_agree > score_when_only_one


def test_empty_rankings_return_empty_result():
    assert reciprocal_rank_fusion([[], []]) == []


from project_rag.retriever.text_retriever import parse_query_variants


def test_parse_query_variants_splits_on_lines():
    response = "How do I earn victory points?\nWhat are the ways to control objectives?"
    assert parse_query_variants(response, 2) == [
        "How do I earn victory points?",
        "What are the ways to control objectives?",
    ]


def test_parse_query_variants_drops_blank_lines():
    response = "\nFirst variant\n\nSecond variant\n\n"
    assert parse_query_variants(response, 2) == ["First variant", "Second variant"]


def test_parse_query_variants_truncates_to_expected_count():
    response = "one\ntwo\nthree\nfour"
    assert parse_query_variants(response, 2) == ["one", "two"]


class _StubRetriever(TextRetriever):
    def __init__(self, bm25_by_query, vector_by_query):
        self._chunks_by_id = {
            i: {"text": f"chunk {i}", "page_no": i, "heading": None, "chunk_type": "prose"}
            for i in range(1, 8)
        }
        self._bm25_by_query = bm25_by_query
        self._vector_by_query = vector_by_query

    def _bm25_ranking(self, query, k):
        return self._bm25_by_query[query]

    def _vector_ranking(self, query, k):
        return self._vector_by_query[query]


def test_retrieve_with_queries_finds_a_chunk_only_one_phrasing_reaches():
    retriever = _StubRetriever(
        {"points": [1, 2, 3], "victory points": [5, 6, 7]},
        {"points": [1, 2, 4], "victory points": [5, 6, 3]},
    )
    results = retriever.retrieve_with_queries(["points", "victory points"], k=3)
    assert 5 in {chunk.chunk_id for chunk in results}


def test_retrieve_with_queries_truncates_to_k():
    retriever = _StubRetriever({"a": [1, 2, 3, 4]}, {"a": [4, 3, 2, 1]})
    assert len(retriever.retrieve_with_queries(["a"], k=2)) == 2


def test_retrieve_matches_single_query_fusion():
    retriever = _StubRetriever({"a": [1, 2, 3]}, {"a": [3, 2, 1]})
    single = [c.chunk_id for c in retriever.retrieve("a", k=3)]
    multi = [c.chunk_id for c in retriever.retrieve_with_queries(["a"], k=3)]
    assert single == multi


def test_fingerprint_changes_when_any_chunk_text_changes():
    base = [{"chunk_id": 1, "text": "a"}, {"chunk_id": 2, "text": "b"}]
    changed = [{"chunk_id": 1, "text": "a"}, {"chunk_id": 2, "text": "b2"}]
    assert chunks_fingerprint(base) == chunks_fingerprint(list(base))
    assert chunks_fingerprint(base) != chunks_fingerprint(changed)
