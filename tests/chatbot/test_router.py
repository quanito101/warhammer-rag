"""
Unit tests for the pure formatting functions in chatbot/router.py.
build_router_agent itself (which loads an embedding model, opens the
SQLite database, and calls the Gemini API) is an integration concern,
exercised by scripts/check_router.py.
"""

from project_rag.chatbot.router import format_sql_result, format_text_results
from project_rag.retriever.sql_retriever import SQLResult
from project_rag.retriever.text_retriever import RetrievedChunk


def test_format_text_results_includes_page_and_type():
    chunk = RetrievedChunk(
        chunk_id=1,
        text="A unit scores points by holding an objective marker.",
        page_no=12,
        heading=None,
        chunk_type="prose",
        rrf_score=0.5,
    )
    formatted = format_text_results([chunk])
    assert "page 12" in formatted
    assert "prose" in formatted
    assert "holding an objective marker" in formatted


def test_format_text_results_empty_list():
    assert format_text_results([]) == "No matching rules text found."


def test_format_sql_result_includes_sql_and_rows():
    result = SQLResult(
        question="What is the toughness of an Intercessor Squad?",
        sql="SELECT t FROM datasheets_models WHERE DatasheetId = '000001157'",
        columns=["t"],
        rows=[("4",)],
    )
    formatted = format_sql_result(result)
    assert "SELECT t FROM datasheets_models" in formatted
    assert "('4',)" in formatted


def test_format_sql_result_no_rows():
    result = SQLResult(question="...", sql="SELECT 1 WHERE 0", columns=["1"], rows=[])
    formatted = format_sql_result(result)
    assert "No rows returned" in formatted
