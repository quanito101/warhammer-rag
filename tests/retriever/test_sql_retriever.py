"""
Unit tests for the pure SQL-safety logic in retriever/sql_retriever.py.
SQLRetriever itself (which calls the Gemini API and the real database) is
an integration concern, exercised by scripts/check_sql_retriever.py.
"""

import pytest

from project_rag.retriever.sql_retriever import (
    SQLGenerationError,
    check_sql_is_safe,
    extract_sql,
)


def test_extract_sql_from_fenced_block():
    response = "Here is the query:\n```sql\nSELECT * FROM datasheets;\n```"
    assert extract_sql(response) == "SELECT * FROM datasheets;"


def test_extract_sql_without_fence_falls_back_to_full_response():
    response = "SELECT * FROM datasheets;"
    assert extract_sql(response) == "SELECT * FROM datasheets;"


def test_safe_select_passes():
    check_sql_is_safe("SELECT name FROM datasheets WHERE factionId = 'SM'")


def test_multiple_statements_rejected():
    with pytest.raises(SQLGenerationError):
        check_sql_is_safe("SELECT * FROM datasheets; DROP TABLE datasheets;")


def test_non_select_rejected():
    with pytest.raises(SQLGenerationError):
        check_sql_is_safe("DELETE FROM datasheets")


def test_disallowed_keyword_rejected():
    with pytest.raises(SQLGenerationError):
        check_sql_is_safe("SELECT * FROM datasheets WHERE 1=1; UPDATE datasheets SET name='x'")


def test_identifier_containing_keyword_substring_not_flagged():
    check_sql_is_safe("SELECT update_count FROM datasheets_models_cost")


def test_with_clause_select_passes():
    check_sql_is_safe("WITH t AS (SELECT 1 AS x) SELECT x FROM t")


def test_with_clause_hiding_a_delete_rejected():
    with pytest.raises(SQLGenerationError):
        check_sql_is_safe("WITH t AS (SELECT 1) DELETE FROM datasheets")
