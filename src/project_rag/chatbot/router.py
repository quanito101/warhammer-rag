"""
Routes a user's question to the appropriate retriever (text or SQL) and
answers from whichever one(s) it calls, using LangGraph's prebuilt ReAct
agent: the two retrievers are exposed as tools, and the LLM decides which
to call based on the question -- the "Router agent" pattern from course 3
(slide 26), implemented via the ReAct loop (course 3, slide 28) rather
than a hand-written if/else dispatcher, since LangGraph's
create_react_agent already gives both for the cost of defining two tool
functions.

Why LangGraph specifically (not Google ADK, not a plain Python dispatcher):
Florent's course teaches the router/ReAct/planner-executor patterns as
framework-agnostic concepts and never names ADK. LangGraph is both the
course's own example library (course 1, "turnkey solutions") and the
framework explicitly named in this project's internship-requirement
notes, so it is the one choice that satisfies the brief, matches what was
taught, and carries over outside this course.
"""

from __future__ import annotations

from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from project_rag.config import ensure_api_key
from project_rag.llm.gemini_client import DEFAULT_MODEL
from project_rag.retriever.sql_retriever import SQLGenerationError, SQLResult, SQLRetriever
from project_rag.retriever.text_retriever import RetrievedChunk, TextRetriever

REQUEST_TIMEOUT_SECONDS = 60
RULES_TOP_K = 8

SYSTEM_PROMPT = """You are a Warhammer 40,000 rules assistant with two tools:
retrieve_rules_text answers questions about core game rules, stratagems,
scoring, and procedures, from the rulebook. retrieve_unit_data answers
questions about specific unit stats (toughness, wounds, weapons, points
cost), from the unit datasheet database. Pick whichever tool fits the
question -- or both, if the question needs rules context and specific
unit numbers together. Always cite what you used (page number for rules
text, or that the figure came from the unit database) in your answer.

Search economically: call retrieve_rules_text once per question with 2 or
3 differently worded queries, and call it a second time only if the first
result is clearly unrelated to the question. Never search more than twice.

When a unit name is generic and matches many datasheets (for example "a
space marine"), do not pick one row. Ask retrieve_unit_data for a breakdown
of the value with example unit names, then answer with the most common
case, name the units that differ, and say which assumption you made.

Do not name an edition of the game: the sources do not state one.

Answer only from what the tools return. The rulebook you can search is the
Core Rules only: it does not contain mission packs, secondary objective
cards, or Crusade rules. Never add rules, names or numbers from your own
knowledge of Warhammer 40,000. If the retrieved text does not cover part
of the question, say so plainly instead of filling the gap.
"""


def format_text_results(results: list[RetrievedChunk]) -> str:
    """Formats TextRetriever results into the plain-text block the router
    LLM sees as a tool result. Kept as a standalone function so the
    formatting itself is unit-testable without building a full agent or
    loading an embedding model.
    """
    if not results:
        return "No matching rules text found."
    return "\n\n".join(
        f"[page {result.page_no}, {result.chunk_type}] {result.text}" for result in results
    )


def format_sql_result(result: SQLResult) -> str:
    """Formats a SQLResult into the plain-text block the router LLM sees
    as a tool result. Kept standalone for the same testability reason as
    format_text_results.
    """
    if not result.rows:
        return f"SQL: {result.sql}\nNo rows returned."
    rows_text = "\n".join(str(row) for row in result.rows)
    return f"SQL: {result.sql}\nColumns: {result.columns}\nRows:\n{rows_text}"


def build_router_agent(
    text_retriever: TextRetriever | None = None,
    sql_retriever: SQLRetriever | None = None,
    model: str = DEFAULT_MODEL,
):
    """Builds and returns a LangGraph ReAct agent wired to both
    retrievers as tools. Retrievers are created lazily (TextRetriever
    loads an embedding model and a persisted Chroma index; SQLRetriever
    needs a Gemini API key) only if not supplied, so tests can inject
    fakes instead of paying that cost.
    """
    ensure_api_key()
    text_retriever = text_retriever or TextRetriever()
    sql_retriever = sql_retriever or SQLRetriever()

    @tool
    def retrieve_rules_text(queries: list[str]) -> str:
        """Searches the Core Rules (stratagems, scoring, procedures, core
        mechanics). Pass 2 or 3 differently worded search queries for the
        same question; spell out ambiguous terms (for example "victory
        points scored in a mission" as well as "points")."""
        return format_text_results(
            text_retriever.retrieve_with_queries(queries, k=RULES_TOP_K)
        )

    @tool
    def retrieve_unit_data(question: str) -> str:
        """Retrieves unit datasheet data (stats, weapons, points cost) by
        generating and running a SQL query against the unit database.
        Faction names must match exactly: "Space Marines" and "Chaos Space
        Marines" are different factions."""
        try:
            return format_sql_result(sql_retriever.retrieve(question))
        except SQLGenerationError as error:
            return f"The generated SQL was rejected: {error}"

    llm = ChatGoogleGenerativeAI(
        model=model, timeout=REQUEST_TIMEOUT_SECONDS, max_retries=1
    )
    return create_react_agent(
        llm, tools=[retrieve_rules_text, retrieve_unit_data], prompt=SYSTEM_PROMPT
    )
