"""
Text-to-SQL retriever over wahadb.sqlite (unit datasheets), following the
basic pipeline from the course material: a static system prompt carrying
the data dictionary, the user's question appended, and the LLM asked to
write a SQL query -- rather than the dynamic few-shot or table-selection
variants, since this project's data dictionary (data/raw/
SQL_AGENT_DATA_DICTIONARY.md) already fits comfortably in a single prompt
(21 tables, ~600 lines) and dynamic retrieval of dictionary fragments is
the course's own fallback for dictionaries too large to fit at all.

Security: the course flags Text-to-SQL as the one component that can
delete or corrupt data (DROP/DELETE/SET) if given write access, and
recommends read-only access and propagating the user's own rights at
most. This project has no user-rights model (it's a single-user CLI/chat
tool), so the applicable half of that guidance is read-only access,
applied twice: the SQLite connection itself is opened in SQLite's own
read-only URI mode, and execute_sql additionally rejects anything that
isn't a single SELECT statement before it ever reaches the connection, in
case the LLM-generated text contains more than one statement or a
mutating keyword.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from project_rag.llm.cache import ResponseCache
from project_rag.llm.gemini_client import GeminiClient

DEFAULT_DB_PATH = Path("data/raw/wahadb.sqlite")
DEFAULT_DICTIONARY_PATH = Path("data/raw/SQL_AGENT_DATA_DICTIONARY.md")

_SQL_BLOCK_PATTERN = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_DISALLOWED_KEYWORDS = (
    "DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "ATTACH", "DETACH",
    "PRAGMA", "VACUUM", "REINDEX", "CREATE", "REPLACE",
)

_SYSTEM_PROMPT_TEMPLATE = """You are a SQL generation assistant for a Warhammer 40,000 rules database.
Given the data dictionary below and a user's question, write a single
read-only SQLite SELECT query that answers it. Output only the SQL query,
in a single ```sql code block, with no explanation before or after it.

{data_dictionary}
"""


class SQLGenerationError(RuntimeError):
    """Raised when the LLM's response doesn't contain exactly one safe
    SELECT statement."""


@dataclass
class SQLResult:
    question: str
    sql: str
    columns: list[str]
    rows: list[tuple]


def extract_sql(llm_response: str) -> str:
    """Pulls the SQL query out of an LLM response, which is expected to
    wrap it in a ```sql code block per the system prompt's instruction.
    Falls back to the whole response, stripped, if no code block is
    found -- some models omit the fence for a single-statement answer.
    """
    match = _SQL_BLOCK_PATTERN.search(llm_response)
    return match.group(1).strip() if match else llm_response.strip()


def check_sql_is_safe(sql: str) -> None:
    """Raises SQLGenerationError if `sql` is not a single, read-only
    SELECT statement. A statement may start with WITH (a common table
    expression), since the keyword filter below still rejects any
    mutating statement placed inside it.

    This is a second line of defense behind opening the SQLite connection
    itself in read-only mode (see SQLRetriever._connect_read_only): a
    mutating keyword should already fail at the connection level, but
    checking here gives a clearer error message and catches a
    multi-statement payload (e.g. "SELECT ...; DROP TABLE ...") before it
    is sent to sqlite3.execute at all. Keyword matching uses word
    boundaries so a benign identifier like `update_count` is not flagged
    just for containing the substring "update".
    """
    statements = [s.strip() for s in sql.split(";") if s.strip()]
    if len(statements) != 1:
        raise SQLGenerationError(
            f"Expected exactly one SQL statement, got {len(statements)}: {sql!r}"
        )
    statement = statements[0]
    if not statement.upper().startswith(("SELECT", "WITH")):
        raise SQLGenerationError(f"Only SELECT statements are allowed, got: {statement!r}")
    upper = statement.upper()
    for keyword in _DISALLOWED_KEYWORDS:
        if re.search(rf"\b{keyword}\b", upper):
            raise SQLGenerationError(
                f"Disallowed keyword '{keyword}' found in generated SQL: {statement!r}"
            )


class SQLRetriever:
    """Translates a natural-language question into SQL against
    wahadb.sqlite, using the data dictionary as the LLM's system prompt,
    and executes it read-only.
    """

    def __init__(
        self,
        db_path: Path | str = DEFAULT_DB_PATH,
        dictionary_path: Path | str = DEFAULT_DICTIONARY_PATH,
        llm_client: GeminiClient | None = None,
    ) -> None:
        self._db_path = Path(db_path)
        self._data_dictionary = Path(dictionary_path).read_text()
        self._llm = llm_client or GeminiClient(cache=ResponseCache())

    def _connect_read_only(self) -> sqlite3.Connection:
        """Opens the SQLite file through its read-only URI mode
        (mode=ro), which raises at the OS/SQLite level if anything tries
        to write -- a second, independent backstop to check_sql_is_safe,
        not a replacement for it.
        """
        uri = f"file:{self._db_path}?mode=ro"
        return sqlite3.connect(uri, uri=True)

    def generate_sql(self, question: str) -> str:
        """Asks the LLM to write a SQL query for `question`, given the
        data dictionary as system context, and returns the extracted SQL
        text (not yet validated or executed).
        """
        system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(data_dictionary=self._data_dictionary)
        response = self._llm.generate(prompt=question, system_instruction=system_prompt)
        return extract_sql(response)

    def execute_sql(self, sql: str) -> tuple[list[str], list[tuple]]:
        """Validates `sql` with check_sql_is_safe, then runs it against
        the read-only connection. Returns (column_names, rows).
        """
        check_sql_is_safe(sql)
        connection = self._connect_read_only()
        try:
            cursor = connection.execute(sql)
            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()
        finally:
            connection.close()
        return columns, rows

    def retrieve(self, question: str) -> SQLResult:
        """Generates SQL for `question`, executes it, and returns the
        full result bundle (question, SQL, columns, rows) for the caller
        to format into an answer or a chunk-like context.
        """
        sql = self.generate_sql(question)
        columns, rows = self.execute_sql(sql)
        return SQLResult(question=question, sql=sql, columns=columns, rows=rows)
