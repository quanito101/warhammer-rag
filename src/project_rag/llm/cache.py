"""On-disk cache of LLM responses, keyed by the exact request.

The free Gemini tier allows 20 requests per day, so a request already
answered once (same model, system instruction and prompt) is read from
a local SQLite file instead of being sent again.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path

DEFAULT_CACHE_PATH = Path("data/processed/llm_cache.sqlite")


def make_key(model: str, system_instruction: str | None, prompt: str) -> str:
    """SHA-256 of the three inputs that determine a response."""
    payload = json.dumps([model, system_instruction, prompt], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ResponseCache:
    """Key-to-response store backed by one SQLite file. A lock guards the
    connection because LangGraph runs tool calls in worker threads.
    """

    def __init__(self, path: Path | str = DEFAULT_CACHE_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, response TEXT NOT NULL)"
        )
        self._connection.commit()

    def get(self, key: str) -> str | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT response FROM responses WHERE key = ?", (key,)
            ).fetchone()
        return row[0] if row else None

    def set(self, key: str, response: str) -> None:
        with self._lock:
            self._connection.execute(
                "INSERT OR REPLACE INTO responses (key, response) VALUES (?, ?)",
                (key, response),
            )
            self._connection.commit()
