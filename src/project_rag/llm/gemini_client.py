"""
Thin wrapper around the Gemini API (Google AI Studio), the LLM backend the
project brief requires. Centralized here so both the text-to-SQL retriever
(retriever/sql_retriever.py), the router's query-rewriting step
(chatbot/router.py), and the eventual answer-generation step call through
one place rather than each holding their own client setup, model name,
and retry logic.

Requires a GEMINI_API_KEY environment variable, generated from an AI
Studio account per the course setup instructions.

Retries: the free AI Studio tier's rate limit is tight (observed: 5
requests/minute for gemini-3.8-flash) and this project now makes several
LLM calls per user question (router reasoning, query-rewriting for RAG
Fusion, SQL generation), so a 429 (rate limit) or 503 (temporary
overload) during normal use is expected, not exceptional, and is retried
automatically with exponential backoff rather than surfacing as a crash.
A non-retryable error (e.g. a bad model name, a 404) still raises
immediately on the first attempt.
"""

from __future__ import annotations

import logging
import os
import time

import httpx
from google import genai
from google.genai import errors, types

from project_rag.config import ensure_api_key

from project_rag.llm.cache import ResponseCache, make_key

DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
MAX_RETRIES = 7
REQUEST_TIMEOUT_SECONDS = 60
logger = logging.getLogger(__name__)
INITIAL_BACKOFF_SECONDS = 2.0
_RETRYABLE_STATUS_CODES = {429, 503}


def _status_code(error: Exception) -> int | None:
    """Best-effort extraction of an HTTP status code from a
    google.genai.errors.APIError. The SDK's exact attribute name isn't
    pinned to one spelling across versions, so this checks a few
    plausible places rather than assuming one -- if none match, the
    caller treats the error as non-retryable rather than guessing.
    """
    for attr in ("code", "status_code"):
        value = getattr(error, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(error, "response", None)
    return getattr(response, "status_code", None)


def _is_retryable(error: Exception) -> bool:
    """True for a dropped or failed connection (httpx.TransportError), or
    an API error with a retryable status code (429, 503). A daily-quota
    429 ("PerDay" in the message) is never retried: waiting seconds cannot
    fix it and each retry would only waste time.
    """
    if isinstance(error, httpx.TransportError):
        return True
    if "PerDay" in str(error):
        return False
    return _status_code(error) in _RETRYABLE_STATUS_CODES


class GeminiClient:
    """Wraps a single genai.Client and a chosen model name."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        cache: ResponseCache | None = None,
    ) -> None:
        resolved_key = api_key or ensure_api_key()
        self._client = genai.Client(
            api_key=resolved_key,
            http_options=types.HttpOptions(
                timeout=REQUEST_TIMEOUT_SECONDS * 1000,
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )
        self._model = model
        self._cache = cache
        self.request_count = 0
        self.cache_hits = 0

    def generate(self, prompt: str, system_instruction: str | None = None) -> str:
        """Sends `prompt` to the configured model and returns the plain
        text response. `system_instruction` is passed through as Gemini's
        dedicated system-instruction field rather than concatenated into
        the prompt, so it stays out of what counts as user-turn content.

        Retries on a rate-limit (429), temporary-overload (503)
        response or dropped connection, up to MAX_RETRIES times, with exponential backoff
        starting at INITIAL_BACKOFF_SECONDS. Any other error (including
        an unrecognized status code, per _status_code) is raised
        immediately on the first attempt.

        With a cache, a previously answered request is returned from disk
        and counted in cache_hits; request_count counts every attempt
        actually sent to the API, retries included.
        """
        cache_key = make_key(self._model, system_instruction, prompt)
        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                self.cache_hits += 1
                return cached

        config = {"system_instruction": system_instruction} if system_instruction else {}
        backoff_seconds = INITIAL_BACKOFF_SECONDS

        for attempt in range(MAX_RETRIES):
            try:
                self.request_count += 1
                started = time.monotonic()
                logger.info("gemini request %d (attempt %d)", self.request_count, attempt + 1)
                response = self._client.models.generate_content(
                    model=self._model, contents=prompt, config=config
                )
                logger.info("gemini request answered in %.1fs", time.monotonic() - started)
                if self._cache is not None:
                    self._cache.set(cache_key, response.text)
                return response.text
            except (errors.APIError, httpx.TransportError) as error:
                is_last_attempt = attempt == MAX_RETRIES - 1
                if not _is_retryable(error) or is_last_attempt:
                    raise
                logger.info("retryable error, waiting %.0fs: %.120s", backoff_seconds, error)
                time.sleep(backoff_seconds)
                backoff_seconds *= 2

        raise RuntimeError("unreachable: generate() retry loop exited without returning or raising")
