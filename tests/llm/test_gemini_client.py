"""
Unit tests for the pure status-code extraction in llm/gemini_client.py.
The retry loop itself (GeminiClient.generate) depends on the real
google-genai SDK's exception shape and the network, so it is exercised
through the diagnostic scripts under real API conditions instead.
"""

import httpx

from project_rag.llm.gemini_client import _is_retryable, _status_code


class _FakeErrorWithCodeAttr(Exception):
    def __init__(self, code):
        self.code = code


class _FakeErrorWithStatusCodeAttr(Exception):
    def __init__(self, status_code):
        self.status_code = status_code


class _FakeResponse:
    def __init__(self, status_code):
        self.status_code = status_code


class _FakeErrorWithResponseAttr(Exception):
    def __init__(self, status_code):
        self.response = _FakeResponse(status_code)


def test_status_code_from_code_attribute():
    assert _status_code(_FakeErrorWithCodeAttr(429)) == 429


def test_status_code_from_status_code_attribute():
    assert _status_code(_FakeErrorWithStatusCodeAttr(503)) == 503


def test_status_code_from_nested_response_attribute():
    assert _status_code(_FakeErrorWithResponseAttr(429)) == 429


def test_status_code_none_when_no_recognized_attribute():
    assert _status_code(RuntimeError("unrelated error")) is None


def test_dropped_connection_is_retryable():
    assert _is_retryable(httpx.RemoteProtocolError("Server disconnected"))


def test_503_is_retryable():
    assert _is_retryable(_FakeErrorWithCodeAttr(503))


def test_404_is_not_retryable():
    assert not _is_retryable(_FakeErrorWithCodeAttr(404))


class _FakeErrorWithMessage(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def test_per_minute_429_is_retryable():
    assert _is_retryable(_FakeErrorWithMessage(429, "PerMinutePerProjectPerModel"))


def test_daily_quota_429_is_not_retryable():
    assert not _is_retryable(_FakeErrorWithMessage(429, "GenerateRequestsPerDayPerProjectPerModel"))
