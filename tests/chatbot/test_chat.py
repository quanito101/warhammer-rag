from types import SimpleNamespace

import pytest

from project_rag.chatbot import chat
from project_rag.chatbot.chat import ask, invoke_with_retry, is_transient_error, message_text


class FakeAgent:
    def invoke(self, payload, config=None):
        reply = SimpleNamespace(content=[
            {"type": "text", "text": "Answer."},
            {"type": "text", "text": " More.", "extras": {"signature": "xyz"}},
        ])
        return {"messages": [*payload["messages"], reply]}


def test_message_text_accepts_a_plain_string():
    assert message_text("hello") == "hello"


def test_message_text_keeps_only_text_blocks():
    content = [{"type": "text", "text": "a"}, {"type": "other", "text": "b"}]
    assert message_text(content) == "a"


def test_ask_returns_answer_and_extended_history():
    answer, history = ask(FakeAgent(), [], "What is a stratagem?")
    assert answer == "Answer. More."
    assert history[0] == {"role": "user", "content": "What is a stratagem?"}
    assert len(history) == 2


def test_ask_passes_previous_history_to_the_agent():
    _, first = ask(FakeAgent(), [], "one")
    _, second = ask(FakeAgent(), first, "two")
    assert len(second) == 4


def test_503_is_transient():
    assert is_transient_error(Exception("503 UNAVAILABLE high demand"))


def test_per_minute_429_is_transient():
    assert is_transient_error(Exception("429 RESOURCE_EXHAUSTED PerMinute"))


def test_daily_quota_429_is_not_transient():
    assert not is_transient_error(Exception("429 GenerateRequestsPerDayPerProjectPerModel"))


class FlakyAgent:
    def __init__(self, failures, error):
        self.calls = 0
        self.failures = failures
        self.error = error

    def invoke(self, payload, config=None):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error
        return {"messages": []}


def test_invoke_with_retry_recovers_after_a_transient_error(monkeypatch):
    monkeypatch.setattr(chat.time, "sleep", lambda seconds: None)
    agent = FlakyAgent(2, Exception("503 UNAVAILABLE"))
    assert invoke_with_retry(agent, {}) == {"messages": []}
    assert agent.calls == 3


def test_invoke_with_retry_raises_immediately_on_other_errors(monkeypatch):
    monkeypatch.setattr(chat.time, "sleep", lambda seconds: None)
    agent = FlakyAgent(5, ValueError("bad input"))
    with pytest.raises(ValueError):
        invoke_with_retry(agent, {})
    assert agent.calls == 1


def test_dropped_connection_is_transient():
    assert is_transient_error(Exception("RemoteProtocolError: Server disconnected without sending a response."))


def test_504_deadline_exceeded_is_transient():
    assert is_transient_error(Exception("504 DEADLINE_EXCEEDED"))
