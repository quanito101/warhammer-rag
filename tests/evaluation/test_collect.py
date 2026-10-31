import json
from types import SimpleNamespace

from project_rag.evaluation.collect import (
    build_record,
    collect,
    extract_contexts,
    extract_tool_calls,
    pending_questions,
    split_passages,
)

TEXT_OUTPUT = "[page 21, table] Strength table\n\nline two\n\n[page 11, prose] Command points rule"
ITEM = {"id": "q01", "type": "rules_table", "question": "Q?", "ground_truth": "A."}


def tool_message(name, content):
    return SimpleNamespace(type="tool", name=name, content=content)


def ai_message(content, tool_calls=None):
    return SimpleNamespace(type="ai", content=content, tool_calls=tool_calls or [])


def test_split_passages_keeps_inner_blank_lines():
    assert split_passages(TEXT_OUTPUT) == [
        "[page 21, table] Strength table\n\nline two",
        "[page 11, prose] Command points rule",
    ]


def test_extract_contexts_handles_both_tools():
    messages = [
        ai_message("thinking"),
        tool_message("retrieve_rules_text", TEXT_OUTPUT),
        tool_message("retrieve_unit_data", "SQL: SELECT 1\nRows:\n(4,)"),
        ai_message("answer"),
    ]
    contexts = extract_contexts(messages)
    assert len(contexts) == 3
    assert contexts[2].startswith("SQL:")


def test_extract_tool_calls():
    messages = [ai_message("", [{"name": "retrieve_rules_text", "args": {"queries": ["a"]}, "id": "1"}])]
    assert extract_tool_calls(messages) == [{"name": "retrieve_rules_text", "args": {"queries": ["a"]}}]


def test_build_record_uses_last_message_as_answer():
    messages = [tool_message("retrieve_rules_text", TEXT_OUTPUT), ai_message("Final answer")]
    record = build_record(ITEM, messages)
    assert record["answer"] == "Final answer"
    assert record["id"] == "q01"
    assert len(record["contexts"]) == 2


def test_pending_questions_skips_done_and_filters_ids():
    questions = [{"id": "q01"}, {"id": "q02"}, {"id": "q03"}]
    results = [{"id": "q01"}]
    assert [q["id"] for q in pending_questions(questions, results, None)] == ["q02", "q03"]
    assert [q["id"] for q in pending_questions(questions, results, ["q03"])] == ["q03"]


class FakeAgent:
    def __init__(self, fail_on=None):
        self.calls = 0
        self.fail_on = fail_on

    def invoke(self, payload, config=None):
        self.calls += 1
        if self.calls == self.fail_on:
            raise RuntimeError("429 RESOURCE_EXHAUSTED GenerateRequestsPerDay")
        return {"messages": [ai_message("answer " + payload["messages"][0]["content"])]}


def questions(n):
    return [{**ITEM, "id": f"q{i:02d}", "question": f"Q{i}"} for i in range(1, n + 1)]


def test_collect_saves_and_resumes(tmp_path):
    path = tmp_path / "results.json"
    collect(FakeAgent(), questions(3), path, limit=2)
    assert len(json.loads(path.read_text())) == 2
    agent = FakeAgent()
    collect(agent, questions(3), path)
    assert agent.calls == 1
    assert len(json.loads(path.read_text())) == 3


def test_collect_stops_on_daily_quota_and_keeps_results(tmp_path):
    path = tmp_path / "results.json"
    results = collect(FakeAgent(fail_on=2), questions(3), path)
    assert len(results) == 1
    assert len(json.loads(path.read_text())) == 1
