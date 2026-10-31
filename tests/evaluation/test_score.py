from project_rag.evaluation.score import (
    is_number,
    pending_tasks,
    record_score,
    score_with_retry,
    summarize,
    to_ragas_row,
)

RECORD = {"id": "q01", "question": "Q?", "answer": "A", "contexts": ["c1", "c2"], "ground_truth": "G"}


def test_to_ragas_row_wraps_each_field_in_a_list():
    assert to_ragas_row(RECORD) == {
        "question": ["Q?"],
        "answer": ["A"],
        "contexts": [["c1", "c2"]],
        "ground_truth": ["G"],
    }


def test_pending_tasks_skip_scored_metrics_and_respect_limit():
    results = [{"id": "q01"}, {"id": "q02"}, {"id": "q03"}]
    scores = [{"id": "q01", "faithfulness": 1.0, "context_recall": 0.5}, {"id": "q02", "faithfulness": 0.2}]
    tasks = pending_tasks(results, scores, ["faithfulness", "context_recall"])
    assert [(r["id"], m) for r, m in tasks] == [
        ("q02", "context_recall"),
        ("q03", "faithfulness"),
        ("q03", "context_recall"),
    ]
    limited = pending_tasks(results, scores, ["faithfulness", "context_recall"], limit=1)
    assert [(r["id"], m) for r, m in limited] == [("q02", "context_recall")]


def test_pending_tasks_retry_nan_scores():
    scores = [{"id": "q01", "faithfulness": float("nan")}]
    assert [m for _, m in pending_tasks([{"id": "q01"}], scores, ["faithfulness"])] == ["faithfulness"]


def test_record_score_creates_and_updates_entries():
    scores = []
    record_score(scores, "q01", "faithfulness", 0.5)
    record_score(scores, "q01", "context_recall", 1.0)
    assert scores == [{"id": "q01", "faithfulness": 0.5, "context_recall": 1.0}]


def test_summarize_ignores_missing_and_nan_values():
    scores = [
        {"id": "q01", "faithfulness": 1.0, "context_recall": float("nan")},
        {"id": "q02", "faithfulness": 0.5, "context_recall": 1.0},
        {"id": "q03", "faithfulness": None},
    ]
    assert summarize(scores, ["faithfulness", "context_recall"]) == {"faithfulness": 0.75, "context_recall": 1.0}


def test_is_number_rejects_nan_and_none():
    assert is_number(0.0)
    assert not is_number(float("nan"))
    assert not is_number(None)


def test_score_with_retry_waits_on_per_minute_limit_then_succeeds():
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("429 RESOURCE_EXHAUSTED PerMinute")
        return {"id": "q01"}

    assert score_with_retry(flaky, wait_seconds=0) == {"id": "q01"}
    assert len(calls) == 3


def test_score_with_retry_raises_daily_quota_immediately():
    calls = []

    def daily():
        calls.append(1)
        raise RuntimeError("429 GenerateRequestsPerDayPerProjectPerModel")

    try:
        score_with_retry(daily, wait_seconds=0)
    except RuntimeError:
        pass
    assert len(calls) == 1
