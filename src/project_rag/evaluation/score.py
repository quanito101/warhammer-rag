"""Scores the stored evaluation results with RAGAS, using Gemini as the
judge. Run after project_rag.evaluation.collect.

Run with: uv run python -m project_rag.evaluation.score [--metrics faithfulness context_recall]
Each metric of each question is written to data/eval/scores.json as soon as
it is scored, and anything already scored is skipped on the next run.

The judge talks to Gemini through its OpenAI-compatible endpoint, the
workaround RAGAS documents for the safety-settings error of the google-genai
SDK. Answer Relevancy is not offered because it needs embeddings.
"""

import argparse
import json
import logging
import time
from pathlib import Path

from project_rag.chatbot.chat import is_transient_error
from project_rag.config import ensure_api_key
from project_rag.evaluation.collect import DAILY_QUOTA_MARKER, RESULTS_PATH, load_results, save_results
from project_rag.llm.gemini_client import DEFAULT_MODEL

SCORES_PATH = Path("data/eval/scores.json")
GEMINI_OPENAI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
JUDGE_MAX_TOKENS = 8192
SCORE_ATTEMPTS = 4
RATE_LIMIT_WAIT_SECONDS = 65.0
DEFAULT_METRICS = ["faithfulness", "context_recall"]
METRIC_CLASSES = {
    "faithfulness": "Faithfulness",
    "context_recall": "ContextRecall",
    "context_precision": "ContextPrecision",
}
logger = logging.getLogger(__name__)


def to_ragas_row(record: dict) -> dict:
    """Maps a stored result to the field names RAGAS expects."""
    return {
        "question": [record["question"]],
        "answer": [record["answer"]],
        "contexts": [record["contexts"]],
        "ground_truth": [record["ground_truth"]],
    }


def is_number(value) -> bool:
    """True for a real score; RAGAS reports a failed judge call as nan."""
    return isinstance(value, (int, float)) and value == value


def pending_tasks(results: list[dict], scores: list[dict], metrics: list[str], limit: int | None = None) -> list[tuple[dict, str]]:
    """(record, metric) pairs that still need a score, for the first `limit`
    questions that have any. A metric already scored is never repeated.
    """
    done = {entry["id"]: entry for entry in scores}
    tasks: list[tuple[dict, str]] = []
    questions = 0
    for record in results:
        missing = [m for m in metrics if not is_number(done.get(record["id"], {}).get(m))]
        if not missing:
            continue
        if limit is not None and questions == limit:
            break
        questions += 1
        tasks.extend((record, metric) for metric in missing)
    return tasks


def record_score(scores: list[dict], question_id: str, metric: str, value: float) -> None:
    """Stores `value` in the entry for `question_id`, creating it if needed."""
    for entry in scores:
        if entry["id"] == question_id:
            entry[metric] = value
            return
    scores.append({"id": question_id, metric: value})


def summarize(scores: list[dict], metrics: list[str]) -> dict[str, float]:
    """Mean of each metric over the questions where it is a number."""
    means = {}
    for metric in metrics:
        values = [s[metric] for s in scores if isinstance(s.get(metric), (int, float)) and s[metric] == s[metric]]
        if values:
            means[metric] = sum(values) / len(values)
    return means


def build_metrics(names: list[str]) -> dict:
    """Builds the RAGAS metric objects, by name, with a Gemini judge."""
    from project_rag.evaluation.ragas_compat import patch_vertexai_import

    patch_vertexai_import()
    from openai import OpenAI
    from ragas import metrics as ragas_metrics
    from ragas.llms import llm_factory

    client = OpenAI(api_key=ensure_api_key(), base_url=GEMINI_OPENAI_URL, max_retries=1)
    judge = llm_factory(
        DEFAULT_MODEL, provider="openai", client=client, max_tokens=JUDGE_MAX_TOKENS
    )
    return {name: getattr(ragas_metrics, METRIC_CLASSES[name])(llm=judge) for name in names}


def score_metric(record: dict, metric_object, name: str) -> float | None:
    """Scores one stored result on one metric."""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.run_config import RunConfig

    result = evaluate(
        Dataset.from_dict(to_ragas_row(record)),
        metrics=[metric_object],
        run_config=RunConfig(max_workers=1),
        raise_exceptions=True,
    )
    return result.scores[0].get(name)


def score_with_retry(score_fn, wait_seconds: float = RATE_LIMIT_WAIT_SECONDS, attempts: int = SCORE_ATTEMPTS):
    """Calls `score_fn`, waiting and retrying on a per-minute rate limit or
    a temporary overload. A daily-quota error is raised at once.
    """
    for attempt in range(attempts):
        try:
            return score_fn()
        except Exception as error:
            if not is_transient_error(error) or attempt == attempts - 1:
                raise
            logger.info("rate limited or overloaded, waiting %.0f s", wait_seconds)
            time.sleep(wait_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS, choices=sorted(METRIC_CLASSES))
    parser.add_argument("--limit", type=int, default=None, help="maximum number of questions to work on now")
    parser.add_argument("--redo", nargs="*", default=[], help="discard stored scores for these question ids and score them again")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    scores = [entry for entry in load_results(SCORES_PATH) if entry["id"] not in args.redo]
    tasks = pending_tasks(load_results(RESULTS_PATH), scores, args.metrics, args.limit)
    metric_objects = build_metrics(args.metrics)
    for record, name in tasks:
        logger.info("scoring %s: %s", record["id"], name)
        try:
            value = score_with_retry(lambda: score_metric(record, metric_objects[name], name))
        except Exception as error:
            if DAILY_QUOTA_MARKER in str(error):
                logger.warning("daily quota reached, stopping")
                break
            raise
        if not is_number(value):
            logger.warning("no score for %s %s, not saved", record["id"], name)
            break
        record_score(scores, record["id"], name, value)
        save_results(SCORES_PATH, scores)
    print(json.dumps(summarize(scores, args.metrics), indent=2))


if __name__ == "__main__":
    main()
