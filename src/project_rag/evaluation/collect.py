"""Runs the router agent over the evaluation questions and stores, for each
one, the answer and the contexts the agent retrieved. Scoring is a separate
step, so a run can be spread over several days of free-tier quota.

Run with: uv run python -m project_rag.evaluation.collect [--limit N] [--ids q01 q02 [--redo]]
Results are written after every question to data/eval/results.json, and
questions already in that file are skipped on the next run.
"""

import argparse
import json
import logging
import re
from pathlib import Path

from project_rag.chatbot.chat import configure_logging, invoke_with_retry, message_text
from project_rag.chatbot.router import build_router_agent

QUESTIONS_PATH = Path("data/eval/questions.json")
RESULTS_PATH = Path("data/eval/results.json")
PASSAGE_SPLIT = re.compile(r"\n\n(?=\[page \d+, \w+\])")
DAILY_QUOTA_MARKER = "PerDay"
logger = logging.getLogger(__name__)


def split_passages(tool_output: str) -> list[str]:
    """Splits the text tool's output into one string per retrieved passage."""
    return PASSAGE_SPLIT.split(tool_output)


def extract_contexts(messages: list) -> list[str]:
    """Collects the retrieved contexts from an agent run: one entry per
    rulebook passage and one per SQL result.
    """
    contexts: list[str] = []
    for message in messages:
        if getattr(message, "type", None) != "tool":
            continue
        output = message_text(message.content)
        if getattr(message, "name", None) == "retrieve_rules_text":
            contexts.extend(split_passages(output))
        else:
            contexts.append(output)
    return contexts


def extract_tool_calls(messages: list) -> list[dict]:
    """Lists the tool calls the agent made, in order."""
    return [
        {"name": call["name"], "args": call["args"]}
        for message in messages
        for call in getattr(message, "tool_calls", None) or []
    ]


def build_record(item: dict, messages: list) -> dict:
    """One result record: the question data plus the agent's answer,
    retrieved contexts and tool calls.
    """
    return {
        "id": item["id"],
        "type": item["type"],
        "question": item["question"],
        "ground_truth": item["ground_truth"],
        "answer": message_text(messages[-1].content),
        "contexts": extract_contexts(messages),
        "tool_calls": extract_tool_calls(messages),
    }


def load_results(path: Path) -> list[dict]:
    """Reads stored results, or an empty list if none exist yet."""
    return json.loads(path.read_text()) if path.exists() else []


def save_results(path: Path, results: list[dict]) -> None:
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False))


def pending_questions(questions: list[dict], results: list[dict], ids: list[str] | None) -> list[dict]:
    """Questions still to run: not in `results`, and in `ids` when given."""
    done = {record["id"] for record in results}
    return [q for q in questions if q["id"] not in done and (not ids or q["id"] in ids)]


def collect(agent, questions: list[dict], results_path: Path, limit: int | None = None) -> list[dict]:
    """Runs the agent on each pending question and saves after each one.
    Stops cleanly when the daily quota is exhausted so finished work is kept.
    """
    results = load_results(results_path)
    todo = pending_questions(questions, results, None)
    for item in todo[:limit]:
        logger.info("running %s", item["id"])
        try:
            outcome = invoke_with_retry(agent, {"messages": [{"role": "user", "content": item["question"]}]})
        except Exception as error:
            if DAILY_QUOTA_MARKER in str(error):
                logger.warning("daily quota reached, stopping after %d results", len(results))
                break
            raise
        results.append(build_record(item, outcome["messages"]))
        save_results(results_path, results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="maximum number of questions to run now")
    parser.add_argument("--ids", nargs="*", default=None, help="only these question ids")
    parser.add_argument("--redo", action="store_true", help="discard stored results for --ids and run them again")
    args = parser.parse_args()
    configure_logging()
    questions = json.loads(QUESTIONS_PATH.read_text())
    if args.ids:
        questions = [q for q in questions if q["id"] in args.ids]
    if args.redo and args.ids:
        kept = [r for r in load_results(RESULTS_PATH) if r["id"] not in args.ids]
        save_results(RESULTS_PATH, kept)
    results = collect(build_router_agent(), questions, RESULTS_PATH, args.limit)
    print(f"{len(results)} of {len(questions)} questions have results in {RESULTS_PATH}")


if __name__ == "__main__":
    main()
