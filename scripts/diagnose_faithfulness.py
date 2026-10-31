"""Prints the claims the RAGAS judge extracts from an answer and its verdict
on each, to see why a Faithfulness score is low. Costs 2 judge requests.

Run with: uv run python scripts/diagnose_faithfulness.py q02
"""

import asyncio
import sys

from project_rag.evaluation.collect import RESULTS_PATH, load_results
from project_rag.evaluation.score import build_metrics


async def diagnose(record: dict, metric) -> None:
    row = {
        "user_input": record["question"],
        "response": record["answer"],
        "retrieved_contexts": record["contexts"],
    }
    statements = (await metric._create_statements(row, None)).statements
    verdicts = await metric._create_verdicts(row, statements, None)
    for item in verdicts.statements:
        print(f"[{'supported' if item.verdict else 'NOT supported'}] {item.statement}\n    reason: {item.reason}\n")
    supported = sum(1 for item in verdicts.statements if item.verdict)
    print(f"{supported} of {len(verdicts.statements)} statements supported")


def main() -> None:
    question_id = sys.argv[1]
    record = next(r for r in load_results(RESULTS_PATH) if r["id"] == question_id)
    asyncio.run(diagnose(record, build_metrics(["faithfulness"])["faithfulness"]))


if __name__ == "__main__":
    main()
