"""Prints the generated SQL and rows for sample unit questions (LLM requests).

Run with: uv run python scripts/check_sql_retriever.py
"""

from project_rag.retriever.sql_retriever import SQLRetriever

retriever = SQLRetriever()

questions = [
    "What is the toughness of an Intercessor Squad?",
    "List all stratagems that cost 1 CP for the Space Marines faction.",
]

for question in questions:
    print(f"\n{'=' * 80}\nQuestion: {question}\n{'=' * 80}")
    result = retriever.retrieve(question)
    print(f"Generated SQL:\n{result.sql}\n")
    print(f"Columns: {result.columns}")
    for row in result.rows[:10]:
        print(row)
