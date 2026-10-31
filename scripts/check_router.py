"""Runs sample questions through the router agent and prints its tool calls
and answers (LLM requests). Pass question numbers (1 to 5) to run a subset.

Run with: uv run python scripts/check_router.py [1 2 ...]
"""

import sys

from project_rag.chatbot.chat import configure_logging, invoke_with_retry, message_text
from project_rag.chatbot.router import build_router_agent

QUESTIONS = [
    "What is the rule for grenades?",
    "How can I use a stratagem?",
    "What are the different ways to score points?",
    "What is the toughness of an Intercessor Squad?",
    "What is the toughness of a space marine?",
]

configure_logging()
selected = [int(arg) for arg in sys.argv[1:]] or range(1, len(QUESTIONS) + 1)
agent = build_router_agent()

for number in selected:
    question = QUESTIONS[number - 1]
    print(f"\n{'=' * 80}\nQuestion {number}: {question}\n{'=' * 80}")
    result = invoke_with_retry(agent, {"messages": [{"role": "user", "content": question}]})
    for message in result["messages"]:
        for call in getattr(message, "tool_calls", None) or []:
            print(f"[tool call] {call['name']}: {call['args']}")
    print(message_text(result["messages"][-1].content))
