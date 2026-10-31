"""Command-line chat loop around the router agent.

Run with: uv run python -m project_rag.chatbot.chat
The conversation history is kept so follow-up questions work.
"""

import logging
import time

from langchain_core.callbacks import BaseCallbackHandler

from project_rag.chatbot.router import build_router_agent
from project_rag.config import ensure_api_key

INVOKE_ATTEMPTS = 4
RECURSION_LIMIT = 10
INVOKE_BACKOFF_SECONDS = 15.0
TRANSIENT_MARKERS = (
    "503",
    "Server disconnected",
    "RemoteProtocolError",
    "ConnectError",
    "Timeout",
    "timed out",
    "504",
    "DEADLINE_EXCEEDED",
)
EXIT_COMMANDS = {"exit", "quit", ":q"}
PROMPT = "You> "
logger = logging.getLogger(__name__)


class CallLogger(BaseCallbackHandler):
    """Logs the start, end and failure of each agent model call and tool
    call, so a slow question shows which step it is waiting on.
    """

    def on_chat_model_start(self, serialized, messages, **kwargs):
        logger.info("agent model call started")

    def on_llm_end(self, response, **kwargs):
        logger.info("agent model call finished")

    def on_llm_error(self, error, **kwargs):
        logger.info("agent model call failed: %.120s", error)

    def on_tool_start(self, serialized, input_str, **kwargs):
        logger.info("tool %s started", serialized.get("name"))

    def on_tool_end(self, output, **kwargs):
        logger.info("tool finished")


def configure_logging() -> None:
    """Timestamped progress lines on the terminal."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpcore", "sentence_transformers", "google_genai", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def message_text(content) -> str:
    """Plain text of a model message. Gemini 3 returns a list of blocks
    (text plus signature data); only the text blocks are kept.
    """
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "") for block in content if block.get("type") == "text"
    )


def is_transient_error(error: Exception) -> bool:
    """True for a temporary overload (503), a dropped connection or a
    per-minute rate limit (429), where waiting helps. A daily-quota 429 is not transient.
    """
    text = str(error)
    if any(marker in text for marker in TRANSIENT_MARKERS):
        return True
    return "429" in text and "PerDay" not in text


def invoke_with_retry(agent, payload: dict) -> dict:
    """agent.invoke with waiting retries on transient errors. The router's
    own model calls do not go through GeminiClient, so they need this.
    """
    wait = INVOKE_BACKOFF_SECONDS
    for attempt in range(INVOKE_ATTEMPTS):
        try:
            return agent.invoke(payload, config={"callbacks": [CallLogger()], "recursion_limit": RECURSION_LIMIT})
        except Exception as error:
            if not is_transient_error(error) or attempt == INVOKE_ATTEMPTS - 1:
                raise
            time.sleep(wait)
            wait *= 2


def ask(agent, history: list, question: str) -> tuple[str, list]:
    """Sends `question` after `history` to the agent. Returns the answer
    text and the updated message history.
    """
    messages = [*history, {"role": "user", "content": question}]
    result = invoke_with_retry(agent, {"messages": messages})
    updated_history = result["messages"]
    return message_text(updated_history[-1].content), updated_history


def main() -> None:
    ensure_api_key(prompt_if_missing=True)
    agent = build_router_agent()
    history: list = []
    print("Warhammer 40,000 rules assistant. Type 'exit' to quit.")
    while True:
        try:
            question = input(PROMPT).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in EXIT_COMMANDS:
            break
        try:
            answer, history = ask(agent, history, question)
        except Exception as error:
            print(f"Error: {error}\n")
            continue
        print(f"\n{answer}\n")


if __name__ == "__main__":
    main()
