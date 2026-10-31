# Router agent, chat interface and LLM layer

## Router: a ReAct agent with two tools

**Choice.** `chatbot/router.py` builds a LangGraph `create_react_agent` with two
tools, `retrieve_rules_text` (text retriever) and `retrieve_unit_data` (SQL
retriever). The model decides per question which tool to call, or both.

**Why.** This is the course's "router agent" pattern (an LLM with several
retrievers picks which to call) and its ReAct pattern at the same time, because the
prebuilt ReAct loop already is both. Two tool functions replace a hand-written
if/else dispatcher, which would need its own question classifier. A hybrid question
("What D6 roll does a Strength 8 weapon need to wound an Intercessor Squad?") needs
both sources, and the agent handles it in one run.

**Why LangGraph.** It is the framework named in the course material and in the
internship requirements this project was written against, and the course teaches
the patterns without tying them to another framework such as Google ADK.

## Grounding rules in the system prompt

Retrieval can be correct and the answer still wrong. On the question "What are the
different ways to score points?", retrieval found the Capture and Control chunk, but
the first answer also listed mission packs, secondary objectives and bonus points
that appear nowhere in the PDF: the model filled the gap from its own knowledge of
the game. The system prompt therefore contains these rules:

- Answer only from what the tools return. The searchable rulebook is the Core Rules
  only (no mission packs, secondary objective cards or Crusade rules).
- Say so when the retrieved text does not cover part of the question.
- Do not name an edition of the game; the sources do not state one.
- Cite the page for rules text and say when a figure comes from the unit database.
- For a generic unit name, ask for a breakdown and state the assumption (see
  `retrieval.md`).
- Search economically: one `retrieve_rules_text` call with 2 or 3 phrasings, a second
  call only if the first result is clearly unrelated, never more than two.

The out-of-scope evaluation question (Leviathan secondary objectives) checks the
first and second rules: the agent answers that the Core Rules do not cover it.

## Cost control

The free Gemini tier allows 20 requests per day per project and model and 5 per
minute, and an unconstrained agent searched about ten times per question. Four
measures bring a typical question to 2 to 4 requests:

| Measure | Where | Effect |
|---|---|---|
| Query list in one tool call | `retrieve_rules_text(queries)` | No separate query-rewriting request |
| At most two searches | system prompt | Stops search loops |
| Recursion limit 10 | `RECURSION_LIMIT` in `chatbot/chat.py` | Hard cap if the prompt is ignored |
| Response cache | `llm/cache.py` | An identical SQL-generation request is read from `data/processed/llm_cache.sqlite` |

The cache covers `GeminiClient` calls (SQL generation). The router's own reasoning
calls go through `ChatGoogleGenerativeAI` and are not cached.

## Resilience to API errors

Gemini returns temporary errors often enough that the chat must absorb them.

- **Retried with waiting:** 503 (overload), 504 (deadline), a per-minute 429, dropped
  connections and timeouts. `GeminiClient` retries with exponential backoff, and
  `invoke_with_retry` in `chatbot/chat.py` wraps a whole agent run, because the
  router's model calls bypass `GeminiClient`.
- **Never retried:** a daily-quota 429 (its message contains `PerDay`). Waiting
  seconds cannot fix it, and each retry would be another request against the quota.
- **Single retry layer:** the Google SDK's own retries (5 attempts by default) are
  switched off, so each failure costs one request. The LangChain model uses
  `max_retries=1`.
- **Timeouts:** every request has a 60 second timeout, so a stalled connection fails
  and is retried instead of hanging.

## Chat interface

`chatbot/chat.py` is a command-line loop (`uv run project-rag`). It keeps the
conversation history, so follow-up questions work, and prints timestamped progress
lines for each model call and tool call (a `CallLogger` callback), which shows what a
slow question is waiting on. `exit`, `quit` or `:q` ends the session.

## API key and model

- **Key.** `config.ensure_api_key` looks for `GEMINI_API_KEY` in the environment,
  then in a `.env` file at the project root (see `.env.example`), then, in the chat
  only, asks for it with hidden input. Pasted keys are cleaned of whitespace, quotes
  and control characters, which otherwise make the API reject the key. The key is
  never written to disk by the code.
- **Model.** `gemini-3.6-flash` by default, overridable with `GEMINI_MODEL`. All
  evaluation numbers come from this model. A free key allows about 20 requests per
  day on it, which is roughly 5 to 6 questions; a key with billing enabled has no
  such limit.
- **LLM backend.** Google AI Studio (Gemini), as the brief requires, through the
  `google-genai` SDK (SQL generation) and `langchain-google-genai` (router).
