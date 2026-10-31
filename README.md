# Warhammer 40,000 rules assistant

A retrieval-augmented chatbot that answers questions about the Warhammer 40,000
Core Rules (a 60-page PDF) and about unit statistics (a SQLite database). It uses
Google AI Studio (Gemini) as its LLM.

Mini-project for the course taught by Florent Storme, M2 Computational Linguistics
(NLP), Université Paris Cité.

## Authors

- **HO Ngoc Le Quan** (Quân) ([@quanito101](https://github.com/quanito101))
- **Pablo Gagliardi** ([@pabgagliardi](https://github.com/pabgagliardi))

The PDF parser, the chunker, the two retrievers (text and SQL) and the router agent
were built together by both authors in pair-programming sessions. Quân then built the
chat interface, the LLM client with retries and cache, the evaluation pipeline and the
documentation.

## What it does

```
                      question
                          |
                          v
        Router agent (LangGraph ReAct, Gemini)
              |                         |
   retrieve_rules_text         retrieve_unit_data
              |                         |
       Text retriever             SQL retriever
   BM25 + embeddings + RRF     Gemini writes SQL, read-only
              |                         |
   201 chunks of the Core       wahadb.sqlite (unit
   Rules (page numbers)         datasheets)
              \                         /
               v                       v
        answer grounded in the retrieved text,
        with page citations or "from the unit database"
```

Examples: "What is the rule for grenades?", "How can I use a stratagem?",
"What are the different ways to score points?", "What is the toughness of a space
marine?", "What D6 roll does a Strength 8 weapon need to wound an Intercessor Squad?"

## Quick start

Requirements: [uv](https://docs.astral.sh/uv/) (it installs Python 3.14 if needed)
and an internet connection. Everything else, including the two data files, is in
the repository.

```bash
uv sync
```

**1. Provide your Google AI Studio API key** (create one at
https://aistudio.google.com/apikey). Any one of these works:

- Run `export GEMINI_API_KEY=your-key` in the terminal.
- Copy `.env.example` to `.env` and put the key in it.
- Do nothing: the chat asks for the key with hidden input if it finds none.

**2. Start the chat.**

```bash
uv run project-rag
```

The first start downloads the embedding model (`all-MiniLM-L6-v2`, about 90 MB) and
builds the search index, which takes about a minute. Type `exit` to quit.

A free AI Studio key allows about 20 requests per day on the default model
(`gemini-3.6-flash`), and a question uses 2 to 4 requests, so expect 5 to 6 questions
per day. A key with billing enabled has no such limit. `GEMINI_MODEL` selects
another model.

## Tests and evaluation

```bash
uv run pytest                                              # unit tests, no API calls
uv run python -m project_rag.evaluation.collect            # ask the 10 evaluation questions
uv run python -m project_rag.evaluation.score --metrics faithfulness context_recall context_precision
```

The evaluation uses about 100 LLM requests. Stored results and scores from the last
run are in `data/eval/`; see `docs/evaluation.md` for the method and the numbers.

To rebuild the chunks from the PDF (a few minutes, no API key needed):
`uv run python scripts/build_chunks.py`.

## Results

On 10 hand-written questions with reference answers (RAGAS, Gemini as judge):

| Metric | Mean of 10 | Mean without the out-of-scope question |
|---|---|---|
| Faithfulness | 0.90 | n/a |
| Context Recall | 0.90 | 0.94 |
| Context Precision | 0.88 | 0.94 |

## Technical choices

Each document states the choices, the rejected alternatives and the reasons.

| Choice | Short justification | Details |
|---|---|---|
| Docling parser, targeted repair of dice and CP icons | Handles tables and reading order; icon-only values would otherwise be silently wrong | `docs/parsing_and_chunking.md` |
| Recursive chunking (900 characters, 120 overlap), atomic tables, one chunk per stratagem with its name in the text | Heading structure is unusable; tables and cards must not be split | `docs/parsing_and_chunking.md` |
| BM25 + embeddings merged by Reciprocal Rank Fusion | Exact names and paraphrase both matter; no score weighting needed | `docs/retrieval.md` |
| The agent supplies 2 or 3 query phrasings, 8 passages returned | Fixes a polysemy miss ("points") and a missed exception, at no extra LLM request | `docs/retrieval.md` |
| Text-to-SQL with the full data dictionary, read-only twice | The dictionary fits in one prompt; the database can never be modified | `docs/retrieval.md` |
| LangGraph ReAct agent with two tools | One mechanism gives routing and multi-source questions | `docs/chatbot.md` |
| Grounding rules in the system prompt | The model added rules that are not in the PDF | `docs/chatbot.md` |
| Retries, timeouts and a response cache | Gemini returns temporary errors; the free quota is small | `docs/chatbot.md` |
| RAGAS with a Gemini judge, resumable runs | Separates retrieval from answer failures; fits a small quota | `docs/evaluation.md` |

## Repository layout

```
src/project_rag/
  parser/       Docling parsing and the icon fixes
  chunker/      Chunking and chunks.json writer
  retriever/    text_retriever.py, sql_retriever.py
  llm/          Gemini client (retries, timeouts) and response cache
  chatbot/      router.py (agent and prompt), chat.py (command-line chat)
  evaluation/   collect.py, score.py, RAGAS compatibility shim
  config.py     API key lookup
scripts/        Build and check scripts; exploration/ holds parser investigation scripts
tests/          Unit tests (no API calls)
docs/           Technical choices and justifications
data/raw/       Core Rules PDF, wahadb.sqlite, SQL data dictionary
data/processed/ chunks.json (the search index is generated on first run)
data/eval/      Questions, results and scores
```

## Data

The Core Rules PDF and the unit database are course material and are included for
the purpose of this assignment. Warhammer 40,000 is a trademark of Games Workshop.
