# Retrieval

The system has two retrievers, one per source. They match the two modules the
brief asks for and are independent of each other. Choosing between them for a
given question is the job of the router (see `chatbot.md`).

| Retriever | Source | Method |
|---|---|---|
| Text (`retriever/text_retriever.py`) | 201 chunks of the Core Rules (`data/processed/chunks.json`) | BM25 and embedding search, merged by Reciprocal Rank Fusion |
| SQL (`retriever/sql_retriever.py`) | Unit datasheets (`data/raw/wahadb.sqlite`) | Gemini writes a SQL query from a data dictionary, which runs read-only |

## Text retriever

### Two channels merged by Reciprocal Rank Fusion

**Choice.** BM25 (lexical) and vector search (semantic) both rank the chunks, and
Reciprocal Rank Fusion (RRF) merges the rankings: `score(chunk) = sum over rankings
of 1 / (rank + 60)`.

**Why.** BM25 is strong on exact terms such as a stratagem or unit name, and blind
to paraphrase. Vector search handles paraphrase ("how do I get extra attacks") and
can miss a rare exact term. RRF combines them without putting raw scores on one
scale and without hand-tuned weights, which is the standard method taught in the
course. Keeping both channels also leaves a comparison between them available.

**Details.**
- Embeddings: `all-MiniLM-L6-v2` (Sentence-BERT) through ChromaDB's
  `SentenceTransformerEmbeddingFunction`. It is small and fast, and at 201 chunks
  the embedding model is unlikely to be the bottleneck. Alternatives from the MTEB
  leaderboard were not compared.
- BM25 tokenisation is deliberately simple: lowercase, split on letters and digits,
  no stemming, no stopword removal. The lexical channel exists to catch exact proper
  nouns, and stemming would blur them.
- The Chroma collection is persisted in `data/processed/chroma_db/` so embeddings
  are computed once. The collection stores a hash of all chunk texts; when the
  chunks change, the index is rebuilt. (Comparing only the number of chunks would
  keep stale embeddings when the text changes but the count does not.)

### Multiple query phrasings (RAG Fusion)

**Problem observed.** The question "what are the different ways to score points?"
did not retrieve the chunk that answers it (Capture and Control, page 59), even
within the top 10. The word "points" is overloaded in the book: most uses are
army-building points ("Points per Army: 1000 / 2000 / 3000"), and victory points
scored in a mission are a rarer sense. Both BM25 and the embeddings follow the
dominant sense.

**Choice.** The text tool accepts a list of 2 or 3 phrasings of the question, and
`retrieve_with_queries` fuses the BM25 and vector rankings of all of them in one
RRF call. The router agent writes the phrasings itself in its tool call and is told
to spell out ambiguous terms ("victory points scored in a mission", not "points").

**Why the agent writes them.** An earlier variant asked a second LLM request to
rewrite the query (`retrieve_with_query_fusion`, still in the code). The agent
already writes good search queries in its tool call, so the extra request was
redundant and doubled the cost of every search.

| Query set | Rank of the Capture and Control chunk |
|---|---|
| Original question, plain `retrieve()` | outside the top 10 |
| Original question with LLM-written variants | 2 |
| The agent's own three queries | 1 |
| Original question plus "victory points scored in a mission" | 4 |

### Number of passages returned

**Choice.** The router receives 8 passages per search (`RULES_TOP_K` in
`chatbot/router.py`).

**Why.** In the question "Can a unit shoot in the turn it Advanced?", the passage
with the exception (Assault weapons, page 25) ranks 1st for the plain question and
for the agent's best query, but 6th when the agent's three queries are fused: a
weakly worded query pulls in other Advance passages that appear in several
rankings, and RRF rewards agreement across rankings. With 5 passages the exception
was missing and Context Recall was 0.5. With 8 it is retrieved and recall is 1.0.
Context Precision stayed at 0.94 over the questions the sources can answer, so the
extra passages did not dilute the ranking (see `evaluation.md`).

| Query | BM25 rank | Vector rank |
|---|---|---|
| unit shoot after advance advance move | 34 | 22 |
| advance shooting phase eligible to shoot | 5 | 2 |
| advanced can it shoot assault | 1 | 1 |
| Can a unit shoot in the turn it Advanced? | 1 | 4 |

## SQL retriever

### Text-to-SQL with a static data dictionary

**Choice.** The basic pipeline: the full data dictionary
(`data/raw/SQL_AGENT_DATA_DICTIONARY.md`, 593 lines: schema, join graph, example
queries, instructions for SQL agents) is the system prompt, Gemini writes one
`SELECT`, and the retriever executes it.

**Why.** The course describes a dynamic variant (retrieve only the relevant tables
or examples per question) for dictionaries too large for one prompt. This one fits,
so a retrieval layer on top would solve a problem the project does not have.

### Read-only access, enforced twice

The course flags Text-to-SQL as the component that could delete or corrupt data if
it had write access, and recommends read-only access. Two independent controls
apply:

1. The SQLite file is opened in read-only URI mode (`file:...?mode=ro`), so any write
   fails in the database layer regardless of the SQL.
2. `check_sql_is_safe` rejects the generated SQL before execution unless it is
   exactly one statement starting with `SELECT` or `WITH`, and it rejects any
   mutating keyword (`DROP`, `DELETE`, `INSERT`, `UPDATE`, `ALTER`, `ATTACH`,
   `DETACH`, `PRAGMA`, `VACUUM`, `REINDEX`, `CREATE`, `REPLACE`) using word
   boundaries, so an identifier such as `update_count` is not flagged.

Neither is enough alone: the read-only mode does not stop a multi-statement string
from reaching `execute`, and a keyword blocklist is never airtight. `WITH` is
allowed because common table expressions are read-only and the model writes them
for questions with a breakdown. A rejected query is returned to the agent as a tool
message, not raised as an error, so the agent can try a different query.

### Ambiguous unit names

"What is the toughness of a space marine?" has no single answer: the database has
292 Space Marines datasheets, not one unit called "Space Marine". The router prompt
handles it: the agent asks the SQL tool for a breakdown of the value with example
unit names, answers with the most common case, names the units that differ, and
states the assumption it made. For toughness, 4 is the most common value (122
datasheets); the others are 3 (1), 5 (41), 6 (25), 7 (9) and 8 to 14 (vehicles).
Alternatives considered: asking a clarifying question back (costs an extra turn
before the user gets any answer) and picking one unit silently (hides the
ambiguity).
