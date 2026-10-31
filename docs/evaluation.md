# Evaluation

The evaluation measures retrieval and answer generation separately, because they
fail in different ways, and it is built so that one run can be spread over several
days of API quota.

## Metrics (RAGAS)

| Metric | Question it answers | Inputs |
|---|---|---|
| Faithfulness | Is every claim of the answer supported by the retrieved context? | answer, contexts |
| Context Recall | Does the retrieved context contain what the reference answer says? | contexts, reference |
| Context Precision | Are the relevant passages ranked first? | question, contexts, reference |

**Why these three.** Two failures showed up in manual tests before any metric
existed: a retrieval miss (the word "points" in a scoring question) and an answer
that added rules the PDF does not contain. Context Recall and Context Precision
measure the first, Faithfulness the second. Answer Relevancy is not used: it needs
an embedding model on the judge side, which adds a second API to configure, and
it does not measure either of the two failures above.

## Question set

`data/eval/questions.json` holds 10 questions, each with a reference answer written
from the source text and a `source` field naming the page or table. They were
written by hand so that every reference is checkable, and they cover each way the
system can succeed or fail.

| id | Type | Question |
|---|---|---|
| q01 | rules_table | What D6 result is needed to wound when the attack's Strength is twice the target's Toughness or more? |
| q02 | rules_stratagem | What does the Fire Overwatch stratagem do, and what does it cost? |
| q03 | rules_multi_chunk | Can a unit shoot in the turn it Advanced? (rule and exception are in different chunks) |
| q04 | rules_text | What are the effects of a unit being Battle-shocked? |
| q05 | rules_text | How many Command points do players gain in the Command phase, and can they gain more elsewhere? |
| q06 | rules_scoring | How are Victory points scored in the Capture and Control mission? |
| q07 | sql | What is the toughness of an Intercessor Squad? |
| q08 | sql | How many points does a 10-model Intercessor Squad cost? |
| q09 | hybrid | What D6 roll does a Strength 8 weapon need to wound an Intercessor Squad? (database and rules) |
| q10 | out_of_scope | Which secondary objectives are available in the Leviathan mission pack? (correct behaviour: say the sources do not cover it) |

RAGAS can also generate questions from the chunks. That was not used: each generated
question costs additional LLM requests, and hand-written questions with sourced
references are easier to verify.

## Procedure

Two scripts, kept apart so that each can stop and resume:

1. `uv run python -m project_rag.evaluation.collect` runs the router on every
   question and stores the answer, the retrieved contexts (one per rulebook passage,
   one per SQL result) and the tool calls in `data/eval/results.json`, after each
   question. Finished questions are skipped on the next run; `--ids q02 --redo`
   runs chosen questions again.
2. `uv run python -m project_rag.evaluation.score` scores the stored results and
   writes each metric of each question to `data/eval/scores.json` as soon as it
   exists. A retry or a later run repeats only what is missing; `--redo q02`
   discards chosen scores. Add `--metrics faithfulness context_recall
   context_precision` to choose metrics (the first two are the default).

**Judge.** `gemini-3.6-flash`, the same model as the one answering. All scores come
from this one judge model.

**Cost.** About 3 judge requests per question for Faithfulness and Context Recall,
and one more per retrieved passage for Context Precision.

## Integration notes

- RAGAS reaches Gemini through its OpenAI-compatible endpoint, the workaround RAGAS
  documents for an invalid-safety-settings error of the google-genai SDK.
- The judge's output limit is raised to 8192 tokens. At RAGAS's default of 1024,
  Gemini 3 models (which spend part of the limit on internal reasoning) returned
  truncated structured output.
- RAGAS reports a failed judge call as `nan` by default. The scorer asks it to raise,
  stores only real numbers, waits and retries on a per-minute rate limit or a 503,
  and stops on a daily-quota error.
- `ragas` 0.4.3 imports a Vertex AI class that `langchain-community` 0.4 no longer
  has. `evaluation/ragas_compat.py` registers an empty stand-in before the import;
  nothing here uses Vertex AI.
- `ragas` depends on `scikit-network`, which has no prebuilt package for Python 3.14
  and failed to compile on the development machine. RAGAS imports it only in its
  test-set generation code, so `pyproject.toml` excludes it with
  `override-dependencies`. RAGAS's own question generator therefore does not work in
  this environment.

## Results

Final run, judge `gemini-3.6-flash`.

| id | Faithfulness | Context Recall | Context Precision |
|---|---|---|---|
| q01 | 0.67 | 1.00 | 1.00 |
| q02 | 1.00 | 1.00 | 1.00 |
| q03 | 1.00 | 1.00 | 0.68 |
| q04 | 1.00 | 1.00 | 0.81 |
| q05 | 1.00 | 1.00 | 1.00 |
| q06 | 0.83 | 1.00 | 1.00 |
| q07 | 1.00 | 1.00 | 1.00 |
| q08 | 1.00 | 0.50 | 1.00 |
| q09 | 0.50 | 1.00 | 1.00 |
| q10 | 1.00 | 0.50 | 0.33 |

| | Before the fixes | Final run |
|---|---|---|
| Faithfulness, mean of 10 | 0.76 | 0.90 |
| Context Recall, mean of 10 | 0.85 | 0.90 |
| Context Recall, without q10 | 0.94 | 0.94 |
| Context Precision, mean of 10 | not measured | 0.88 |
| Context Precision, without q10 | not measured | 0.94 |

The per-question scores of the earlier run are in `data/eval/baseline/`.

### What the evaluation found

1. **Stratagem chunks did not contain their names (q02).** Faithfulness was 0.08: the
   judge rejected all 12 claims of a correct answer because the retrieved passage
   never named "Fire Overwatch". The name was stored only in the chunk's `heading`
   field. It is now the first line of the chunk text (see `parsing_and_chunking.md`),
   and the score is 1.00.
2. **A relevant passage ranked just outside the top 5 (q03).** The Assault exception
   was missing and Context Recall was 0.5. Returning 8 passages instead of 5 fixed it
   (see `retrieval.md`); the answer now states the exception and cites page 25.

### How to read the numbers

- **q10 is out of scope on purpose.** Its reference says the sources cannot answer,
  so there is nothing in the retrieved context for Recall or Precision to find. The
  agent's refusal is correct and scores Faithfulness 1.00. The figures "without q10"
  are the fairer retrieval measure.
- **Faithfulness checks claims against the retrieved context only, not against the
  question.** A claim that restates the question ("the attack has Strength 8", q09) is
  counted as unsupported, so a correct answer scores below 1. A diagnostic that prints
  each extracted claim and the judge's verdict is in `scripts/diagnose_faithfulness.py`.
  q01 is probably the same effect; this was not verified claim by claim.
- **q08 Recall (0.50)** reflects the reference answer, which adds "(a 5-model squad
  costs 80 points)" although the question asks only about 10 models. In one run the
  agent also queried the 5-model rows, in another it did not.

### Limits

- Each question was run once per setup, and every answer text changed between runs,
  including database questions that the fixes do not touch. Only the large changes
  (q02, q03) can be attributed to a specific fix; differences of a few tenths on one
  question are within run-to-run variation.
- One judge model, which is also the answering model.
- Ten questions is a small sample. The set is meant to show that the pipeline works
  and to expose failure modes, not to estimate accuracy.
