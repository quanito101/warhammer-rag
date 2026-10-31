# Parsing and chunking

This document covers how the Core Rules PDF (60 pages) becomes the 201 text
chunks that the text retriever searches. Each section states a choice, the
alternative that was rejected and the reason.

## Parser: Docling

**Choice.** Docling parses the whole PDF in one pipeline (`parser/docling_parser.py`).

**Why.** Docling returns table structure, reading order and typed text items
(`section_header`, `text`, `table`) with page numbers, which is most of what
chunking needs. Assembling separate single-purpose tools (a text extractor, a
table extractor, an OCR step) would add integration code and more places for
errors to spread.

**How it was checked.** The PDF is heavily stylised: parchment textures, dice
values drawn as icons, stratagem cards, a graphical example datasheet. So the
parser output was compared by hand with the PDF on a representative sample of
pages (4 to 5, 6, 9, 21, 37, 41 to 42) before any chunking code was written.
The defects below come from that comparison.

## Parsing defects and what is done about each

| Defect | Where | Effect if left alone | Treatment |
|---|---|---|---|
| Icon-only values are dropped | Wound Roll table (p. 21), D3 table (p. 9), stratagem CP badges (pp. 41 to 42) | A chunk with plausible but wrong data, for example a wound roll column that reads `+` instead of `2+` to `6+` | Targeted post-processing, see below |
| Reading order is scrambled on pages with several competing regions | Introduction spread (pp. 4 to 5), stratagem pages | Text from unrelated sections spliced into one chunk | Structural chunking for those pages, page 4 excluded |
| Heading structure is flat and not unique | Whole document | Headings cannot serve as chunk boundaries or lookup keys | Headings are used only as a flat sequence, never as a hierarchy |
| Duplicated header row, stray control characters | Some tables and headings | Cosmetic noise in chunk text | Stripped by `chunker._strip_control_chars` and in table post-processing |
| Example datasheet (p. 37) has no text layer | Page 37 | The worked example is missing | Not recovered, see below |

### Icon-only values

Dice pips and CP-cost badges are a small, closed set of recurring visual
patterns, so they are repaired by document-specific code in
`parser/icon_fixups.py` instead of a general solution:

- **Dice tables.** `fix_wound_roll_table_grid` and `fix_d3_table_grid` replace
  the cells that extracted to `+`, `or` or an empty string with the correct
  values (`2+` to `6+`, and `1 or 2`, `3 or 4`, `5 or 6`).
- **Stratagem CP costs.** Docling does extract all 11 CP values, but in an order
  that does not match the stratagem they belong to (page 42 opens with four
  `1CP` items before any stratagem name). Reading position cannot be trusted,
  so the floating badges are ignored and the cost is looked up by stratagem
  name (`fix_stratagem_cp_cost`) when the stratagem's block is built. Every
  stratagem heading on pages 41 and 42 matches a key of the lookup exactly.
- **Example datasheet (p. 37).** Not recovered. It is a rendered graphic with no
  text, and it is an illustration, not game data: real unit statistics come
  from the SQLite database.

### Reading order and headings

Docling orders plain prose pages and pages with one callout box correctly, and
scrambles pages with several competing regions. Every heading also reports
`level 1`, and many heading texts repeat (`MOVEMENT` and `BENEFIT OF COVER` five
times each, `SELECT TARGETS` in three unrelated sections). Two consequences
shaped the chunking design:

- Chunk boundaries do not depend on a heading hierarchy, which does not exist
  in the parser output.
- Pages with several regions are chunked by structure that the code controls
  (one chunk per named stratagem) instead of by inferred reading order.

Page 4 (table of contents and introduction) is excluded: it is navigation, and
both its headings and body text are corrupted by the layout.

## Chunking strategy

**Choice.** Recursive character chunking (`RecursiveCharacterTextSplitter` from
`langchain-text-splitters`) for prose, with two kinds of chunk that never go
through the splitter.

**Why recursive chunking.** The course material orders chunking strategies with
fixed-length first and recursive second, ahead of semantic or hierarchy-based
chunking. Recursive splitting tries paragraph breaks, then sentence breaks, then
word breaks, so a rule is rarely cut mid-sentence. Hierarchy-based chunking was
rejected because the heading structure is unusable here (see above).

**Parameters.** `chunk_size = 900` characters (about 150 words) and
`chunk_overlap = 120`. Most rules in the book run 100 to 300 words, so a rule
fits in one or two chunks and the overlap keeps the sentences around a cut
together. The context window of the LLM does not constrain this value; it is a
retrieval granularity choice. The retrieval failures found in the evaluation
had other causes (see `evaluation.md`), so the value was kept.

**Chunk types.**

| Type | Count | Rule |
|---|---|---|
| table | 7 | A table is one atomic chunk, never split. The two dice tables are repaired first. |
| stratagem | 11 | One chunk per named stratagem, built from its heading to the next heading. The text starts with the stratagem's name, then the card text, then `[CP cost: ...]`. |
| prose | 183 | Everything else, split recursively. |

**The stratagem name is part of the text.** Retrieval and the answering model
see only the chunk text, not the `heading` field. When the name was stored only
in `heading`, the passage never said which stratagem it described, and a
faithfulness judge rejected all 12 claims of a correct "Fire Overwatch" answer
for that reason. Putting the name first in the text (`stratagem_chunk_text`)
fixed the score (0.08 to 1.00) and makes the name searchable by BM25 and by
the embeddings.

**Page numbers.** Every chunk carries the page where it starts. For prose, the
splitter records each chunk's start offset in the concatenated text and
`page_for_offset` maps it back to a page. This is what makes page citations in
answers reliable.

## Output

`scripts/build_chunks.py` runs the whole pipeline and writes
`data/processed/chunks.json`: a list of chunks with `chunk_id`, `text`,
`page_no`, `heading` and `chunk_type`. The retrievers read this file, so the
PDF is not parsed again at query time. The file is included in the repository;
rebuilding it takes a few minutes because Docling parses all 60 pages.

## Known limitations

- Icon-only values other than dice pips and CP badges were not systematically
  searched for. The sample pages showed no others, but this is not proof for
  all 60 pages.
- Prose is concatenated before splitting, with no awareness of section
  boundaries. A diagram (the battle round phase list on page 10) is interleaved
  with surrounding prose in one chunk. The phase names are also stated plainly
  elsewhere, so nothing is lost to retrieval, and the evaluation did not show a
  wrong answer caused by it.
- Three tables on page 55 share the title `Points per Army`. Their content is
  intact (each has its own `1000`, `2000` or `3000` row), only the `heading`
  metadata collides, and nothing looks up a table by heading.
