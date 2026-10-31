"""
Splits a parsed Core Rules DoclingDocument into chunks for retrieval.

Three chunk types, handled separately rather than uniformly, per
docs/parsing_and_chunking.md:

- Table chunks (chunk_tables): one atomic chunk per table, never touched
  by the recursive splitter. Assumes the table's cell values have already
  been corrected via parser.icon_fixups where needed (e.g. the Wound Roll
  and D3 tables) -- this module does not apply those fixups itself.
- Stratagem chunks (chunk_stratagems): one atomic chunk per named
  Core Stratagem on pp. 41-42, grouped from its section_header to the
  next, with its CP cost injected by name lookup rather than trusted from
  the page's floating CP-badge text (see Finding 1/4: badge position is
  unreliable, badge text itself is noise here and discarded).
- Prose chunks (chunk_prose): everything else, split with
  RecursiveCharacterTextSplitter. Page 4 (TOC/Introduction) and pages
  41-42 (handled separately as stratagem chunks) are excluded.

chunk_document combines all three into one list.
"""

from __future__ import annotations

from bisect import bisect_right
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from project_rag.parser.icon_fixups import STRATAGEM_CP_COSTS, fix_stratagem_cp_cost

CHUNK_SIZE = 900
CHUNK_OVERLAP = 120

EXCLUDED_PAGES = {4}
STRATAGEM_PAGES = {41, 42}

_CP_BADGE_PATTERN = re.compile(r"^\d+CP$")

_CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _strip_control_chars(text: str) -> str:
    """Removes stray control characters Docling sometimes glues onto
    text near decorative icons (bullet markers, the page-number triangle,
    category icons) -- e.g. the \\x07/\\x08 bytes seen in Findings 1 and
    4 ('ONLY WAR\\x08', bullets rendered as '\\x07'). Printable text and
    normal whitespace (space, tab, newline) are untouched.
    """
    return _CONTROL_CHAR_PATTERN.sub("", text)


@dataclass
class Chunk:
    text: str
    page_no: int | None
    heading: str | None
    chunk_type: str


def _is_stratagem_page_noise(text: str) -> bool:
    """True for text items on the stratagem pages that are not part of
    any stratagem's actual content: floating CP-cost badges (matched by
    _CP_BADGE_PATTERN, e.g. "1CP", "2CP") and the stray single-letter
    icon-glyph fragment ("F") documented in Finding 1.
    """
    stripped = text.strip()
    return bool(_CP_BADGE_PATTERN.match(stripped)) or stripped == "F"


def chunk_tables(doc) -> list[Chunk]:
    """One chunk per table in `doc.tables`, rendered as pipe-separated
    rows. Assumes any needed icon fixups (Wound Roll, D3) have already
    been applied to the table's cells before this is called.
    """
    chunks: list[Chunk] = []
    for table in doc.tables:
        page_no = table.prov[0].page_no if table.prov else None
        rows = [
            " | ".join(_strip_control_chars(cell.text) for cell in row)
            for row in table.data.grid
        ]
        heading = _strip_control_chars(table.data.grid[0][0].text) if table.data.grid else None
        chunks.append(
            Chunk(text="\n".join(rows), page_no=page_no, heading=heading, chunk_type="table")
        )
    return chunks


def stratagem_chunk_text(heading: str, lines: list[str], cp_cost: str | None) -> str:
    """Text of one stratagem chunk: the stratagem's name, its card text and,
    when known, its CP cost. The name is part of the text because retrieval
    and the answering model only see the text, not the heading field.
    """
    text = f"{heading}\n{chr(10).join(lines).strip()}"
    return f"{text}\n\n[CP cost: {cp_cost}]" if cp_cost else text


def chunk_stratagems(doc) -> list[Chunk]:
    """One chunk per named Core Stratagem on pp. 41-42, from its
    section_header to the next. Non-stratagem headers on these pages
    (e.g. "STRATAGEMS KEY", "STRATAGEM CATEGORIES") are not stratagem
    names and are skipped as chunk boundaries, but any text under them is
    still discarded since it precedes the first real stratagem card. CP
    cost is injected by name lookup (fix_stratagem_cp_cost) rather than
    read from the page's own CP-badge text, which is unreliable by
    position (Finding 1) and discarded here via
    _is_stratagem_page_noise.
    """
    chunks: list[Chunk] = []
    current_heading: str | None = None
    current_page: int | None = None
    current_lines: list[str] = []

    def flush() -> None:
        if current_heading is None:
            return
        result = fix_stratagem_cp_cost(current_heading)
        text = stratagem_chunk_text(
            current_heading, current_lines, result.value if result.applied else None
        )
        chunks.append(
            Chunk(text=text, page_no=current_page, heading=current_heading, chunk_type="stratagem")
        )

    for item in doc.texts:
        if not item.prov or item.prov[0].page_no not in STRATAGEM_PAGES:
            continue
        if item.label == "section_header":
            if item.text.strip().upper() not in STRATAGEM_CP_COSTS:
                continue
            flush()
            current_heading = _strip_control_chars(item.text.strip())
            current_page = item.prov[0].page_no
            current_lines = []
            continue
        if current_heading is None or _is_stratagem_page_noise(item.text):
            continue
        current_lines.append(_strip_control_chars(item.text))

    flush()
    return chunks


def page_for_offset(part_starts: list[int], part_pages: list[int | None], offset: int) -> int | None:
    """Page of the text part that contains character `offset`, where
    part_starts[i] is the offset at which part i begins.
    """
    index = bisect_right(part_starts, offset) - 1
    return part_pages[index] if index >= 0 else None


def chunk_prose(doc) -> list[Chunk]:
    """Recursive-splits everything in doc.texts that isn't excluded
    (page 4, per Finding 4) or handled separately (the stratagem pages,
    per chunk_stratagems).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
    )
    parts: list[str] = []
    part_pages: list[int | None] = []
    last_page: int | None = None
    for item in doc.texts:
        page_no = item.prov[0].page_no if item.prov else None
        if page_no in EXCLUDED_PAGES or page_no in STRATAGEM_PAGES:
            continue
        if item.label == "page_footer":
            continue
        if page_no is not None:
            last_page = page_no
        parts.append(_strip_control_chars(item.text))
        part_pages.append(last_page)

    separator = "\n\n"
    full_text = separator.join(parts)
    part_starts = []
    position = 0
    for part in parts:
        part_starts.append(position)
        position += len(part) + len(separator)

    pieces = splitter.create_documents([full_text])
    return [
        Chunk(
            text=piece.page_content,
            page_no=page_for_offset(part_starts, part_pages, piece.metadata["start_index"]),
            heading=None,
            chunk_type="prose",
        )
        for piece in pieces
    ]


def chunk_document(doc) -> list[Chunk]:
    """All chunks for `doc`: tables, then stratagems, then prose."""
    return [*chunk_tables(doc), *chunk_stratagems(doc), *chunk_prose(doc)]


def save_chunks(chunks: list[Chunk], path: Path | str) -> None:
    """Writes `chunks` to `path` as a JSON array, one object per chunk
    (text, page_no, heading, chunk_type, and a 0-based chunk_id matching
    its position in the list). This is the persistence step the Status
    section of docs/parsing_and_chunking.md flagged as still missing:
    chunk_document's output previously only existed inside a running
    script's memory.
    """
    payload = [{"chunk_id": i, **asdict(chunk)} for i, chunk in enumerate(chunks)]
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def load_chunks(path: Path | str) -> list[dict]:
    """Reads chunks back from a file written by save_chunks. Returns
    plain dicts (not Chunk instances, since chunk_id isn't a Chunk
    field) -- callers needing Chunk objects can reconstruct them with
    Chunk(**{k: v for k, v in d.items() if k != "chunk_id"}).
    """
    return json.loads(Path(path).read_text())