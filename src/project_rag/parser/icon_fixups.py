"""
Targeted post-processing for icon-only values that Docling silently drops.

Background: Docling's text extraction cannot read game values that are
rendered purely as icons in the PDF (dice-pip faces, CP-cost badges).
Cells that should contain a value like "2+" or "1CP" are actually extracted 
as a bare "+", the word "or", or nothing at all. See docs/parsing_and_chunking.md
("Finding 1") for details.

This is NOT a general OCR/icon-recognition solution. The icon vocabulary in 
this book is small and closed (six D6 faces, a few CP-cost badges), so we manually
map known tables/cards rather than building a general recogniser. 

WOUND_ROLL_TABLE: Wound Roll table (Core Rules, p. 21). Row order matches
the table as printed: Strength vs Toughness comparison -> D6 result
required to wound.

D3_ROLL_TABLE: Rolling a D3 table (Core Rules, p. 9). D6 result -> D3
result (per "roll a D6 and halve the result, rounding up").

STRATAGEM_CP_COSTS: All 11 Core Stratagems shown in the Core Rules (pp.
41-42). Codex-specific stratagems are NOT covered here -- this book only
contains the Core set.
"""

from __future__ import annotations
 
from dataclasses import dataclass
 
 
WOUND_ROLL_TABLE: list[tuple[str, str]] = [
    ("Strength is TWICE (or more than twice) the Toughness.", "2+"),
    ("Strength is GREATER than the Toughness.", "3+"),
    ("Strength is EQUAL to the Toughness.", "4+"),
    ("Strength is LESS than the Toughness.", "5+"),
    ("Strength is HALF (or less than half) the Toughness.", "6+"),
]
 
D3_DICE_RANGE_TEXT: dict[int, str] = {
    1: "1 or 2",
    2: "3 or 4",
    3: "5 or 6",
}
 
STRATAGEM_CP_COSTS: dict[str, int] = {
    "COMMAND RE-ROLL": 1,
    "COUNTER-OFFENSIVE": 2,
    "EPIC CHALLENGE": 1,
    "INSANE BRAVERY": 1,
    "GRENADE": 1,
    "TANK SHOCK": 1,
    "RAPID INGRESS": 1,
    "FIRE OVERWATCH": 1,
    "GO TO GROUND": 1,
    "SMOKESCREEN": 1,
    "HEROIC INTERVENTION": 2,
}
 
 
@dataclass
class IconFixupResult:
    applied: bool
    value: str | None
    note: str
 
 
def fix_wound_roll_row(condition_text: str) -> IconFixupResult:
    """Given the left-hand condition text of a Wound Roll table row,
    return the correct D6-required value.
 
    Match is case-insensitive and tolerant of minor whitespace/punctuation
    differences from re-parsing, since the exact string coming out of
    Docling may not match verbatim.
    """
    normalized = " ".join(condition_text.split()).lower()
    for condition, value in WOUND_ROLL_TABLE:
        if " ".join(condition.split()).lower() == normalized:
            return IconFixupResult(True, value, "matched Wound Roll table")
    return IconFixupResult(False, None, "no match in WOUND_ROLL_TABLE")
 
 
def fix_d3_dice_result(d3_result_text: str) -> IconFixupResult:
    """Given a D3 RESULT cell's (already-correct) text ("1", "2", or
    "3"), return the DICE RESULT text that should replace the broken
    "or" cell in the same row.
    """
    try:
        d3_value = int(d3_result_text.strip())
    except ValueError:
        return IconFixupResult(False, None, f"'{d3_result_text}' is not a D3 value")
    if d3_value in D3_DICE_RANGE_TEXT:
        return IconFixupResult(
            True, D3_DICE_RANGE_TEXT[d3_value], "matched D3_DICE_RANGE_TEXT"
        )
    return IconFixupResult(False, None, f"D3 value {d3_value} out of range 1-3")
 
 
def fix_stratagem_cp_cost(stratagem_name: str) -> IconFixupResult:
    """Given a stratagem name, return its CP cost.
 
    Match is case-insensitive. Only covers the Core Stratagems in this
    rulebook -- returns applied=False for anything else (e.g. a Codex
    stratagem, which this pipeline doesn't index).
    """
    key = stratagem_name.strip().upper()
    if key in STRATAGEM_CP_COSTS:
        return IconFixupResult(
            True, f"{STRATAGEM_CP_COSTS[key]}CP", "matched STRATAGEM_CP_COSTS"
        )
    return IconFixupResult(False, None, "not a known Core Stratagem")
 
 
def fix_wound_roll_table_grid(grid: list[list]) -> list[str]:
    """Given the Wound Roll table's `table.data.grid` (as returned by
    Docling), correct each broken "+" cell in place using
    fix_wound_roll_row. Returns a list of notes describing what was
    changed, for logging/review.
 
    Confirmed against a real parsed table: row 0 is a merged header (the
    "WOUND ROLL" title and the first header line collapse into one
    cell), and rows 1-5 are the 5 data rows, each with the condition
    text in column 0 and the (broken) required-roll value in column 1.
    """
    notes: list[str] = []
    for row in grid[1:]:
        condition_cell, value_cell = row[0], row[1]
        if value_cell.text.strip() != "+":
            continue
        result = fix_wound_roll_row(condition_cell.text)
        if result.applied:
            notes.append(f"Row '{condition_cell.text}': '+' -> '{result.value}'")
            value_cell.text = result.value
        else:
            notes.append(
                f"Could not resolve value for '{condition_cell.text}' ({result.note})"
            )
    return notes
 
 
def fix_d3_table_grid(grid: list[list]) -> list[str]:
    """Given a D3 table's `table.data.grid` (as returned by Docling),
    correct each broken "or" cell in place using the D3 RESULT value
    already present in the same row. Returns a list of notes describing
    what was changed, for logging/review.
 
    Confirmed against a real parsed table: rows 0-1 are headers (row 0
    duplicates the "ROLLING A D3" title across both columns; row 1 holds
    "DICE RESULT" / "D3 RESULT"), and rows 2 onward are data rows with
    the D3 RESULT column already correct.
    """
    notes: list[str] = []
    for row in grid[2:]:
        dice_result_cell, d3_result_cell = row[0], row[1]
        if dice_result_cell.text.strip().lower() != "or":
            continue
        result = fix_d3_dice_result(d3_result_cell.text)
        if result.applied:
            notes.append(
                f"Row with D3 result '{d3_result_cell.text}': "
                f"DICE RESULT '{dice_result_cell.text}' -> '{result.value}'"
            )
            dice_result_cell.text = result.value
        else:
            notes.append(
                f"Could not resolve DICE RESULT for D3 result "
                f"'{d3_result_cell.text}' ({result.note})"
            )
    return notes
 
 
def apply_known_fixups(chunk_text: str, chunk_type: str) -> tuple[str, list[str]]:
    """Apply whichever fixup is relevant to a parsed chunk, based on its
    type. Returns the (possibly corrected) text and a list of notes
    describing what was changed, for logging/review.
 
    `chunk_type` is expected to be one of: "wound_roll_table",
    "d3_roll_table", "stratagem_card". Extend this as more icon-only
    patterns are identified elsewhere in the book.
 
    Expects the stratagem name to be the first line of `chunk_text` when
    chunk_type is "stratagem_card". The wound_roll_table / d3_roll_table
    cases are table-structured rather than freeform text, so they are
    better handled row-by-row where the table is actually parsed (before
    it is flattened to a chunk) -- call fix_wound_roll_row / fix_d3_roll
    directly there rather than through this text-level entry point.
    """
    notes: list[str] = []
 
    if chunk_type == "stratagem_card":
        first_line = chunk_text.strip().splitlines()[0] if chunk_text.strip() else ""
        result = fix_stratagem_cp_cost(first_line)
        if result.applied:
            chunk_text = f"{chunk_text}\n\n[CP cost: {result.value}]"
            notes.append(f"Injected CP cost for '{first_line}': {result.value}")
        else:
            notes.append(f"Could not resolve CP cost for '{first_line}' ({result.note})")
 
    return chunk_text, notes