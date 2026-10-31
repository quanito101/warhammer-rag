"""Parses the Core Rules PDF with Docling, applies the table fixes, chunks it
and writes data/processed/chunks.json. Takes a few minutes.

Run with: uv run python scripts/build_chunks.py
"""

from project_rag.parser.docling_parser import parse_pdf
from project_rag.parser.icon_fixups import fix_d3_table_grid, fix_wound_roll_table_grid
from project_rag.chunker.chunker import chunk_document, save_chunks

doc = parse_pdf("data/raw/Core Rules.pdf")

fix_d3_table_grid(doc.tables[0].data.grid)
fix_wound_roll_table_grid(doc.tables[1].data.grid)

chunks = chunk_document(doc)

by_type: dict[str, int] = {}
for chunk in chunks:
    by_type[chunk.chunk_type] = by_type.get(chunk.chunk_type, 0) + 1

print(f"Total chunks: {len(chunks)}")
for chunk_type, count in by_type.items():
    print(f"  {chunk_type}: {count}")

print()
print("--- D3 table chunk (should show corrected dice-range text) ---")
d3_chunks = [c for c in chunks if c.chunk_type == "table" and c.page_no == 9]
if d3_chunks:
    print(d3_chunks[0].text)

print()
print("--- Wound Roll table chunk (should be the fixed values) ---")
wound_roll_chunks = [c for c in chunks if c.chunk_type == "table" and c.page_no == 21]
if wound_roll_chunks:
    print(wound_roll_chunks[0].text)

print()
print("--- Sample stratagem chunk (Grenade) ---")
grenade_chunks = [c for c in chunks if c.chunk_type == "stratagem" and c.heading == "GRENADE"]
if grenade_chunks:
    print(grenade_chunks[0].text)

print()
print("--- Checking for leftover control characters across all chunks ---")
control_char_hits = [c for c in chunks if any(ord(ch) < 32 and ch not in "\n\t" for ch in c.text)]
print(f"Chunks with leftover control characters: {len(control_char_hits)}")

print()
output_path = "data/processed/chunks.json"
save_chunks(chunks, output_path)
print(f"Saved {len(chunks)} chunks to {output_path}")