from project_rag.parser.docling_parser import parse_pdf

doc = parse_pdf("data/raw/Core Rules.pdf")

wound_roll_table = doc.tables[1]

for row_idx, row in enumerate(wound_roll_table.data.grid):
    for col_idx, cell in enumerate(row):
        print(f"row {row_idx}, col {col_idx}: {cell.text!r}")
