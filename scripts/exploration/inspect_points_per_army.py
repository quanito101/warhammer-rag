from project_rag.parser.docling_parser import parse_pdf

doc = parse_pdf("data/raw/Core Rules.pdf")

p55_tables = [t for t in doc.tables if t.prov and t.prov[0].page_no == 55]
print(f"Tables on page 55: {len(p55_tables)}")
for i, t in enumerate(p55_tables):
    grid = t.data.grid
    print(f"\n--- Table {i} ({len(grid)} rows) ---")
    for row in grid[:4]:
        print(" | ".join(c.text for c in row))
