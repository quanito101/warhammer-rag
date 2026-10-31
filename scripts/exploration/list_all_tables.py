from project_rag.parser.docling_parser import parse_pdf

doc = parse_pdf("data/raw/Core Rules.pdf")

for i, table in enumerate(doc.tables):
    page_no = table.prov[0].page_no
    title_cell = table.data.grid[0][0].text
    print(f"[{i}] page {page_no}: {title_cell!r} ({table.data.num_rows} rows x {table.data.num_cols} cols)")
