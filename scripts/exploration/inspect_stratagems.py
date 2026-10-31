from project_rag.parser.docling_parser import parse_pdf

doc = parse_pdf("data/raw/Core Rules.pdf")

for item in doc.texts:
    if not item.prov:
        continue
    page_no = item.prov[0].page_no
    if page_no not in (41, 42):
        continue
    preview = item.text[:80]
    print(f"page {page_no} | {item.label} | {preview!r}")