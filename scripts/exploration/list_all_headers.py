from project_rag.parser.docling_parser import parse_pdf

doc = parse_pdf("data/raw/Core Rules.pdf")

for item in doc.texts:
    if item.label != "section_header":
        continue
    page_no = item.prov[0].page_no if item.prov else None
    level = getattr(item, "level", None)
    print(f"page {page_no} | level {level} | {item.text!r}")
