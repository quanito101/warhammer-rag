from project_rag.parser.docling_parser import parse_pdf

doc = parse_pdf("data/raw/Core Rules.pdf")

print(type(doc.tables))
print(len(doc.tables))
print(doc.tables[0])