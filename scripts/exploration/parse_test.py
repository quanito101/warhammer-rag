from docling.document_converter import DocumentConverter

converter = DocumentConverter()
result = converter.convert("data/raw/Core Rules.pdf")
doc = result.document

markdown = doc.export_to_markdown()
with open("data/processed/core_rules_raw.md", "w") as f:
    f.write(markdown)