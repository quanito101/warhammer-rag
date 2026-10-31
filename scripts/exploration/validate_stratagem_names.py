from project_rag.parser.docling_parser import parse_pdf
from project_rag.parser.icon_fixups import fix_stratagem_cp_cost

doc = parse_pdf("data/raw/Core Rules.pdf")

matched = []
unmatched_headers = []

for item in doc.texts:
    if item.label != "section_header":
        continue
    if not item.prov or item.prov[0].page_no not in (41, 42):
        continue
    result = fix_stratagem_cp_cost(item.text)
    if result.applied:
        matched.append((item.text, result.value))
    else:
        unmatched_headers.append(item.text)

print(f"Matched {len(matched)} stratagem headers to a known CP cost:")
for name, cp in matched:
    print(f"  {name}: {cp}")

print()
print(f"Section headers on pp. 41-42 NOT matched (expected: page/key headers, not stratagems):")
for name in unmatched_headers:
    print(f"  {name!r}")
