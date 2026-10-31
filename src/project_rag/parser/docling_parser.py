"""
Parses the Core Rules PDF with Docling and exposes the resulting
DoclingDocument, rather than only a markdown export.

Keeping the DoclingDocument (not just its markdown rendering) matters
because table cells need to be inspected and corrected row-by-row before
they are flattened to text -- see icon_fixups.py and
docs/parsing_and_chunking.md for why.
"""

from __future__ import annotations

from pathlib import Path

from docling.document_converter import DocumentConverter
from docling_core.types.doc.document import DoclingDocument


def parse_pdf(path: Path | str) -> DoclingDocument:
    """Parse a PDF at `path` and return its DoclingDocument.

    This does no chunking, table fixups, or markdown export -- callers
    are expected to work with the returned document directly (e.g. to
    inspect doc.tables) or pass it on to other parsing/chunking steps.
    """
    converter = DocumentConverter()
    result = converter.convert(str(path))
    return result.document


def export_markdown(doc: DoclingDocument, output_path: Path | str) -> None:
    """Write `doc`'s markdown rendering to `output_path`.

    This is a diagnostic/debugging aid (e.g. for eyeballing output
    against the source PDF), not part of the parsing pipeline itself.
    """
    markdown = doc.export_to_markdown()
    Path(output_path).write_text(markdown)