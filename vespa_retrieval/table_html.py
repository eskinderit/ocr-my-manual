"""Table cleanup and validation shared by OCR processing and static builds.

Structural validation does not establish OCR transcription accuracy.
"""
import re
from html.parser import HTMLParser


def clean_table_html(text: str) -> str:
    """Extract table fragments, preserving the first header and column widths."""
    if not text:
        return text
    blocks = re.findall(r'<table[^>]*>.*?</table>', text, re.DOTALL | re.IGNORECASE)
    if not blocks:
        return text
    combined = ''.join(blocks)
    header = re.search(r'<thead>.*?</thead>', combined, re.DOTALL | re.IGNORECASE)
    columns = re.search(r'<colgroup[^>]*>.*?</colgroup>', combined, re.DOTALL | re.IGNORECASE)
    body = re.sub(r'<thead>.*?</thead>', '', combined, flags=re.DOTALL | re.IGNORECASE)
    rows = re.findall(r'<tr>.*?</tr>', body, re.DOTALL | re.IGNORECASE)
    return '<table border="1">' + (columns.group(0) if columns else '') + (header.group(0) if header else '') + ''.join(rows) + '</table>'


class TableShape(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows = []
        self.current = None
        self.spans = False

    def handle_starttag(self, tag, attrs):
        if tag == 'tr':
            self.current = 0
        if tag in ('th', 'td') and self.current is not None:
            self.current += 1
            if any(key in ('rowspan', 'colspan') and value != '1' for key, value in attrs):
                self.spans = True

    def handle_endtag(self, tag):
        if tag == 'tr' and self.current is not None:
            self.rows.append(self.current)
            self.current = None


def validate_table(html: str, columns: int = 3) -> int:
    """Require a closed table and a fixed number of unmerged cells per row."""
    if columns < 1:
        raise ValueError('Column count must be positive')
    shape = TableShape()
    shape.feed(html)
    if '</table>' not in html.lower() or not shape.rows or any(n != columns for n in shape.rows) or shape.spans:
        raise ValueError(f'Invalid {columns}-column table: rows={shape.rows}, spans={shape.spans}')
    return len(shape.rows)