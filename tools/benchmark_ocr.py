"""Reproducible paired crop OCR evaluation; never modifies production transcriptions."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import statistics
import unicodedata
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

from vespa_retrieval.ocr_backends import BACKENDS, infer

ROOT = Path(__file__).resolve().parent.parent


class TableParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows = []
        self.row = None
        self.cell = None

    def handle_starttag(self, tag, attrs):
        if tag == 'tr':
            self.row = []
        elif tag in ('td', 'th') and self.row is not None:
            self.cell = []
        elif tag == 'br' and self.cell is not None:
            self.cell.append(' ')

    def handle_data(self, value):
        if self.cell is not None:
            self.cell.append(value)

    def handle_endtag(self, tag):
        if tag in ('td', 'th') and self.cell is not None:
            self.row.append(''.join(self.cell).strip())
            self.cell = None
        elif tag == 'tr' and self.row is not None:
            self.rows.append(self.row)
            self.row = None


def table_rows(text):
    """Parse HTML, Markdown and unmerged Paddle OTSL; never invent missing cells."""
    if '<fcel>' in text:
        if any(token in text for token in ('<lcel>', '<ucel>', '<xcel>')):
            return []  # Merged cells require span-aware scoring, not guessed flattening.
        rows = []
        for line in text.split('<nl>'):
            tokens = re.split(r'(<fcel>|<ecel>)', line)
            row = []
            for i in range(1, len(tokens), 2):
                row.append(tokens[i+1].strip() if tokens[i] == '<fcel>' else '')
            if row:
                rows.append(row)
        return rows
    parser = TableParser()
    parser.feed(text)
    if parser.rows:
        return parser.rows
    rows = []
    for line in text.splitlines():
        if '|' not in line:
            continue
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        if all(re.fullmatch(r':?-{2,}:?', c.replace(' ', '')) for c in cells):
            continue
        rows.append(cells)
    return rows


def normalize(text, content=False):
    """Ignore transport markup/line wrapping, not lexical substitutions or numbers."""
    text = html.unescape(re.sub(r'<[^>]+>', ' ', text))
    text = unicodedata.normalize('NFC', text).replace('’', "'")
    text = re.sub(r'(?<=\w)-\s*\n\s*(?=\w)', '', text)
    text = re.sub(r'(?m)^\s*#{1,6}\s+', '', text).replace('**', '').replace('`', '')
    if content:
        text = text.casefold()
        text = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in text)
    return ' '.join(text.split())


def distance(a, b):
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j-1] + (left != right)))
        previous = current
    return previous[-1]


def numbers(text):
    return re.findall(r'\d+(?:[.,/-]\d+)*', text)


def score(sample, text):
    if sample['kind'] == 'table':
        expected = sample['rows']
        actual = table_rows(text)
        matches = sum(normalize(c) == normalize(actual[i][j])
                      for i, row in enumerate(expected) for j, c in enumerate(row)
                      if i < len(actual) and j < len(actual[i]))
        count = sum(map(len, expected))
        predicted_count = sum(map(len, actual))
        reference_text = normalize(' '.join(c for row in expected for c in row), True)
        if actual:
            predicted_text = normalize(' '.join(c for row in actual for c in row), True)
        else:
            predicted_text = normalize(text, True)
        return {'cell_matches': matches, 'cell_count': count,
                'predicted_cell_count': predicted_count,
                'cell_accuracy': matches / max(count, predicted_count),
                'shape_exact': list(map(len, actual)) == list(map(len, expected)),
                'parsed_rows': actual, 'parse_success': bool(actual),
                'text_content_cer': distance(reference_text, predicted_text) / len(reference_text)}
    ref = normalize(sample['reference'])
    pred = normalize(text)
    cref = normalize(sample['reference'], True)
    cpred = normalize(text, True)
    result = {'char_errors': distance(ref, pred), 'char_count': len(ref),
              'content_char_errors': distance(cref, cpred), 'content_char_count': len(cref),
              'word_errors': distance(cref.split(), cpred.split()), 'word_count': len(cref.split())}
    result.update(cer=result['char_errors']/len(ref),
                  content_cer=result['content_char_errors']/len(cref),
                  wer=result['word_errors']/len(cref.split()))
    if 'numbers' in sample:
        expected = Counter(sample['numbers'])
        predicted = Counter(numbers(text))
        result['numbers_exact'] = expected == predicted
        result['numbers_expected'] = sample['numbers']
        result['numbers_predicted'] = numbers(text)
    return result


def summarize(records):
    text = [r['scores'] for r in records if 'cer' in r['scores']]
    tables = [r['scores'] for r in records if 'cell_accuracy' in r['scores']]
    result = {'samples': len(records), 'failures': sum('error' in r for r in records)}
    for key, num, den in [('cer', 'char_errors', 'char_count'),
                          ('content_cer', 'content_char_errors', 'content_char_count'),
                          ('wer', 'word_errors', 'word_count')]:
        result[key] = sum(s[num] for s in text) / sum(s[den] for s in text)
    result['macro_content_cer'] = statistics.mean(s['content_cer'] for s in text)
    result['table_cell_accuracy'] = sum(s['cell_matches'] for s in tables) / sum(max(s['cell_count'], s['predicted_cell_count']) for s in tables)
    result['table_shapes_exact'] = sum(s['shape_exact'] for s in tables)
    checked = [s for s in text if 'numbers_exact' in s]
    result['numeric_crops_exact'] = sum(s['numbers_exact'] for s in checked)
    result['numeric_crops_checked'] = len(checked)
    elapsed = [r['elapsed_seconds'] for r in records if 'elapsed_seconds' in r]
    result['median_seconds'] = statistics.median(elapsed) if elapsed else None
    return result


def run(args):
    references = json.loads(args.reference.read_text())
    layouts = json.loads((ROOT / 'outputs/layouts_with_texts.json').read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    result = {'reference': references, 'reference_sha256': hashlib.sha256(args.reference.read_bytes()).hexdigest(),
              'baseline_sha256': hashlib.sha256((ROOT / 'outputs/layouts_with_texts.json').read_bytes()).hexdigest(),
              'models': {}, 'records': {}}
    manifest = ROOT / 'models/ocr_benchmark/manifest.json'
    if manifest.exists():
        result['models'] = json.loads(manifest.read_text())
    for backend in ['existing_dots_output'] + args.backends:
        records = []
        for sample in references['samples']:
            element = layouts[sample['page']]['elements'][sample['element']]
            image = ROOT / 'outputs' / element['crop']
            record = {'id': sample['id'], 'kind': sample['kind'], 'crop': element['crop'],
                      'image_sha256': hashlib.sha256(image.read_bytes()).hexdigest()}
            if backend == 'existing_dots_output':
                record['text'] = element.get('text') or element.get('table', {}).get('raw_text', '')
            else:
                target = args.output / f"{backend}-{sample['id']}.json"
                prompt = 'Table Recognition:' if backend == 'paddle16' and sample['kind'] == 'table' else None
                try:
                    if args.rescore:
                        raw = json.loads(target.read_text())
                    else:
                        raw = infer(image, backend, prompt=prompt, max_tokens=4096)
                        target.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + '\n')
                    record.update(text=raw['text'], elapsed_seconds=raw['elapsed_seconds'],
                                  finish_reason=raw['response']['choices'][0]['finish_reason'])
                    if record['finish_reason'] != 'stop':
                        record['error'] = 'Generation truncated or did not stop normally'
                except Exception as exc:
                    record.update(text='', error=str(exc))
            record['scores'] = score(sample, record['text'])
            records.append(record)
            print(backend, sample['id'], json.dumps(record['scores'], ensure_ascii=False), flush=True)
        result['records'][backend] = records
        result.setdefault('summary', {})[backend] = summarize(records)
        (args.output / 'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=ROOT / 'tests/ocr_reference.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/ocr_benchmark')
    parser.add_argument('--rescore', action='store_true', help='Score saved raw responses without re-running models')
    parser.add_argument('--backends', nargs='+', choices=BACKENDS, default=['lighton3', 'paddle16'])
    run(parser.parse_args())