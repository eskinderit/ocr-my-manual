"""Small shared helpers; no OCR server or detector is loaded on import."""
import json
import re
from pathlib import Path

from PIL import Image


def image_sort_key(name: str) -> tuple:
    """Order numbered pages numerically, with a safe fallback for any filename."""
    match = re.search(r'(\d+)$', Path(name).stem)
    return (int(match.group(1)) if match else float('inf'), name.casefold())


def normalize_bbox(bbox, width: int, height: int) -> list[int]:
    if len(bbox) != 4:
        raise ValueError('A bounding box requires four coordinates')
    x1, y1, x2, y2 = (int(round(value)) for value in bbox)
    box = [max(0, x1), max(0, y1), min(width, x2), min(height, y2)]
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f'Invalid detected crop: {box}')
    return box


def prepare_retry_image(source: Path, destination: Path, maximum_side: int) -> dict:
    """Save a transformed copy, never replacing the original crop."""
    if maximum_side < 1:
        raise ValueError('Maximum image side must be positive')
    if source.resolve() == destination.resolve():
        raise ValueError('Retry image must not overwrite its source')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as original:
        image = original.convert('RGB')
        image.thumbnail((maximum_side, maximum_side), Image.Resampling.LANCZOS)
        image.save(destination)
    return {'maximum_side': maximum_side, 'method': 'Lanczos'}


def completion_text(result: dict) -> str:
    if result['response']['choices'][0]['finish_reason'] != 'stop':
        raise ValueError('Incomplete/truncated OCR completion')
    text = result['text'].strip()
    if not text:
        raise ValueError('Empty OCR response')
    return text


def write_json(path: Path, value) -> None:
    """Replace a checkpoint atomically so interruption cannot leave partial JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_document(path: Path, elements: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = '\n\n'.join(element.get('table', {}).get('html_inline') or element.get('text', '') for element in elements)
    path.write_text(text, encoding='utf-8')