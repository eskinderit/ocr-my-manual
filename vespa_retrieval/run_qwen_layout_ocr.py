"""OCR fresh detector regions with Qwen; checkpoint raw evidence and preserve figures."""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image

from .ocr_backends import infer
from .table_html import clean_table_html
from .pipeline_utils import completion_text, image_sort_key, normalize_bbox, prepare_retry_image, write_document, write_json


TEXT_PROMPT = "Transcribe all visible lettering, including single letters inside emblems, stylized words, dates and numbers. Preserve the original language and spelling. Output only visible text. Do not describe the image. If the content is tabular, output a complete HTML table preserving rows, columns and all cells. Do not invent text."
TABLE_PROMPT = "Transcribe this table exactly in its original language as a complete HTML table. Preserve row and column relationships, empty cells, ditto marks, numbers and units. Use rowspan and colspan where needed. Use only table, tr, th and td tags; no style attributes, CSS, explanations or code fences. Keep markup compact."


def run(root, imgs, compact_tables=False, *, endpoint=None, maximum_side=1800):
    root.mkdir(parents=True, exist_ok=True)
    detections = json.loads((root / 'layouts.json').read_text())
    combined_path = root / 'layouts_with_texts.json'
    combined = json.loads(combined_path.read_text()) if combined_path.exists() else {}
    audit_path = root / 'ocr_audit.json'
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else {'backend': 'qwen3vl4b', 'records': []}
    errors = 0
    for name in sorted(detections, key=image_sort_key):
        if name in combined and not any(el.get('ocr_error') for el in combined[name]['elements']):
            print('Resume: already complete', name, flush=True)
            continue
        boxes = detections[name]
        if not isinstance(boxes, list):
            raise ValueError(f'Detector failed for {name}: {boxes}')
        elements = []
        with Image.open(imgs / name) as original:
            image = original.convert('RGB')
        for index, box in enumerate(sorted(boxes, key=lambda e: (e['bbox'][1], e['bbox'][0]))):
            bbox = normalize_bbox(box['bbox'], image.width, image.height)
            crop = Path('crops') / Path(name).stem / f'{Path(name).stem}_{index}.png'
            crop_path = root / crop
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            image.crop(tuple(bbox)).save(crop_path)
            el = dict(box, bbox=bbox, crop=crop.as_posix(), text='')
            elements.append(el)
            if el['label'].lower() in ('figure', 'picture', 'image'):
                continue
            prompt = TABLE_PROMPT if el['label'].lower() == 'table' else TEXT_PROMPT
            request_image = crop_path
            if compact_tables and el['label'].lower() == 'table':
                # Keep the original crop intact; reduce visual context only for retry.
                request_image = root / 'retry_inputs' / crop_path.name
                transform = prepare_retry_image(crop_path, request_image, maximum_side)
                prompt = "Read this table into a compact HTML table with the same columns. Put related cause and remedy text in the same row. Use only table, tr, th, td and br tags. No rowspan, colspan, styles, repeated empty rows or explanations. Write all visible text in Italian, then close the table and stop. Empty execution cells should remain empty."
                el['ocr_image_transform'] = dict(transform, input=str(request_image.relative_to(root)))
            print('OCR', name, index, el['label'], flush=True)
            try:
                result = infer(request_image, 'qwen3vl4b', endpoint=endpoint, prompt=prompt, max_tokens=6144 if el['label'].lower() == 'table' else 4096, timeout=600)
                audit['records'].append({'page':name,'element':index,'crop_sha256':hashlib.sha256(crop_path.read_bytes()).hexdigest(),'result':result})
                text = completion_text(result)
                if '<table' in text.lower():
                    table = clean_table_html(text)
                    if '</table>' not in table.lower():
                        raise ValueError('Incomplete HTML table')
                    el['table'] = {'raw_text':text, 'html_inline':table}
                elif el['label'].lower() == 'table':
                    raise ValueError('Table response is not a complete HTML table')
                else:
                    el['text'] = text
            except Exception as exc:
                el['ocr_error'] = str(exc)
                el['text'] = '[OCR failed — inspect original crop]'
                errors += 1
                print('FAILED', name, index, str(exc), flush=True)
        doc = root / 'layout_documents' / f'{Path(name).stem}.md'
        write_document(doc, elements)
        combined[name] = {'elements':elements,'document_markdown':str(doc)}
        write_json(combined_path, combined)
        write_json(audit_path, audit)
    print('Completed',len(combined),'pages; request failures',errors,flush=True)
    return errors


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('outputs/qwen_pipeline'))
    parser.add_argument('--imgs',type=Path,default=Path('imgs'))
    parser.add_argument('--compact-tables',action='store_true',help='Retry failed table pages using smaller visual context and no merged cells')
    parser.add_argument('--endpoint',help='Override the full chat-completions URL')
    parser.add_argument('--maximum-side',type=int,default=1800,help='Maximum retry image side')
    args = parser.parse_args()
    raise SystemExit(1 if run(args.root,args.imgs,args.compact_tables,endpoint=args.endpoint,maximum_side=args.maximum_side) else 0)