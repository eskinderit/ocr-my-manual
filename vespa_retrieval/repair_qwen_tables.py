"""Repair late troubleshooting tables with validated three-column Qwen OCR."""
import argparse
import hashlib
import json
from pathlib import Path

from .table_html import clean_table_html, validate_table
from .pipeline_utils import completion_text, prepare_retry_image, write_document, write_json
from .ocr_backends import infer


PROMPT = "This Italian troubleshooting table has exactly THREE columns: Ricerca e individuazione dell'inconveniente, Provvedimenti, Esecuzione. Transcribe ALL visible text faithfully as compact HTML. Use exactly three th/td cells per row, no colspan or rowspan, no extra blank column. Make a separate row for each fault and align its remedy on the same row, as printed. Section headings may use their own row. Do NOT group all remedies in a section into one cell. Keep empty execution cells empty; put any agency note in the execution column. Use br for line breaks within cells. No styles, CSS, explanations or code fences. Close the table and stop."


def run(root, maximum_side=2600, force=False, *, pages=None, endpoint=None):
    combined_path=root/'layouts_with_texts.json'
    combined=json.loads(combined_path.read_text())
    report_path=root/'table_repairs.json'
    report=json.loads(report_path.read_text()) if report_path.exists() else {'method':'Qwen three-column troubleshooting retry','records':[]}
    page_names = list(pages) if pages is not None else [f'vespa_manual_{n}.png' for n in range(32,36)]
    missing = [name for name in page_names if name not in combined or not any(e.get('label') == 'table' for e in combined[name]['elements'])]
    if missing:
        raise ValueError(f'Pages missing a table: {missing}')
    for name in page_names:
        el=next(e for e in combined[name]['elements'] if e['label']=='table')
        if el.get('table_repair') and not force:
            print('Already repaired',name,flush=True); continue
        crop=root/el['crop']
        retry=root/'retry_inputs'/f'{crop.stem}_three_columns.png'
        retry.parent.mkdir(parents=True,exist_ok=True)
        transform = prepare_retry_image(crop, retry, maximum_side)
        print('Repair OCR',name,flush=True)
        result=infer(retry,'qwen3vl4b',endpoint=endpoint,prompt=PROMPT,max_tokens=6144,timeout=600)
        record={'page':name,'previous_table':el.get('table'),'crop_sha256':hashlib.sha256(crop.read_bytes()).hexdigest(),'request_image_sha256':hashlib.sha256(retry.read_bytes()).hexdigest(),'image_transform':transform,'result':result}
        report['records'].append(record)
        write_json(report_path, report)
        html=clean_table_html(completion_text(result))
        row_count=validate_table(html)
        # These printed troubleshooting tables have a wide cause/remedy pair and narrow execution column.
        html=html.replace('<table border="1">','<table border="1"><colgroup><col style="width:43%"><col style="width:42%"><col style="width:15%"></colgroup>',1)
        el['table']={'raw_text':result['text'],'html_inline':html}
        el['table_repair']={'rows':row_count,'columns':3,'evidence':'table_repairs.json','input':str(retry.relative_to(root))}
        el['ocr_image_transform']=record['image_transform']
        el.pop('ocr_error',None)
        document = root / 'layout_documents' / f'{Path(name).stem}.md'
        combined[name]['document_markdown'] = str(document)
        write_json(combined_path, combined)
        write_document(document, combined[name]['elements'])
        print('Validated',name,row_count,'rows / 3 columns',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('outputs/qwen_pipeline'))
    parser.add_argument('--maximum-side',type=int,default=2600)
    parser.add_argument('--force',action='store_true')
    parser.add_argument('--pages',nargs='+',help='Exact layout image keys to repair; default: existing Vespa pages 33–36')
    parser.add_argument('--endpoint',help='Override the full chat-completions URL')
    args=parser.parse_args()
    run(args.root,args.maximum_side,args.force,pages=args.pages,endpoint=args.endpoint)