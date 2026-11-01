"""Build image-backed OCR evidence and a readable report from saved benchmark results."""

import html
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / 'outputs/ocr_benchmark'
DEST = ROOT / 'docs/ocr_evaluation'
NAMES = {'existing_dots_output': 'Existing dots output', 'lighton3': 'LightOnOCR-3-1B Q8_0', 'paddle16': 'PaddleOCR-VL-1.6 BF16'}


def main():
    result = json.loads((SOURCE / 'results.json').read_text())
    DEST.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE / 'results.json', DEST / 'results.json')
    (DEST / 'model_manifest.json').write_text(json.dumps(result['models'], ensure_ascii=False, indent=2) + '\n')
    for raw in SOURCE.glob('lighton3-*.json'):
        shutil.copy2(raw, DEST / raw.name)
    for raw in SOURCE.glob('paddle16-*.json'):
        shutil.copy2(raw, DEST / raw.name)
    metrics = []
    for key, summary in result['summary'].items():
        latency = 'not measured' if summary['median_seconds'] is None else f"{summary['median_seconds']:.3f} s"
        metrics.append(f"<tr><td>{NAMES[key]}</td><td>{summary['cer']:.2%}</td><td>{summary['content_cer']:.2%}</td><td>{summary['wer']:.2%}</td><td>{summary['table_cell_accuracy']:.0%}</td><td>{summary['numeric_crops_exact']}/5</td><td>{latency}</td></tr>")
    cards = []
    for sample in result['reference']['samples']:
        records = {key: next(r for r in rows if r['id'] == sample['id']) for key, rows in result['records'].items()}
        crop = ROOT / 'outputs' / records['existing_dots_output']['crop']
        image_name = sample['id'] + '.png'
        shutil.copy2(crop, DEST / image_name)
        reference = sample.get('reference') or '\n'.join(' | '.join(row) for row in sample['rows'])
        columns = [f"<article><h3>Image-verified reference</h3><pre>{html.escape(reference)}</pre></article>"]
        for key, record in records.items():
            scores = record['scores']
            if sample['kind'] == 'table':
                label = f"Exact positioned cells: {scores['cell_matches']}/{scores['cell_count']} · table text CER: {scores['text_content_cer']:.2%}"
            else:
                label = f"Strict CER: {scores['cer']:.2%} · content CER: {scores['content_cer']:.2%}"
            raw_link = '' if key == 'existing_dots_output' else f'<a href="{key}-{sample["id"]}.json">Raw response + timings</a>'
            columns.append(f"<article><h3>{NAMES[key]}</h3><p class='metric'>{label}</p><pre>{html.escape(record['text'])}</pre>{raw_link}</article>")
        cards.append(f"<section><h2>{html.escape(sample['id'].replace('_', ' ').title())}</h2><p>{html.escape(sample['page'])} · {sample['kind']} · SHA-256 {records['existing_dots_output']['image_sha256']}</p><a href='{image_name}'><img loading='lazy' src='{image_name}' alt='Original crop for {sample['id']}'></a><p>{html.escape(sample.get('notes', ''))}</p><div class='comparison'>{''.join(columns)}</div></section>")
    document = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Vespa · OCR evidence</title><style>
    :root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#0d1013;color:#dce2e9;font:14px system-ui,sans-serif;line-height:1.6}main{max-width:1600px;margin:auto;padding:32px}h1,h2,h3{line-height:1.3}h1{font-size:36px}h2{color:#f0b184}h3{font-size:14px}a{color:#85bdec}a:focus-visible{outline:2px solid #f0b184}section,.intro{background:#171b20;border:1px solid #30363d;border-radius:10px;padding:24px;margin:24px 0}section>p{font-size:12px;color:#a8b4c1;overflow-wrap:anywhere}img{display:block;max-width:100%;max-height:460px;width:auto;background:white;margin:20px 0;border-radius:6px}.comparison{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}article{background:#11171d;border:1px solid #30363d;border-radius:6px;padding:14px;min-width:0}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.7 ui-monospace,monospace;max-height:520px;overflow:auto}.metric{font-size:11px;color:#f0b184}table{width:100%;border-collapse:collapse;font-size:12px}th,td{text-align:left;padding:12px;border-bottom:1px solid #30363d}.scroll{overflow:auto}.warning{border-left:3px solid #f0b184;padding-left:16px}@media(max-width:1100px){.comparison{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:650px){main{padding:14px}.comparison{grid-template-columns:1fr}section,.intro{padding:16px}h1{font-size:27px}}</style></head><body><main>
    <p><a href="../index.html">← Document studio</a> · <a href="../ocr_candidate/index.html">Hybrid candidate preview</a> · <a href="results.json">Full scored evidence</a> · <a href="model_manifest.json">Weights, hashes and runtime</a></p>
    <h1>Better OCR, measured on your manual.</h1><p>Local paired experiment · 31 October 2026 · no cloud inference</p>
    <div class="intro"><h2>Decision: LightOn for text, preserve existing tables</h2><p>LightOnOCR-3-1B reduces text content CER from 2.40% to 0.36% on this diagnostic sample. Its table words are correct, but it loses structured cells. Paddle preserves dimensions but drops ditto cells and shifts values. Neither new model is reliable for the logo or every figure number.</p><p class="warning">This is a small, purposively selected diagnostic set, not a random accuracy estimate. The existing baseline is saved pipeline output, not a freshly rerun dots model. Nine ordinary text/caption/specification crops, one logo and two tables are judged separately. The logo and figure-number failure remain visible below.</p>
    <h3>How this was judged</h3><p>References were transcribed directly from images before generation; three difficult references received an independent image-only second review. Identical PNG bytes were sent to both models at temperature 0, seed 42, 4096-token output budget. LightOn received image-only prompts; Paddle received OCR: or Table Recognition:. The servers apply their own documented vision preprocessing.</p><p>Strict CER is Levenshtein character distance/reference length after removing transport markup and line-wrap hyphenation. Content CER also case-folds and ignores punctuation. WER uses content-normalized words. Decimal/unit numbers are checked separately. Table score requires the right text in the right cell, with extra cells penalized; HTML, Markdown and unmerged Paddle OTSL are parsed without inventing omitted cells. This is not TEDS or a page-layout benchmark.</p><p>Figure 16 in the carter reference is visually read as 16 by two image reviewers; all OCR outputs say 15. Treat that digit as a review item. With that crop excluded, the text ranking is unchanged. The logo reference ignores its decorative flourish.</p>
    <div class="scroll"><table><thead><tr><th>Engine</th><th>Strict text CER ↓</th><th>Content CER ↓</th><th>WER ↓</th><th>Exact table cells ↑</th><th>Numeric crops ↑</th><th>Median crop latency</th></tr></thead><tbody>__METRICS__</tbody></table></div>
    <p>Character metrics cover ten non-table crops (1,670 content characters); tables cover 50 cells. Latency is local HTTP + server-side vision encoding + generation, excluding model loading and client payload preparation. One pass, mixed prompt-cache state, not a cold-start throughput study. GPU: RTX A5000 Laptop 16 GB; native llama.cpp build 124/e71b805. LightOn uses a community Q8_0 conversion with BF16 projector; Paddle uses official BF16 GGUF.</p>
    <h3>Whole-manual candidate, not certified ground truth</h3><p>155 eligible text blocks were re-OCR'd across all 36 pages with no request failures, about 76 seconds of summed request time. 84 records were protected/skipped, including tables, figures, abandon and titles. Existing tables and the source JSON are unchanged. This preview is an unreviewed candidate, not an automatic production replacement. A further caption spot-check still changes “della” to “delle” (page 30): newer is not universally correct.</p></div>
    __CARDS__</main></body></html>"""
    (DEST / 'index.html').write_text(document.replace('__METRICS__', ''.join(metrics)).replace('__CARDS__', ''.join(cards)), encoding='utf-8')
    print('OCR evidence written to', DEST / 'index.html')


if __name__ == '__main__':
    main()