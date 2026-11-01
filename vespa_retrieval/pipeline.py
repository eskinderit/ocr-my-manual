"""Full input → fresh layout → Qwen OCR → self-contained site orchestration."""
import json
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path

from .build_ui import build
from .inputs import prepare_inputs
from .pipeline_utils import write_json
from .run_qwen_layout_ocr import run as run_ocr
from .server import local_server


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def detect(scans: Path, run_dir: Path, model: str):
    log_path = run_dir / 'logs' / 'layout.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('w', encoding='utf-8') as log:
        subprocess.run([sys.executable, '-m', 'vespa_retrieval.detect_layout', '--model', model,
                        '--imgs', str(scans), '--out', str(run_dir / 'layouts')],
                       cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    metadata = run_dir / 'layouts.json'
    if not metadata.exists():
        raise RuntimeError(f'Layout detection produced no metadata; inspect {log_path}')
    results = json.loads(metadata.read_text(encoding='utf-8'))
    expected = {p.name for p in scans.glob('*.png')}
    if set(results) != expected or any(not isinstance(boxes, list) or not boxes for boxes in results.values()):
        raise RuntimeError(f'Incomplete or empty layout detections; inspect {log_path}')


def run_pipeline(source: Path, output: Path, *, dpi=200,
                 detector_model='juliozhao/DocLayout-YOLO-DocStructBench', endpoint=None,
                 server_binary=None, ocr_model=None, projector=None, port=8003):
    source = source.resolve()
    output = output.resolve()
    if source.is_dir() and (output == source or source in output.parents):
        raise ValueError('Output must not be inside the photo input folder')
    # Never overwrite or silently resume data from another manual/settings.
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'input': str(source), 'output': str(output), 'status': 'preparing', 'dpi': dpi,
                'detector_model': detector_model, 'ocr_model_alias': 'qwen3vl4b'}
    manifest_path = output / 'pipeline_manifest.json'
    try:
        scans = output / 'originals'
        records = prepare_inputs(source, scans, dpi)
        manifest.update(status='detecting', pages=len(records))
        write_json(manifest_path, manifest)
        print(f'Detecting layout on {len(records)} pages…', flush=True)
        detect(scans, output, detector_model)
        manager = nullcontext(endpoint) if endpoint else local_server(
            server_binary or PROJECT_ROOT / 'vendor/llama.cpp/build/bin/llama-server',
            ocr_model or PROJECT_ROOT / 'models/logo_ocr/Qwen3-VL-4B-Instruct.Q8_0.gguf',
            projector or PROJECT_ROOT / 'models/logo_ocr/Qwen3-VL-4B-Instruct.mmproj-f16.gguf',
            output / 'logs' / 'ocr_server.log', port)
        manifest['status'] = 'ocr'
        write_json(manifest_path, manifest)
        with manager as selected_endpoint:
            failures = run_ocr(output, scans, endpoint=selected_endpoint)
        if failures:
            raise RuntimeError(f'{failures} OCR regions failed; checkpoints are preserved in {output}')
        site = output / 'site'
        build(output / 'layouts_with_texts.json', site, output, True, imgs_dir=scans)
        manifest.update(status='complete', site=str(site / 'index.html'))
        write_json(manifest_path, manifest)
        return site / 'index.html'
    except BaseException as exc:
        manifest.update(status='failed', error=str(exc))
        write_json(manifest_path, manifest)
        raise