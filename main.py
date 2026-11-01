"""Run the complete manual pipeline on a PDF or alphabetically ordered photos."""
import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from vespa_retrieval.pipeline import run_pipeline


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog='ocr-my-manual', description=__doc__)
    parser.add_argument('input', type=Path, help='PDF file or folder of photos (alphabetical filename order)')
    parser.add_argument('--output', type=Path, help='New run directory; default: outputs/<input-name>-<timestamp>')
    parser.add_argument('--dpi', type=int, default=200, help='PDF render resolution, 72–600 (default: 200)')
    parser.add_argument('--detector-model', default='juliozhao/DocLayout-YOLO-DocStructBench', help='Local DocLayout-YOLO checkpoint or Hugging Face repo')
    parser.add_argument('--endpoint', help='Full chat-completions URL for an existing Qwen server; otherwise start the local server')
    parser.add_argument('--server-binary', type=Path, help='Path to llama-server')
    parser.add_argument('--ocr-model', type=Path, help='Path to Qwen3-VL decoder GGUF')
    parser.add_argument('--projector', type=Path, help='Path to the companion vision projector GGUF')
    parser.add_argument('--port', type=int, default=8003, help='Port for the automatically managed server')
    args = parser.parse_args(argv)
    output = args.output or Path('outputs') / f'{args.input.stem}-{datetime.now():%Y%m%d-%H%M%S-%f}'
    try:
        site = run_pipeline(args.input, output, dpi=args.dpi, detector_model=args.detector_model,
                            endpoint=args.endpoint, server_binary=args.server_binary,
                            ocr_model=args.ocr_model, projector=args.projector, port=args.port)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f'Pipeline failed: {exc}', file=sys.stderr)
        return 1
    print(f'Complete. Open {site}\nFor manual publishing, copy the entire site directory.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())