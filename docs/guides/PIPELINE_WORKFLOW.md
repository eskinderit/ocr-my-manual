# Pipeline workflow

## Current entry point and organization

Use [main.py](../../main.py) for the complete pipeline. It accepts a PDF or a
photo folder, stages numbered copies in input order, runs fresh layout detection
and Qwen OCR, and builds an independent site. Photos use alphabetical filename
order, not natural numeric ordering. See the [README](../../README.md) for commands,
dependencies and output structure.

Runtime scripts now live in [vespa_retrieval](../../vespa_retrieval/__init__.py)
and run with `python -m vespa_retrieval.<module>`. The viewer template is under
the package templates directory. Optional evaluation/legacy scripts live in
[tools](../../tools/__init__.py) and run with `python -m tools.<module>`.
Detailed run notes have moved into this guides directory.

The sections below describe the earlier incremental refactor and preserve its
history; root-script paths there are superseded by the package layout above.

This refactor separates document processing from the static website. It does not
add uploads, a web backend, or deployment configuration. Existing entry points and
defaults remain compatible, and existing OCR results are not regenerated.

## Components

| Component | Responsibility |
| --- | --- |
| [detect_layout.py](../../vespa_retrieval/detect_layout.py) | DocLayout-YOLO detection and layout metadata |
| [ocr_backends.py](../../vespa_retrieval/ocr_backends.py) | HTTP requests to an already-running model server |
| [run_qwen_layout_ocr.py](../../vespa_retrieval/run_qwen_layout_ocr.py) | Detected-region OCR, figure preservation and page checkpoints |
| [repair_qwen_tables.py](../../vespa_retrieval/repair_qwen_tables.py) | Optional, specialized Italian three-column table retries |
| [table_html.py](../../vespa_retrieval/table_html.py) | Shared table extraction, column-width preservation and structure checks |
| [pipeline_utils.py](../../vespa_retrieval/pipeline_utils.py) | Bounding boxes, page ordering, retry copies, completion checks and atomic JSON writes |
| [build_ui.py](../../vespa_retrieval/build_ui.py) | Static viewer generation and relative asset copies |
| [ui_template.html](../../vespa_retrieval/templates/ui_template.html) | Browser UI, independent of the model server |

`clean_table_html` remains importable from the builder and `validate_table` from
the repair script for compatibility. Their implementations now live in the shared
table module. OCR scripts no longer import the website builder.

## Configurable inputs

Run the steps from the project root. Relative paths are resolved from the current
working directory; absolute paths are also supported. Use a new output directory
for each independent manual/run.

1. **Detect:** `--imgs` selects scan images; `--out` selects the layouts directory.
   The detector writes combined metadata one directory above its layouts directory.
2. **OCR:** `--root` selects that run directory, `--imgs` selects the same scan
   directory, and `--endpoint` overrides the full chat-completions URL. The model
   server must already be running with the `qwen3vl4b` alias. Completed pages are
   skipped on restart; changing inputs/settings requires a fresh run directory.
3. **Optional table retry:** `--root`, `--pages` (exact image keys from combined
   JSON), `--maximum-side`, `--endpoint`, and `--force` are available. The default
   page selection preserves the current Vespa pages 33–36. This prompt is specific
   to Italian troubleshooting tables with three columns; do not apply it to all
   table types. Previous responses are archived before replacement.
4. **Build:** `--source` selects combined OCR JSON, `--asset-root` selects its crop
   and table root, `--imgs` selects source scans, and `--site-dir` selects the static
   destination. `--show-abandon` includes transcribed footers and cover subtitles by
   default. Building requires no GPU, model weights, detector or OCR server.

Compact-table OCR retries use `--compact-tables` and `--maximum-side` (default
1800); specialized table repairs default to 2600. Retry images are transformed
copies, never replacements for the original crops. JSON checkpoints are replaced
atomically; only one process should write a run directory at a time.

## Manual publishing later

The generated site directory contains the viewer plus `originals/`, `crops/`, and
any table assets using relative paths. Keep those files together when copying it
to a static host. Do not copy model weights, environments, vendor tools or raw OCR
audit files into the public site. Source scans are included in the generated site;
only publish documents you intend to make public.

The current generated viewer remains [docs/qwen_pipeline/index.html](../qwen_pipeline/index.html).
Deployment and hosting are deliberately left to you.

## Verification

Tests use temporary images, mocked OCR and a browser DOM harness. They do not
launch the model or change existing manual outputs. Coverage includes custom scan
directories/endpoints, unnumbered filenames, source-preserving transforms,
interrupted checkpoint writes, truncated completions and table column structure.
Structural checks are not an OCR accuracy guarantee.