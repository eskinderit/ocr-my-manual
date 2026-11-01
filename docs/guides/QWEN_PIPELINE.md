# Fresh layout → Qwen OCR pipeline

> Historical run notes. Paths and script names below describe that run before
> reorganization; use the [current README](../../README.md) and
> [entry point](../../main.py) for new PDFs/photo folders. Existing result data
> remains unchanged.

This run uses the original 36 scans with fresh DocLayout-YOLO detection, then Qwen3-VL-4B-Instruct Q8_0 OCR. It does not reuse baseline OCR text or Tesseract results.

- Detector: DocStructBench checkpoint, 1024 inference size, confidence 0.25, confidence-ranked IoU deduplication 0.20, CPU inference.
- OCR: every non-figure region, including titles and detector `abandon` regions. Figures remain unmodified image crops. Tables are requested as HTML with row/column relationships.
- Decoding: temperature 0, top-p 1, seed 42, maximum 4096 text / 6144 table output tokens. Incomplete/empty responses are failures, not successful output.
- Completed: 36 pages, 222 OCR regions (including six tables), 17 preserved figure crops, zero final request failures. The [results webpage](../qwen_pipeline/index.html) has been opened and checked in the browser.
- Outputs and raw responses: [outputs/qwen_pipeline/layouts_with_texts.json](../../outputs/qwen_pipeline/layouts_with_texts.json), [outputs/qwen_pipeline/ocr_audit.json](../../outputs/qwen_pipeline/ocr_audit.json), [outputs/qwen_pipeline/run_manifest.json](../../outputs/qwen_pipeline/run_manifest.json).
- The existing baseline remains untouched. An isolated site is built from this output and its own crops.

## Reproduction

Use the existing OCR-trial environment for the detector and image processing. The base environment's torch installation is broken; no new environment or weights were installed for this run.

1. Run the existing detector with `--model juliozhao/DocLayout-YOLO-DocStructBench --imgs imgs --out outputs/qwen_pipeline/layouts`.
2. Serve the local Qwen Q8_0 decoder and f16 projector with the existing native llama-server on port 8003, alias `qwen3vl4b`, 8192 context, one slot, CUDA offload and flash attention.
3. Run [run_qwen_layout_ocr.py](../../vespa_retrieval/run_qwen_layout_ocr.py). It checkpoints each page and skips already-complete pages on restart. Use a different `--root` directory for a new independent run.
4. Build with [build_ui.py](../../vespa_retrieval/build_ui.py), selecting the new combined source, `--asset-root outputs/qwen_pipeline`, a separate `--site-dir`, and `--show-abandon`. This keeps transcribed cover subtitles and footers visible.

## Limits

The fresh title crop correctly reads **Vespa 150**. However, the detected emblem crop's OCR omits the small **P** that a previous full-cover Qwen prompt recovered. Successful completion does not guarantee perfect OCR. Tables, small numerals, and reading order still need visual review against the original panels.

The large troubleshooting table on source page 32 exceeded the context budget and looped on merged HTML rows. It was retried with an explicitly recorded Lanczos reduction to maximum side 1800 and a no-merged-cells prompt; the unchanged crop is retained. Its resulting three-column HTML is complete but simplified, grouping text into a single body row and omitting the brace annotation. It is not an exact table-structure reconstruction. A heading on that page also visibly contains an OCR spelling/case error; no manual corrections were substituted into the model output.

## Later-table rendering repair

The original OCR on viewer pages 33–36 had inconsistent three/four-cell rows and grouped remedies into unrelated rows. All four tables were reread with Qwen using explicit three-column constraints and maximum-side-2600 retry inputs. [repair_qwen_tables.py](../../vespa_retrieval/repair_qwen_tables.py) rejects spans, missing closing tags and rows without exactly three cells before replacing a table. The replacements have 22, 23, 10 and 7 rows including headers. [outputs/qwen_pipeline/table_repairs.json](../../outputs/qwen_pipeline/table_repairs.json) archives previous tables, new raw responses, input hashes and transforms.

The viewer also had nearly invisible one-source-pixel table rules at 6% fit zoom, and forced equal-height rows with hidden overflow. Grid rules now scale to source dimensions and natural row sizing avoids clipping. Column widths survive HTML cleaning. All four pages were checked in the real browser: one table per page, three cells per row and zero overflowing cells.

This fixes rendering and improves structural alignment, not every transcription. Some section entries remain grouped and spelling or omitted words still require image review. The original scans and table crops are unchanged.