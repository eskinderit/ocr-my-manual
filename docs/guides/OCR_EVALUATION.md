# OCR evaluation — 31 October 2026

> Historical evaluation of the sample Vespa manual. The decision below applies
> to this experiment, not the current Qwen pipeline. After reorganization,
> evaluation scripts live under tools and runtime modules under vespa_retrieval;
> reproduction commands below use the current paths from the repository root.
> Use the [current README](../../README.md) for new manuals.

## Decision

Use **LightOnOCR-3-1B for ordinary text/captions**, but **preserve the existing
table transcriptions and review logos, figure numbers and technical values**.
Do not replace the whole pipeline with either new VLM based on this experiment.

The [image-backed comparison](../ocr_evaluation/index.html) includes all twelve
source crops, image-verified references, every engine's uncorrected text, per-crop
scores and links to full raw responses. The
[machine-readable results](../ocr_evaluation/results.json) and
[model manifest](../ocr_evaluation/model_manifest.json) preserve hashes/settings.

| Metric | Existing saved dots output | LightOnOCR-3-1B Q8_0 | PaddleOCR-VL-1.6 BF16 |
|---|---:|---:|---:|
| Strict character error, ten non-table crops ↓ | 2.50% | **0.41%** | 2.04% |
| Content character error ↓ | 2.40% | **0.36%** | 1.50% |
| Content word error ↓ | 6.54% | **0.77%** | 6.92% |
| Exact text at the correct table-cell position ↑ | **49/50 (98%)** | 0/50 structured cells | 26/50 (52%) |
| Numeric crops entirely correct ↑ | 4/5 | 4/5 | 4/5 |
| Median local request latency | Not measured | **0.594 s** | 1.161 s |

LightOn's table **words were correct** (content CER 0% for both tables), but
its image-only transcription flattened the rows to plain text. Its zero
structured-cell score is **not** a zero text-recognition score. Paddle returned
OTSL, which was parsed explicitly; it omitted ditto marks, shifted cells and
duplicated values in the contents table. Its seven-row dimensions table scored
13/14 cells. Merely counting rows/columns would incorrectly call both Paddle
tables successful.

## Concrete evidence, including failures

- [Publication disclaimer crop](../ocr_evaluation/publication_disclaimer.png):
  existing text says “alla”, “quel”, “opportare”, “inpegarsi” and
  “aggiungevole”; the scan says “nella”, “qui”, “apportare”, “impegnarsi” and
  “aggiornare”. LightOn matches the entire 518-character normalized reference.
  Paddle still makes nine content-word errors.
- [Fuel-tap caption](../ocr_evaluation/fuel_tap_caption.png): existing
  “rubinello” becomes the correct **“rubinetto”** with both new engines.
- [Carter caption](../ocr_evaluation/oil_level_caption.png): both correct
  “cartoncino” to **“carter”**, but still output **15** where two image reviewers
  read **16**. The digit is a review item; the crop is not claimed perfect.
- [Cover logo](../ocr_evaluation/cover_logo.png): the scan reads “Vespa 150”;
  existing OCR says “Vasco da Gamas 1500”, LightOn says “Coral 150”, Paddle says
  “Lapa 150”. None is correct; titles are therefore protected in the candidate.
- Further candidate spot-check outside the frozen set: the page-30 caption reads
  “Schema della lubrificazione”; LightOn produces “Schema delle lubrificazione”.
  It improves other errors but is not universally correct.

## How the evaluation works

1. Select a **purposive diagnostic set** from the existing manual: nine ordinary
   text/caption/specification crops, one stylized logo, two tables. Include known
   failures and already-good controls. This is **not random, held-out, or an
   unbiased whole-manual accuracy estimate**.
2. Read the actual crop images and write references **before generating the
   benchmark outputs**. Preserve historical Italian spelling, case, accents,
   punctuation and numbers. A separate image-only reviewer independently checked
   the disclaimer, engine specifications and carter caption without seeing OCR.
  References and selection notes live in [tests/ocr_reference.json](../../tests/ocr_reference.json).
3. Submit the **same unchanged PNG bytes**, with no client resizing/denoising,
   to both native local engines. LightOn uses its documented image-only prompt;
   Paddle uses `OCR:` or `Table Recognition:`. Temperature 0, top-p 1, seed 42,
   4,096 output tokens, 8,192-token server context. Their native vision
   preprocessing budgets differ; this tests deployed configurations, not an
   equal-pixel compute experiment. All 24 responses finished normally.
4. Score Levenshtein edit distance/reference length. Strict CER strips transport
   markup, folds whitespace and joins line-wrap hyphens. Content CER additionally
   case-folds and ignores punctuation; WER uses the same content-normalized word
   tokens. **Accents remain significant.** Ten non-table crops contain 1,670
   content characters. These metrics exclude both tables, which are scored
   separately over 50 cells.
5. Check numeric strings separately so punctuation normalization cannot hide
   decimal, ratio or dimensional mistakes. Exact-cell table scoring checks text
   **and position**, penalizes extra cells, and supports HTML, Markdown and
   unmerged Paddle OTSL. Missing cells are not reconstructed with heuristics;
   merged-cell TEDS/layout quality is **not** measured here.
6. Preserve crop hashes, source/reference hashes, raw HTTP responses, output
   tokens, native timings and model revisions. Code:
  [benchmark_ocr.py](../../tools/benchmark_ocr.py), [ocr_backends.py](../../vespa_retrieval/ocr_backends.py).

The saved dots baseline was **not rerun**: its projector was unavailable locally
and no dots server was running. Its exact historic prompt, quantization and any
previous manual edits are not fully recoverable, so this is a comparison against
the **current project output**, not a controlled ranking of original model weights.

The logo is short and disproportionately affects macro-average CER; the main
table above uses corpus/micro averaging. Figure-number ambiguity and sample
selection limit the conclusions. Removing the carter crop does not change the
text ranking. No significance or whole-manual accuracy claim is made.

## Safe whole-manual candidate

[reocr_layouts.py](../../tools/reocr_layouts.py) generated
[outputs/ocr_candidates/lighton3/layouts_with_texts.json](../../outputs/ocr_candidates/lighton3/layouts_with_texts.json):
**155 eligible blocks succeeded, zero request failures, 84 records protected or
skipped**, across all 36 pages. Summed request time was approximately 76 seconds
(not including loading); this candidate used a 2,048-token budget. All existing
table records were verified unchanged. Source JSON was not modified.

The [candidate preview](../ocr_candidate/index.html) uses the redesigned UI.
It is **not yet fully manually reviewed** and does not replace
[docs/index.html](../index.html). Candidate audit includes original/new text,
raw responses, status and crop hashes. Protected titles retain their existing
errors; protection is not certification. **39/155** blocks differ after content
normalization; the other **116** changes are formatting/case/punctuation only,
not proven lexical improvements. Existing generated Markdown/PDF
documents remain baseline versions.

## Runtime and model provenance

- GPU: RTX A5000 Laptop, 16 GB. Existing native llama.cpp build 124, commit
  `e71b805`; CUDA, flash attention, one slot per model, localhost only.
- [LightOnOCR-3-1B](https://huggingface.co/lightonai/LightOnOCR-3-1B), released
  October 2026: community Q8_0 decoder + BF16 projector, conversion revision
  `ff5c331ec2cfee9de1c8054c21ccf88b8c5b0faa` from
  [prithivMLmods](https://huggingface.co/prithivMLmods/LightOnOCR-3-1B-GGUF).
  The converter does **not** identify its exact upstream conversion commit.
- [PaddleOCR-VL-1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6), released
  May 2026: official BF16 GGUF decoder/projector, revision
  `511b09642bb324401f15f97cc23bc67e8f0a291d` from
  [PaddlePaddle](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6-GGUF).
- 3.29 GB total weight downloads; SHA-256 verified. Both resident after smoke
  tests used about 4.8 GB GPU RAM. No base Python packages or vendor sources were
  changed: the existing PyTorch install is broken, so inference used native
  binaries and a standard-library client. LightOn's CUDA IM2COL fallback warning
  did not prevent actual generation but may affect speed.
- Latency is one local pass with mixed prompt-cache state, after server warmup;
  it excludes model loading/client payload preparation. No dots speed comparison.
  Paddle's full document-parser pipeline was not tested, only its VLM component.

## Reproduce

Weights are local, ignored by Git. Download the four files listed in the
[manifest](../ocr_evaluation/model_manifest.json) using the revision-pinned
Hugging Face `resolve/<revision>/<filename>` URLs and verify their hashes.
Run each server in a separate terminal:

```bash
vendor/llama.cpp/build/bin/llama-server -m models/ocr_benchmark/LightOnOCR-3-1B.Q8_0.gguf --mmproj models/ocr_benchmark/LightOnOCR-3-1B.mmproj-bf16.gguf --host 127.0.0.1 --port 8001 --alias lightonocr-3-1b -ngl 99 -c 8192 -np 1 -fa on --temp 0 --jinja
vendor/llama.cpp/build/bin/llama-server -m models/ocr_benchmark/PaddleOCR-VL-1.6-GGUF.gguf --mmproj models/ocr_benchmark/PaddleOCR-VL-1.6-GGUF-mmproj.gguf --host 127.0.0.1 --port 8002 --alias paddleocr-vl-1.6 -ngl 99 -c 8192 -np 1 -fa on --temp 0 --jinja
```

Use a working Python interpreter (these scripts require no PyTorch):

```bash
python -m tools.benchmark_ocr
python -m tools.benchmark_ocr --rescore
python -m tools.reocr_layouts --backend lighton3 --output outputs/ocr_candidates/new_run/layouts_with_texts.json
python -m vespa_retrieval.build_ui --source outputs/ocr_candidates/lighton3/layouts_with_texts.json --site-dir docs/ocr_candidate
python -m tools.build_ocr_report
python -m unittest discover -s tests -v
node --test tests/test_ui.cjs
```

`--rescore` uses saved raw responses without requiring running servers. New
candidate destinations must not already exist. Ordinary `python -m vespa_retrieval.build_ui`
still builds the baseline studio. Native servers require no credentials and
should remain bound to localhost; stop them when not in use.