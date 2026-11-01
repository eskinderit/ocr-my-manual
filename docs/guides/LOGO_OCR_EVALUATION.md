# Successful cover OCR — Qwen3-VL-4B

> Historical evaluation of the sample Vespa manual, not a recommendation for
> the current pipeline. After reorganization, probes live under tools and commands
> below run from the repository root; evidence links resolve from this guide.
> Use the [current README](../../README.md) for new manuals.

Date: 31 October 2026. [Image-backed evidence](../logo_ocr/index.html).

## Result

Qwen3-VL-4B-Instruct correctly outputs **Vespa 150** on both the entire original
cover and the isolated logo crop, with two neutral prompts. No expected brand
names, image filenames, corrections or document context were sent in the messages.
The unchanged isolated crop removes the PIAGGIO/year/subtitle context.

The two ordinary full-cover requests omitted the standalone P inside the emblem.
A third generic coverage prompt (include single letters inside emblems) returned
all visibly readable text:

```text
P
PIAGGIO
mod. 1956
Vespa
150
USO E MANUTENZIONE
```

The optional decorative swash/dash near 150 is not a lexical error. Text order
at the top differs but no words/digits are wrong. An independent image reviewer
confirmed the logo, year, number and subtitle, including the emblem P.

One ordinary Italian paragraph was also exact against the frozen image-verified
reference after line-wrap/transport normalization. No broad superiority claim:
Qwen has **not** been benchmarked on the full twelve-crop set or all manual pages.
Logo familiarity learned during pretraining may contribute; the isolated crop
success and neutral prompts do not prove pure glyph-based recognition.

## Evidence

- [Original initial results](../../outputs/logo_ocr/qwen3vl4b/results.json): two full-cover
  requests, two isolated-logo requests, one paragraph control.
- [Complete cover with emblem coverage prompt](../../outputs/logo_ocr/qwen3vl4b/full-cover-emblem-check.json).
- [Pinned weights and hashes](../../outputs/logo_ocr/qwen3vl4b/model_manifest.json).
- [Reproducible neutral probe](../../tools/test_logo_ocr.cjs),
  [generic emblem-coverage check](../../tools/verify_cover_ocr.cjs).

All six responses finished with stop. Temperature 0, top-p 1, seed 42,
2048-token output budget. Full first request ~4.37 seconds; subsequent requests
reuse prompt cache, so faster times are not cold-start benchmark results.
The coverage check took 1.18 seconds with 4033 cached tokens. Existing native
llama.cpp build 124/e71b805, CUDA, 8192-token context, one slot, flash attention.
Original images/production transcriptions remain unchanged.

## Model

Upstream: [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct).
Community conversion: [mradermacher/Qwen3-VL-4B-Instruct-GGUF](https://huggingface.co/mradermacher/Qwen3-VL-4B-Instruct-GGUF),
revision d41f9f8c74871726f466396779ef60c1fe166639. Q8_0 decoder (4.28 GB),
F16 vision projector (0.836 GB), both downloaded and SHA-256 verified. This is a
vision-language model performing OCR, not a dedicated OCR-only recognizer.

GLM-OCR was researched as a fallback but not downloaded/tested: the first stronger
candidate already solved the requested failure, so additional downloads were avoided.
About 5.12 GB new weights remain locally; they are ignored by Git.

## Reproduce

Start the local server, then run the probes from the repository root:

```bash
vendor/llama.cpp/build/bin/llama-server -m models/logo_ocr/Qwen3-VL-4B-Instruct.Q8_0.gguf --mmproj models/logo_ocr/Qwen3-VL-4B-Instruct.mmproj-f16.gguf --host 127.0.0.1 --port 8003 --alias qwen3vl4b -ngl 99 -c 8192 -np 1 -fa on --jinja
node tools/test_logo_ocr.cjs qwen3vl4b 8003
node tools/verify_cover_ocr.cjs
```

The probes save raw outputs under outputs/logo_ocr/qwen3vl4b. Re-running the
probe replaces its own prior probe artifacts, never production layout data.
The temporary server was stopped after evaluation to free GPU resources.
