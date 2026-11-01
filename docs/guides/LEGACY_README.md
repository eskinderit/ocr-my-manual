# Vespa Retrieval: a GPU-poor OCR for instruction manuals

> Historical documentation preserved before project reorganization. Old root-script
> commands below are superseded by the [current README](../../README.md) and
> [main.py](../../main.py). Evaluation utilities now live under tools; runtime
> modules and the UI template now live under vespa_retrieval.

A pipeline to convert your scanned manual in a digital, search-ready way, without finetuning.

## Refactored workflow

The current Qwen workflow is documented in [PIPELINE_WORKFLOW.md](PIPELINE_WORKFLOW.md).
Processing, table repair and static-site building are separate steps with configurable
input/output paths. No upload service or deployment provider is required.

- [run_qwen_layout_ocr.py](../../vespa_retrieval/run_qwen_layout_ocr.py): crop OCR and resumable per-page checkpoints.
- [repair_qwen_tables.py](../../vespa_retrieval/repair_qwen_tables.py): optional three-column Italian troubleshooting-table retries.
- [build_ui.py](../../vespa_retrieval/build_ui.py): build a self-contained static viewer without an OCR server.
- [table_html.py](../../vespa_retrieval/table_html.py) and [pipeline_utils.py](../../vespa_retrieval/pipeline_utils.py): shared table validation,
  image transforms, completion checks and atomic JSON writes.

## Document Studio and newer OCR evaluation

The viewer now follows the dark studio/panel styling of the circular-factory
project, with warm accents, responsive navigation, OCR-text search, keyboard
controls and synchronized zoom. Open [docs/index.html](../index.html).

**Actual local tests:** LightOnOCR-3-1B and PaddleOCR-VL-1.6 were run on identical
manual crops. LightOn reduced content character error from **2.40% to 0.36%**
on a small image-verified diagnostic set, but table structure and some numbers
still need review. Existing tables are therefore preserved in the separate
155-block text candidate; the original output is not overwritten.

- [Image-backed evidence and all model outputs](../ocr_evaluation/index.html)
- [Method, limitations, model provenance and reproduction](OCR_EVALUATION.md)
- [Hybrid candidate viewer — not fully reviewed](../ocr_candidate/index.html)

Build a candidate viewer with `build_ui.py --source <candidate-json> --site-dir
<new-site-directory>`. The defaults still build the original output into `docs/`.

# OCR Engine choice
The choice of the OCR engine to be used is a crucial point. My aim is to propose a pipeline that can be used even without a closed LLMs or proprietary tools (yes, they're the ones which perform better, usually). 

There are tens or hundreds of both open and closed OCR models around and it's not easy to understand which perform better, however, someone developed a platform in which is possible to have a real comparison with your own documents: [OCR Arena](https://www.ocrarena.ai/battle) a fantastic tool to avoid being fooled by benchmaxing practices and linkedin posts "noise" on new models apparent breakthrough performance :) (N.B.: this is not an ad or anything).

Most of the difficulty in OCR tasks lies in the ability to properly parse multi-column layouts, parsing tables and handling images (ok, even formulas can be tricky if there are any in your document, but this is not the case). 
Traditional tools like `Tesseract` perform better if finetuned, also, they need proper preprocessing to handle heterogeneous kinds of input, formatting, font size and so on. The text can be written in different fonts and dimensions in user manuals. Drawings can contain text which may confuse the targets for detection.

LLM-powered OCR models like `dots.ocr` perform detection of text-based elements (their bounding box) and transcribe their content in one shot. At first i thought that it could really mess up the bbox since it doesn't have specialized heads, but in most of the cases it actually works!

# Reconstruction of the User Manual pages

To reconstruct the document digitally I've added an upfront detection of the elements with `DocLayout-YOLO`, which allows to retrieve a finer structure for each document page.


## Error analysis for dots.ocr

Sometimes logos are interpreted as actual text by the detector, this may bring to unexpected transcriptions which are ultimately not correct. 

Dots OCR doesn't natively predict the presence of borders inside tables. Therefore even if the transcription is done correctly, the content may not be perfectly readable.

# Get started

## Download dots.ocr model
```
wget -c \
"https://huggingface.co/ggml-org/dots.ocr-GGUF/resolve/main/dots.ocr-Q8_0.gguf?download=true" \
-O dots.ocr-Q8_0.gguf
```
## Run the model through llama.cpp backend
```
docker run --rm --gpus all -p 8000:8080 -v "$(p
wd)/models":/models   ghcr.io/ggml-org/llama.cpp:server-cuda   --hf-repo ggml-org/Dots.OCR-GGUF --gpu-layers 999
```
## Run the script
 ```
 python ocr_query.py 
 ```

Put your original scanned manual images in the `imgs/` folder before building the browser UI. Supported file types include `.png`, `.jpg`, `.jpeg`, `.webp`, `.tif`, and `.tiff`.

1. Build the site. This reads `outputs/layouts_with_texts.json` and writes a
   self-contained, static site to `docs/` (relative asset paths, table HTML
   embedded inline, `originals/` and `crops/` copied alongside `index.html`):
   ```
   python build_ui.py
   ```
   > If you get `outputs/layouts_with_texts.json not found`, run the pipeline first:
   > `python detect_layout.py` → `python layout_to_doc.py` → `python build_ui.py`.

2. Preview locally. Because the site is self-contained you can just open
   `docs/index.html` in your browser, or serve the `docs/` folder:
   ```
   python -m http.server 8080 --directory docs
   ```
   then open `http://localhost:8080/`.

Pick a page from the sidebar to load it. Use the zoom buttons and the checkboxes
to toggle the element backgrounds and the reconstructed background image.

## Publish it as a GitHub Pages website

The generated `docs/` folder is ready to host on GitHub Pages:

1. Commit and push the `docs/` folder (including `docs/originals/`,
   `docs/crops/` and `docs/index.html`) to your `main` branch:
   ```
   git add docs
   git commit -m "Add static layout viewer site"
   git push
   ```
2. In your repository on GitHub, go to **Settings → Pages**.
3. Under **Build and deployment → Source**, choose **Deploy from a branch**.
4. Set the branch to **main** and the folder to **/docs**, then **Save**.
5. After a minute the site is live at
   `https://<your-username>.github.io/<your-repo>/`.

The page uses only relative paths and embeds table HTML inline, so it works
correctly under the `/<your-repo>/` sub-path that GitHub Pages serves from.
Re-run `python build_ui.py` whenever the pipeline output changes, then commit
the updated `docs/` folder.

## Pipeline reference

The end-to-end flow, from scanned images to the static layout viewer:

```
imgs/*.png
  └─ detect_layout.py      → outputs/layouts/ (+ layouts.json)   [DocLayout-YOLO boxes]
       └─ ocr_query.py     → outputs/ocr_results.json            [dots.ocr transcription]
            └─ layout_to_doc.py → outputs/layouts_with_texts.json [text per element/crop/table]
                 ├─ build_ui.py             → docs/ (static viewer site)
                 └─ build_reconstructed_pdf.py → reconstructed PDF/preview
```

### Local dots.ocr server

The OCR step (`ocr_query.py`) expects a **dots.ocr** vision server exposing an
OpenAI-compatible API on port 8000. Run it with llama.cpp:
```
docker run --rm --gpus all -p 8000:8080 -v "$(pwd)/models":/models \
  ghcr.io/ggml-org/llama.cpp:server-cuda --hf-repo ggml-org/Dots.OCR-GGUF --gpu-layers 999
```

### Tests

Standard-library unit tests live in `tests/`:
```
python -m unittest discover -s tests -v
```
They cover the reconstructed-PDF render targets.

### Repository layout

| Path | Purpose |
|---|---|
| `detect_layout.py` | DocLayout-YOLO element detection |
| `ocr_query.py` | dots.ocr transcription of pages/crops |
| `layout_to_doc.py` | Merge detections + OCR into structured docs |
| `build_ui.py` / `ui_template.html` | Static layout viewer site (`docs/`) |
| `build_reconstructed_pdf.py` | Rebuild a readable PDF/preview from the layout |
| `tests/` | Standard-library unit tests |
| `requirements.txt` | OCR + layout pipeline dependencies |

For the newer-model experiment and safe candidate workflow, see [OCR_EVALUATION.md](OCR_EVALUATION.md).



