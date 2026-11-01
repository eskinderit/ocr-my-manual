import base64
import json
from pathlib import Path
from typing import Dict, List

from PIL import Image
import hashlib
import re

from openai import OpenAI

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except Exception:
    TESSERACT_AVAILABLE = False


client = OpenAI(
    base_url="http://localhost:8000",
    api_key="dummy",
)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_model_name() -> str:
    models = client.models.list()
    return models.data[0].id


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def ocr_crop(image_path: str, model: str | None = None) -> str:
    if model is None:
        model = get_model_name()
    b64 = encode_image(image_path)
    ext = Path(image_path).suffix.lstrip(".").lower()
    mime = f"image/{ext}" if ext else "image/png"

    prompt = "Extract the plain text content shown in the image. Output only the text, no extra commentary."

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ],
            }
        ],
    )
    return response.choices[0].message.content


def extract_table_from_crop(crop_path: Path, out_dir: Path, model: str | None = None, prefer_model: bool = False):
    """Extract table structure.
    By default prefer pytesseract when available, but if `prefer_model=True` (used for table crops)
    we will use the OCR model (Dots) like we do for plain text crops.
    """
    # if pytesseract is available and caller did not request model-first, use it
    if TESSERACT_AVAILABLE and not prefer_model:
        try:
            img = Image.open(crop_path).convert("RGB")
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            n = len(data.get("text", []))
            rows = {}
            for i in range(n):
                text = data["text"][i].strip()
                if not text:
                    continue
                line_num = data.get("line_num", [0] * n)[i]
                left = data.get("left", [0] * n)[i]
                if line_num not in rows:
                    rows[line_num] = []
                rows[line_num].append((left, text))

            table_rows = []
            for ln in sorted(rows.keys()):
                parts = [t for _, t in sorted(rows[ln], key=lambda x: x[0])]
                table_rows.append(parts)

            tsv = "\n".join(["\t".join(r) for r in table_rows])
            html_rows = ["<tr>" + "".join([f"<td>{v}</td>" for v in r]) + "</tr>" for r in table_rows]
            html = "<table border=1>" + "".join(html_rows) + "</table>"

            # save outputs
            tables_root = out_dir / "tables" / crop_path.parent.name
            tables_root.mkdir(parents=True, exist_ok=True)
            base = crop_path.stem
            tsv_path = tables_root / f"{base}.tsv"
            html_path = tables_root / f"{base}.html"
            tsv_path.write_text(tsv, encoding="utf-8")
            html_path.write_text(html, encoding="utf-8")

            return {"tsv": str(tsv_path.relative_to(out_dir)), "html": str(html_path.relative_to(out_dir)), "rows": table_rows}
        except Exception as e:
            # fall through to OCR fallback
            print(f"pytesseract failed: {e}")

    # fallback / model-first path: use OCR model to extract plain text and heuristically split into rows/cols
    try:
        if model is None:
            try:
                model = get_model_name()
            except Exception:
                model = None
        if model is None:
            return {"error": "no OCR model available"}

        raw = ocr_crop(str(crop_path), model=model)
        if isinstance(raw, dict) and raw.get("error"):
            return {"error": raw.get("error")}
        text = raw if isinstance(raw, str) else str(raw)

        # heuristic: split into lines, then split columns by two-or-more spaces or tab
        lines = [l for l in text.splitlines() if l.strip()]
        table_rows = []
        for ln in lines:
            parts = re.split(r"\s{2,}|\t", ln.strip())
            table_rows.append(parts)

        tsv = "\n".join(["\t".join(r) for r in table_rows])
        html_rows = ["<tr>" + "".join([f"<td>{v}</td>" for v in r]) + "</tr>" for r in table_rows]
        html = "<table border=1>" + "".join(html_rows) + "</table>"

        tables_root = out_dir / "tables" / crop_path.parent.name
        tables_root.mkdir(parents=True, exist_ok=True)
        base = crop_path.stem
        tsv_path = tables_root / f"{base}.tsv"
        html_path = tables_root / f"{base}.html"
        tsv_path.write_text(tsv, encoding="utf-8")
        html_path.write_text(html, encoding="utf-8")

        return {"tsv": str(tsv_path.relative_to(out_dir)), "html": str(html_path.relative_to(out_dir)), "rows": table_rows, "raw_text": text}
    except Exception as e:
        return {"error": str(e)}


def load_layouts(path: Path) -> Dict:
    # If path is a directory, read per-image JSON files inside and return a dict mapping
    if path.is_dir():
        layouts = {}
        for p in sorted(path.glob('*.json')):
            try:
                data = json.loads(p.read_text(encoding='utf-8'))
                # map filename (without .json) to data; e.g., vespa_manual_0.png.json -> 'vespa_manual_0.png'
                key = p.stem
                layouts[key] = data
            except Exception:
                # skip invalid files
                continue
        return layouts
    else:
        return json.loads(path.read_text(encoding="utf-8"))


TEXT_WHITELIST = {
    "plain_text",
    "text",
    "title",
    "figure_caption",
    "caption",
    "section-header",
    "section_header",
    "list-item",
    "list_item",
    "footnote",
    "page-header",
    "page-footer",
    "header",
    "footer",
}


def is_text_label(label: str) -> bool:
    if not label:
        return False
    l = label.lower().replace(" ", "_")
    if l in TEXT_WHITELIST:
        return True
    if "text" in l or "title" in l or "caption" in l or "footnote" in l or "header" in l or "list" in l:
        return True
    return False


def process_all(layouts_path: Path, imgs_dir: Path, out_dir: Path):
    layouts = load_layouts(layouts_path)
    try:
        model = get_model_name()
    except Exception:
        model = None
        print("Warning: OpenAI-compatible server unavailable; OCR will be skipped or may error.")
    crops_root = out_dir / "crops"
    docs_root = out_dir / "layout_documents"
    crops_root.mkdir(parents=True, exist_ok=True)
    docs_root.mkdir(parents=True, exist_ok=True)

    combined = {}
    for img_name, elements in layouts.items():
        print(f"Processing {img_name}...")
        img_path = imgs_dir / img_name
        if not img_path.exists():
            # try common image extensions (layout keys are often stored without extension)
            found = None
            for ext in ('.png', '.jpg', '.jpeg', '.webp', '.tif', '.tiff'):
                p = imgs_dir / f"{img_name}{ext}"
                if p.exists():
                    found = p
                    break
            if found:
                img_path = found
            else:
                combined[img_name] = {"error": "image not found"}
                continue
        pil = Image.open(img_path).convert("RGB")
        img_crops_dir = crops_root / Path(img_name).stem
        img_crops_dir.mkdir(parents=True, exist_ok=True)

        elements_with_text = []
        # track seen bboxes and crop hashes to avoid processing identical crops twice
        seen_bboxes = set()
        seen_hashes = set()
        # helper: extract numeric confidence from layout element
        def _get_confidence(e: dict) -> float:
            for k in ("confidence", "score", "probability", "conf"):
                if k in e:
                    try:
                        return float(e.get(k) or 0.0)
                    except Exception:
                        try:
                            return float(str(e.get(k)).strip())
                        except Exception:
                            return 0.0
            return 0.0

        def _iou(bb1, bb2) -> float:
            x1, y1, x2, y2 = bb1
            x3, y3, x4, y4 = bb2
            xi1 = max(x1, x3)
            yi1 = max(y1, y3)
            xi2 = min(x2, x4)
            yi2 = min(y2, y4)
            iw = max(0, xi2 - xi1)
            ih = max(0, yi2 - yi1)
            inter = iw * ih
            area1 = max(0, x2 - x1) * max(0, y2 - y1)
            area2 = max(0, x4 - x3) * max(0, y4 - y3)
            union = area1 + area2 - inter
            return (inter / union) if union > 0 else 0.0
        for i, el in enumerate(elements):
            try:
                bbox = el.get("bbox")
                label = el.get("label", "")
                if not bbox:
                    continue
                x1, y1, x2, y2 = map(int, map(round, bbox))
                # clip
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(pil.width, x2)
                y2 = min(pil.height, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                bbox_key = (x1, y1, x2, y2)
                # IoU deduplication: if a previous element overlaps >0.20, keep the one with higher confidence
                cur_conf = _get_confidence(el)
                skip_current = False
                for prev in list(elements_with_text):
                    prev_bbox = tuple(prev.get('bbox', [0,0,0,0]))
                    if _iou(bbox_key, prev_bbox) > 0.20:
                        prev_conf = _get_confidence(prev)
                        if prev_conf >= cur_conf:
                            skip_current = True
                            break
                        else:
                            # current is higher confidence: remove previous element and its files
                            try:
                                prev_crop = prev.get('crop')
                                if prev_crop:
                                    p = out_dir / prev_crop
                                    if p.exists():
                                        p.unlink()
                                    tables_dir = out_dir / 'tables' / Path(prev_crop).parent.name
                                    stem = Path(prev_crop).stem
                                    tsv = tables_dir / f"{stem}.tsv"
                                    html = tables_dir / f"{stem}.html"
                                    if tsv.exists():
                                        tsv.unlink()
                                    if html.exists():
                                        html.unlink()
                            except Exception:
                                pass
                            try:
                                elements_with_text.remove(prev)
                            except ValueError:
                                pass
                            # don't break; continue checking against other prev elements
                if skip_current:
                    continue

                crop = pil.crop((x1, y1, x2, y2))
                # compute a stable fingerprint of the crop (size+mode+raw bytes)
                try:
                    raw = crop.tobytes()
                except Exception:
                    # fallback to raw PNG bytes
                    from io import BytesIO
                    buf = BytesIO()
                    crop.save(buf, format='PNG')
                    raw = buf.getvalue()
                h = hashlib.md5()
                h.update(raw)
                h.update(f"{crop.size[0]}x{crop.size[1]}_{crop.mode}".encode('utf-8'))
                crop_hash = h.hexdigest()

                # skip exact duplicate bbox or identical crop image
                if bbox_key in seen_bboxes or crop_hash in seen_hashes:
                    # skip adding duplicate element
                    continue

                seen_bboxes.add(bbox_key)
                seen_hashes.add(crop_hash)

                crop_path = img_crops_dir / f"{Path(img_name).stem}_{i}.png"
                crop.save(crop_path)

                text = ""
                if is_text_label(label):
                    try:
                        if model is None:
                            text = ""
                        else:
                            text = ocr_crop(str(crop_path), model=model)
                    except Exception as e:
                        text = f"ERROR: {e}"

                element_out = {
                    "label": label,
                    "bbox": [x1, y1, x2, y2],
                    "score": cur_conf,
                    "text": text,
                    # store crop path relative to OUTPUT_DIR for easier serving
                    "crop": str(crop_path.relative_to(out_dir)),
                }

                # if it's a table, attempt to extract table structure via pytesseract or OCR fallback
                if "table" in label.lower() or "tab" in label.lower():
                    # for tables: prefer the OCR model if available, but allow pytesseract fallback
                    # (prefer_model=False lets pytesseract run when installed; if model is set,
                    # extract_table_from_crop will still use it when appropriate)
                    table_info = extract_table_from_crop(crop_path, out_dir, model=model, prefer_model=False)
                    element_out["table"] = table_info

                elements_with_text.append(element_out)
            except Exception as e:
                elements_with_text.append({"error": str(e)})

        # sort by y then x (reading order)
        elements_with_text = sorted(elements_with_text, key=lambda e: (e.get("bbox", [0, 0, 0, 0])[1], e.get("bbox", [0, 0, 0, 0])[0]))

        # assemble document text
        doc_text = []
        for el in elements_with_text:
            t = el.get("text")
            if t:
                doc_text.append(t.strip())

        doc_md = "\n\n".join(doc_text)
        # save per-image json and markdown
        json_out = docs_root / f"{Path(img_name).stem}_layout.json"
        md_out = docs_root / f"{Path(img_name).stem}.md"
        json_out.write_text(json.dumps(elements_with_text, indent=2, ensure_ascii=False), encoding="utf-8")
        md_out.write_text(doc_md, encoding="utf-8")

        combined[img_name] = {
            "elements": elements_with_text,
            "document_markdown": str(md_out),
        }

    combined_out = out_dir / "layouts_with_texts.json"
    combined_out.write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Done. Saved crops to {crops_root}, documents to {docs_root}, combined JSON to {combined_out}")


if __name__ == "__main__":
    import sys

    # default to directory with per-image JSONs (written by detect_layout.py)
    layouts_path = Path("outputs/layouts")
    imgs_dir = Path("imgs")
    out_dir = Path("outputs")
    if len(sys.argv) > 1:
        layouts_path = Path(sys.argv[1])
    process_all(layouts_path, imgs_dir, out_dir)
