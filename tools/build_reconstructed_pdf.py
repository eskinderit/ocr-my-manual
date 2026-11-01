import argparse
import json
import re
from pathlib import Path
from typing import Optional
from html import unescape

from PIL import Image, ImageDraw, ImageFont


OUT = Path("outputs")
IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff")


def extract_json_from_html(html: str) -> dict:
    marker = 'const data = '
    start = html.find(marker)
    if start < 0:
        raise ValueError('Could not find embedded data declaration in HTML input')
    start += len(marker)
    brace = 0
    in_string = False
    escape = False
    end = None
    for i, ch in enumerate(html[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == '{':
                brace += 1
            elif ch == '}':
                brace -= 1
                if brace == 0:
                    end = i + 1
                    break
    if end is None:
        raise ValueError('Embedded JSON object in HTML could not be parsed')
    json_text = html[start:end]
    return json.loads(json_text)


def load_combined(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".html", ".htm"}:
        return extract_json_from_html(text)
    return json.loads(text)


def load_page_layouts(layout_dir: Path) -> dict:
    layouts = {}
    for path in sorted(layout_dir.glob('*_layout.json')):
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            name = path.stem.replace('_layout', '')
            layouts[name] = data
        except Exception:
            continue
    return layouts


def resolve_image_name(name: str, imgs_dir: Path) -> Optional[str]:
    for ext in ("",) + IMG_EXTS:
        candidate = f"{name}{ext}"
        if (imgs_dir / candidate).exists():
            return candidate
    return None


def numeric_sort_key(name: str) -> int:
    match = re.search(r"(\d+)(?=\.[^.]+$)", name)
    return int(match.group(1)) if match else 10**18


def load_table_file(path_str: str) -> Optional[str]:
    if not isinstance(path_str, str) or not path_str.strip():
        return None
    candidate = Path(path_str)
    if candidate.exists():
        return candidate.read_text(encoding='utf-8', errors='ignore')
    for base in (Path('outputs'), Path('docs')):
        candidate = base / path_str
        if candidate.exists():
            return candidate.read_text(encoding='utf-8', errors='ignore')
    return None


def normalize_table_content(table: dict) -> Optional[str]:
    if not isinstance(table, dict):
        return None

    if isinstance(table.get("raw_text"), str) and table["raw_text"].strip():
        return table["raw_text"].strip()
    if isinstance(table.get("html_inline"), str) and table["html_inline"].strip():
        return table["html_inline"].strip()

    html_value = table.get("html")
    if isinstance(html_value, str) and html_value.strip():
        loaded = load_table_file(html_value.strip())
        if loaded:
            return loaded.strip()
        return html_value.strip()

    tsv_value = table.get("tsv")
    if isinstance(tsv_value, str) and tsv_value.strip():
        loaded = load_table_file(tsv_value.strip())
        if loaded:
            return loaded.strip()
        return tsv_value.strip()

    rows = table.get("rows")
    if isinstance(rows, list):
        flat_rows = []
        for row in rows:
            if isinstance(row, list):
                flat_rows.extend(str(part) for part in row if part is not None)
            elif isinstance(row, str):
                flat_rows.append(row)
        if flat_rows:
            return "\n".join(flat_rows)

    return None


def strip_html(text: str) -> str:
    if not text:
        return ""
    text = unescape(text)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</p>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_table_html(text: str) -> bool:
    return bool(re.match(r"^\s*<table[\s>]", text or "", flags=re.IGNORECASE))


def resolve_render_target(element: dict) -> tuple[Optional[str], Optional[str]]:
    table = element.get("table")
    if isinstance(table, dict):
        table_content = normalize_table_content(table)
        if table_content:
            return "table", table_content

    text = element.get("text")
    if isinstance(text, str) and text.strip():
        return "text", text.strip()

    crop = element.get("crop")
    if isinstance(crop, str) and crop.strip():
        return "crop", crop.strip()

    return None, None


def extract_text(element: dict) -> Optional[str]:
    kind, value = resolve_render_target(element)
    if kind == "text":
        return value
    if kind == "table":
        return strip_html(value or "") if value else None
    return None


def extract_table_rows(element: dict) -> list[list[str]]:
    kind, value = resolve_render_target(element)
    if kind != "table":
        return []
    if not value:
        return []

    text = strip_html(value)
    if not text:
        return []

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    return [lines]


def wrap_text(text: str, max_width: int, draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont) -> list[str]:
    if not text:
        return []
    words = text.split()
    if not words:
        return []

    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def render_page(image_path: Path, record: dict, font_dir: Path, include_background: bool = False) -> Image.Image:
    source_img = Image.open(image_path).convert("RGBA")
    if include_background:
        canvas = source_img.copy()
    else:
        canvas = Image.new("RGBA", source_img.size, "white")
    draw = ImageDraw.Draw(canvas)

    rendered_tables = set()
    rendered_crops = set()

    for element in record.get("elements", []):
        bbox = element.get("bbox")
        if not bbox:
            continue

        x1, y1, x2, y2 = [int(float(v)) for v in bbox]
        x1 = max(0, min(x1, source_img.width - 1))
        y1 = max(0, min(y1, source_img.height - 1))
        x2 = max(x1 + 1, min(x2, source_img.width))
        y2 = max(y1 + 1, min(y2, source_img.height))

        label = str(element.get("label") or "text")
        color = (255, 255, 255, 200)
        outline = (0, 0, 0, 220)
        draw.rectangle((x1, y1, x2, y2), fill=color, outline=outline, width=2)

        kind, value = resolve_render_target(element)

        if kind == "table":
            table_key = value or ""
            if table_key and table_key in rendered_tables:
                continue
            if table_key:
                rendered_tables.add(table_key)
            box_w = max(1, x2 - x1)
            box_h = max(1, y2 - y1)
            font_size = max(10, min(20, int(min(box_w, box_h) / 10)))
            font = load_font(font_size)
            plain_text = strip_html(value or "") if value else ""
            lines = wrap_text(plain_text, box_w - 8, draw, font)
            if lines:
                line_height = max(12, font.getbbox("Ag")[3] + 2)
                start_y = y1 + 4
                for line in lines[:12]:
                    draw.text((x1 + 4, start_y), line, fill=(0, 0, 0, 220), font=font)
                    start_y += line_height

        elif kind == "text":
            text = value or ""
            box_w = max(1, x2 - x1)
            box_h = max(1, y2 - y1)
            font_size = max(12, min(28, int(min(box_w, box_h) / 6)))
            font = load_font(font_size)
            lines = wrap_text(text, box_w - 8, draw, font)
            if lines:
                line_height = max(12, font.getbbox("Ag")[3] + 2)
                start_y = y1 + 4
                for line in lines[:10]:
                    draw.text((x1 + 4, start_y), line, fill=(0, 0, 0, 220), font=font)
                    start_y += line_height

        elif kind == "crop":
            crop_path = value or ""
            if crop_path and crop_path in rendered_crops:
                continue
            if crop_path:
                rendered_crops.add(crop_path)
            resolved_crop = None
            for candidate in [Path(crop_path), Path("outputs") / crop_path, Path("docs") / crop_path, Path(crop_path).name]:
                if candidate.exists():
                    resolved_crop = candidate
                    break
            if resolved_crop is None:
                for base in (Path("outputs"), Path("docs")):
                    candidate = base / crop_path
                    if candidate.exists():
                        resolved_crop = candidate
                        break
            if resolved_crop and resolved_crop.exists():
                try:
                    crop_img = Image.open(resolved_crop).convert("RGBA")
                    crop_w = max(1, x2 - x1)
                    crop_h = max(1, y2 - y1)
                    crop_img = crop_img.resize((crop_w, crop_h), Image.LANCZOS)
                    canvas.paste(crop_img, (x1, y1), crop_img)
                except Exception:
                    pass

    return canvas.convert("RGB")


def build_pdf(input_json: Path, imgs_dir: Path, output_pdf: Path, include_background: bool = False, layout_dir: Optional[Path] = None) -> Path:
    combined = load_combined(input_json)
    page_layouts = load_page_layouts(layout_dir or Path('outputs/layout_documents')) if layout_dir is not None or Path('outputs/layout_documents').exists() else {}
    pages = []

    for name in sorted(combined.keys(), key=numeric_sort_key):
        record = combined[name] or {}
        if page_layouts:
            page_record = page_layouts.get(name)
            if page_record:
                if isinstance(page_record, list) and not record.get("elements"):
                    record = {"elements": page_record}
                elif isinstance(page_record, dict) and not record:
                    record = page_record
        img_name = resolve_image_name(name, imgs_dir)
        if img_name is None:
            print(f"Skipping {name}: no matching image in {imgs_dir}")
            continue
        image_path = imgs_dir / img_name
        if not image_path.exists():
            continue
        pages.append(render_page(image_path, record, imgs_dir, include_background=include_background))

    if not pages:
        raise FileNotFoundError("No pages were rendered; check the input JSON and image folder.")

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    pages[0].save(output_pdf, save_all=True, append_images=pages[1:])
    return output_pdf


def main() -> None:
    default_input = Path('outputs/ui/index.html') if Path('outputs/ui/index.html').exists() else Path('outputs/layouts_with_texts.json')
    parser = argparse.ArgumentParser(description="Create a PDF from reconstructed document page images.")
    parser.add_argument("--input", default=str(default_input), help="Path to the combined layouts JSON file or UI HTML file containing embedded data")
    parser.add_argument("--images", default="imgs", help="Directory containing the original page images")
    parser.add_argument("--output", default="docs/reconstructed_document.pdf", help="Output PDF path")
    parser.add_argument("--with-background", action="store_true", help="Include the original scanned page background behind reconstructed content")
    parser.add_argument("--layout-dir", default="outputs/layout_documents", help="Directory containing per-page layout JSON files")
    args = parser.parse_args()

    input_path = Path(args.input)
    imgs_dir = Path(args.images)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_path}")
    if not imgs_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {imgs_dir}")

    pdf_path = build_pdf(input_path, imgs_dir, output_path, include_background=args.with_background, layout_dir=Path(args.layout_dir))
    print(f"PDF written to {pdf_path}")


if __name__ == "__main__":
    main()
