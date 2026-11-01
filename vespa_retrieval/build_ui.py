import json
from pathlib import Path

from .table_html import clean_table_html

OUT = Path("outputs")


def load_combined(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


TEMPLATE_PATH = Path(__file__).parent / 'templates' / 'ui_template.html'


def build(combined_path: Path | None = None, site_dir: Path | None = None, asset_root: Path | None = None, show_abandon: bool = False, *, imgs_dir: Path | None = None):
    assets = asset_root or OUT
    combined_path = combined_path or OUT / 'layouts_with_texts.json'
    if not combined_path.exists():
        print('outputs/layouts_with_texts.json not found; run layout_to_doc.py first')
        return
    combined = load_combined(combined_path)
    # build docs/originals directly from imgs/ so the site has a single originals copy
    import shutil
    src_imgs = imgs_dir or Path('imgs')
    IMG_EXTS = ('.png', '.jpg', '.jpeg', '.webp', '.tif', '.tiff')

    def resolve_image_name(name: str) -> str | None:
        """Return the actual image filename (with extension) for a layout key.

        Layout keys may be stored without an extension (e.g. 'vespa_manual_0'),
        while the real files are 'vespa_manual_0.png'. Try the name as-is first,
        then common extensions, looking in imgs/.
        """
        for ext in ('',) + IMG_EXTS:
            candidate = f"{name}{ext}"
            if (src_imgs / candidate).exists():
                return candidate
        return None

    # remap records so keys match a real image file.
    resolved = {}
    for name, rec in combined.items():
        img_name = resolve_image_name(name)
        if img_name is None:
            print(f"Warning: no image file found for '{name}'; skipping")
            continue
        resolved[img_name] = rec
    combined = resolved

    # embed each table's HTML directly in the data so the page needs no fetch()
    # (works when opened as a file:// preview and when served by GitHub Pages)
    for rec in combined.values():
        for el in rec.get('elements', []):
            table = el.get('table')
            # if the table was OCR'd and we have raw_text, prefer embedding it
            if isinstance(table, dict):
                if table.get('raw_text') and 'html_inline' not in table:
                    table['html_inline'] = clean_table_html(table['raw_text'])
                # if an external HTML path exists, try to inline that file's contents
                if 'html' in table and 'html_inline' not in table:
                    html_path = assets / table['html']
                    try:
                        if html_path.exists():
                            table['html_inline'] = clean_table_html(html_path.read_text(encoding='utf-8'))
                    except Exception:
                        # be conservative: if reading fails, leave as-is and let the
                        # client attempt a fetch (or fall back to raw_text above)
                        pass

    # Build the page from the deduplicated pipeline output (layouts_with_texts.json).
    # A previous build could leave a stale outputs/ui/index.html with un-deduplicated
    # (overlapping/nested) boxes; it must not be used as the data source.
    # OCR is untrusted text: do not let a literal </script> end the data script.
    source_json = json.dumps(combined).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')

    html = TEMPLATE_PATH.read_text(encoding='utf-8').replace('__LAYOUTS_JSON__', source_json)
    if show_abandon:
        html = html.replace('id="show-abandon">', 'id="show-abandon" checked>')
    if asset_root is not None:
        html = html.replace('Every detail. In place.', 'Fresh DocLayout-YOLO → Qwen3-VL-4B')

    # --- assemble a self-contained site for GitHub Pages under docs/ ---
    # GitHub Pages can serve from the "main" branch /docs folder. The page uses
    # relative asset paths, so index.html, originals/ and crops/ must sit together.
    site_dir = site_dir or Path('docs')
    site_dir.mkdir(parents=True, exist_ok=True)

    def sync_tree(src_dir: Path, dst_dir: Path):
        if not src_dir.exists():
            return
        for f in src_dir.rglob('*'):
            if f.is_file():
                target = dst_dir / f.relative_to(src_dir)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, target)

    sync_tree(src_imgs, site_dir / 'originals')
    sync_tree(assets / 'crops', site_dir / 'crops')
    # copy any extracted table HTML/TSV files so client fetches work
    sync_tree(assets / 'tables', site_dir / 'tables')

    (site_dir / 'index.html').write_text(html, encoding='utf-8')
    # tell GitHub Pages not to run the content through Jekyll
    (site_dir / '.nojekyll').write_text('', encoding='utf-8')
    print('Static site written to', site_dir / 'index.html')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Build the static document studio from existing or candidate OCR.')
    parser.add_argument('--source', type=Path, help='Combined layouts JSON (default: existing outputs)')
    parser.add_argument('--site-dir', type=Path, help='Static destination (default: docs)')
    parser.add_argument('--asset-root', type=Path, help='Crop/table root for an isolated pipeline run')
    parser.add_argument('--show-abandon', action='store_true', help='Show OCR-transcribed detector abandon regions by default')
    parser.add_argument('--imgs', type=Path, help='Source scan directory (default: imgs)')
    args = parser.parse_args()
    build(args.source, args.site_dir, args.asset_root, args.show_abandon, imgs_dir=args.imgs)
