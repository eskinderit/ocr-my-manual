"""Normalize PDF pages or alphabetically ordered photos into numbered scans."""
from pathlib import Path

from PIL import Image, ImageOps

from .pipeline_utils import write_json


IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.tif', '.tiff', '.bmp'}


def prepare_inputs(source: Path, destination: Path, dpi: int = 200) -> list[dict]:
    source = source.resolve()
    if dpi < 72 or dpi > 600:
        raise ValueError('PDF DPI must be between 72 and 600')
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f'Input staging directory is not empty: {destination}')
    if source.is_dir():
        # Alphabetical filename order, deliberately NOT natural/numeric order.
        photos = sorted((p for p in source.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS), key=lambda p: p.name)
        if not photos:
            raise ValueError(f'No supported photos in {source}')
        destination.mkdir(parents=True, exist_ok=True)
        records = []
        for index, photo in enumerate(photos, 1):
            name = f'page_{index:06d}.png'
            with Image.open(photo) as image:
                ImageOps.exif_transpose(image).convert('RGB').save(destination / name)
            records.append({'image': name, 'source': str(photo), 'source_filename': photo.name, 'page': index})
    elif source.is_file() and source.suffix.lower() == '.pdf':
        try:
            import pymupdf
        except ImportError as exc:
            raise RuntimeError('PDF input requires PyMuPDF; install the project requirements') from exc
        destination.mkdir(parents=True, exist_ok=True)
        records = []
        with pymupdf.open(source) as pdf:
            if pdf.needs_pass:
                raise ValueError('Password-protected PDFs must be unlocked before processing')
            if not len(pdf):
                raise ValueError('PDF contains no pages')
            for index, page in enumerate(pdf, 1):
                name = f'page_{index:06d}.png'
                page.get_pixmap(dpi=dpi, alpha=False, colorspace=pymupdf.csRGB).save(destination / name)
                records.append({'image': name, 'source': str(source), 'source_pdf_page': index, 'page': index})
    else:
        raise ValueError('Input must be a PDF file or a directory of photos')
    write_json(destination.parent / 'input_manifest.json', {'ordering': 'PDF page order' if source.is_file() else 'alphabetical filename order (case-sensitive)', 'pages': records})
    return records