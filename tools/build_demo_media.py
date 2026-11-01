"""Build a captioned documentation GIF from actual viewer screenshots."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent.parent
MEDIA = ROOT / 'docs' / 'media'
CASES = [
    ('viewer-cover', 'Cover lettering: original scan → detected layout → OCR reconstruction'),
    ('viewer-text', 'Body text: compare the scanned page with searchable OCR output'),
    ('viewer-table', 'Tables: detected region reconstructed as a three-column HTML table'),
]


def build():
    frames = []
    font = ImageFont.load_default(size=18)
    small = ImageFont.load_default(size=14)
    for name, caption in CASES:
        with Image.open(MEDIA / f'{name}.png') as source:
            # The integrated browser capture includes unused space beyond the app.
            screenshot = source.convert('RGB').crop((0, 0, 1182, 822))
        screenshot.save(MEDIA / f'{name}.jpg', quality=90, optimize=True)
        frame = Image.new('RGB', (1182, 900), '#0d1013')
        frame.paste(screenshot, (0, 78))
        draw = ImageDraw.Draw(frame)
        draw.text((20, 14), caption, font=font, fill='#f0b184')
        draw.text((20, 44), 'Actual saved results • viewer slideshow, not processing speed • OCR still needs review', font=small, fill='#dce2e9')
        frames.append(frame.quantize(colors=192))
    frames[0].save(MEDIA / 'viewer-demo.gif', save_all=True, append_images=frames[1:],
                   duration=3500, loop=0, disposal=2, optimize=True)
    print('Created 3 static images and a 10.5-second looping viewer demonstration')


if __name__ == '__main__':
    build()