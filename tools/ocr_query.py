import base64
import json
from pathlib import Path
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000",
    api_key="dummy",  # vLLM doesn't require a real key
)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OCR_PROMPT = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.
"""


def get_model_name() -> str:
    """Fetch the first available model from the vLLM server."""
    models = client.models.list()
    return models.data[0].id


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def ocr_image(image_path: str, model: str | None = None) -> str:
    if model is None:
        model = get_model_name()
    b64 = encode_image(image_path)
    ext = Path(image_path).suffix.lstrip(".").lower()
    mime = f"image/{ext}" if ext else "image/png"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": OCR_PROMPT,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    }
                ],
            }
        ],
    )
    return response.choices[0].message.content


def ocr_all_images(img_dir: str = "imgs") -> dict[str, str]:
    model = get_model_name()
    print(f"Using model: {model}")
    results = {}
    for img_path in sorted(Path(img_dir).glob("*.png")):
        print(f"Processing {img_path.name}...")
        try:
            results[img_path.name] = ocr_image(str(img_path), model=model)
        except Exception as exc:
            results[img_path.name] = f"ERROR: {exc}"
        save_json_output(OUTPUT_DIR / "ocr_results.json", results)
    return results


def save_text_output(output_path: Path, content: str) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return output_path


def save_json_output(output_path: Path, data: dict[str, str]) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Single image passed as argument
        image_path = sys.argv[1]
        try:
            text = ocr_image(image_path)
        except Exception as exc:
            text = f"ERROR: {exc}"
        output_path = OUTPUT_DIR / f"{Path(image_path).stem}.txt"
        save_text_output(output_path, text)
        print(f"Saved OCR result to {output_path}")
    else:
        # Process all images in imgs/
        results = ocr_all_images("imgs")
        output_file = OUTPUT_DIR / "ocr_results.json"
        save_json_output(output_file, results)
        print(f"Saved OCR results for {len(results)} images to {output_file}")
