"""Dependency-free client for the local native OCR benchmark servers.

Uses only the Python standard library; does not import the base environment's torch.
LightOn's default transcription prompt is image-only; Paddle's is ``OCR:``.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


BACKENDS = {
    "qwen3vl4b": {
        "endpoint": "http://127.0.0.1:8003/v1/chat/completions",
        "model": "qwen3vl4b",
        "prompt": "Transcribe all visible lettering, including single letters inside emblems, stylized words, dates and numbers. Preserve the original language and spelling. Output only visible text. Do not describe the image.",
    },
    "lighton3": {
        "endpoint": "http://127.0.0.1:8001/v1/chat/completions",
        "model": "lightonocr-3-1b",
        "prompt": "",
    },
    "paddle16": {
        "endpoint": "http://127.0.0.1:8002/v1/chat/completions",
        "model": "paddleocr-vl-1.6",
        "prompt": "OCR:",
    },
}


def infer(
    image: str | Path,
    backend: str,
    *,
    endpoint: str | None = None,
    prompt: str | None = None,
    max_tokens: int = 2048,
    timeout: float = 300.0,
) -> dict[str, Any]:
    """Submit one unchanged crop; return OCR text, elapsed time and raw response.

    ``elapsed_seconds`` includes local HTTP and server-side image encoding and
    generation, but not file/base64 or JSON preparation. Native timings are in ``response``.
    No resizing, layout detector or document-parser pipeline is applied here.
    """
    config = BACKENDS[backend]
    path = Path(image).resolve(strict=True)
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    selected_prompt = config["prompt"] if prompt is None else prompt
    content: list[dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}
    ]
    if selected_prompt:
        content.append({"type": "text", "text": selected_prompt})
    payload = {
        "model": config["model"],
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1,
        "stream": False,
        "seed": 42,
    }
    target = endpoint or config["endpoint"]
    request = urllib.request.Request(
        target,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as result:
            response = json.load(result)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{target}: HTTP {exc.code}: {detail}") from exc
    elapsed = time.perf_counter() - started
    return {
        "backend": backend,
        "endpoint": target,
        "image": str(path),
        "prompt": selected_prompt,
        "request_config": {key: value for key, value in payload.items() if key != "messages"},
        "elapsed_seconds": elapsed,
        "text": response["choices"][0]["message"]["content"],
        "response": response,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=BACKENDS)
    parser.add_argument("image", type=Path)
    parser.add_argument("--endpoint")
    parser.add_argument("--prompt", help="Override: grounding, Table Recognition:, etc.")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--output", type=Path, help="Save full benchmark result as JSON")
    args = parser.parse_args()
    result = infer(
        args.image, args.backend, endpoint=args.endpoint, prompt=args.prompt,
        max_tokens=args.max_tokens, timeout=args.timeout,
    )
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()