"""Re-OCR text crops into a NEW candidate, never edit the source layouts.

Crop references remain relative to the project's outputs directory, even when
--source or --output is elsewhere. Candidate JSON retains the source schema;
provenance/metadata lives in the adjacent <output-stem>.audit.json sidecar.
Existing destinations are refused. --limit counts attempts (including failures),
not skipped elements. A failed attempt retains its original text and exits 1.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vespa_retrieval.ocr_backends import BACKENDS, infer


OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"
DEFAULT_SOURCE = OUTPUTS / "layouts_with_texts.json"
TEXT_LABELS = {"plain text", "caption", "figure_caption", "table_caption",
               "heading", "headings", "section heading", "section_header"}
HTML_TABLE = re.compile(r"<\s*/?\s*(?:table|thead|tbody|tfoot|tr|td|th)\b", re.I)


def skip_reason(element: dict[str, Any], include_titles: bool) -> str | None:
    """Conservative allowlist; table protection takes precedence over labels."""
    label = element.get("label", "")
    if "table" in element or label == "table":
        return "table structure preserved"
    text = element.get("text")
    if isinstance(text, str) and HTML_TABLE.search(text):
        return "HTML table preserved"
    if label == "title":
        return None if include_titles else "titles disabled (logo hallucination risk)"
    if label not in TEXT_LABELS:
        return "label not eligible"
    return None


def completion_failure(result: dict[str, Any]) -> tuple[str | None, list[Any]]:
    """Only accept a nonempty, normally completed transcription."""
    response = result.get("response")
    if not isinstance(response, dict):
        return "missing model response", []
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not all(
        isinstance(choice, dict) for choice in choices
    ):
        return "missing or malformed completion choices", []
    reasons = [choice.get("finish_reason") for choice in choices]
    if response.get("error") or result.get("error"):
        return "model reported an error", reasons
    if any(reason != "stop" for reason in reasons):
        return "incomplete completion: " + repr(reasons), reasons
    if response.get("truncated") or result.get("truncated"):
        return "model reported truncation", reasons
    usage = response.get("usage") or {}
    config = result.get("request_config") or {}
    tokens, budget = usage.get("completion_tokens"), config.get("max_tokens")
    if isinstance(tokens, (int, float)) and isinstance(budget, (int, float)):
        if budget > 0 and tokens >= budget:
            return "completion reached token budget (possible truncation)", reasons
    text = result.get("text")
    if not isinstance(text, str) or not text.strip():
        return "empty or non-string transcription", reasons
    message = choices[0].get("message")
    if not isinstance(message, dict) or message.get("refusal"):
        return "missing message or model refusal", reasons
    if message.get("content") != text:
        return "transcription does not match model response", reasons
    return None, reasons


def reocr_layouts(
    source: Path = DEFAULT_SOURCE,
    output: Path | None = None,
    *,
    backend: str = "lighton3",
    limit: int | None = None,
    include_titles: bool = False,
) -> dict[str, Any]:
    """Write candidate and audit; return audit, with failure count for the CLI."""
    if backend not in BACKENDS:
        raise ValueError("unknown backend: " + backend)
    if limit is not None and limit < 0:
        raise ValueError("limit must be nonnegative")
    source = Path(source).resolve(strict=True)
    output = Path(output or OUTPUTS / "ocr_candidates" / backend / "layouts_with_texts.json").resolve()
    audit_path = output.with_name(output.stem + ".audit.json")
    for destination in (output, audit_path):
        if destination in {source, DEFAULT_SOURCE.resolve()}:
            raise ValueError("refusing to overwrite source layouts: " + str(destination))
        if destination.exists() or destination.is_symlink():
            raise ValueError("destination must be NEW: " + str(destination))

    source_bytes = source.read_bytes()
    original = json.loads(source_bytes)
    if not isinstance(original, dict) or not all(
        isinstance(page, dict) and isinstance(page.get("elements"), list)
        and all(isinstance(element, dict) for element in page["elements"])
        for page in original.values()
    ):
        raise ValueError("expected page mapping with elements lists")
    candidate = copy.deepcopy(original)
    audit: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "output": str(output),
        "audit": str(audit_path),
        "crop_base": str(OUTPUTS.resolve()),
        "backend": backend,
        "backend_config": copy.deepcopy(BACKENDS[backend]),
        "include_titles": include_titles,
        "limit": limit,
        "attempted": 0, "succeeded": 0, "failed": 0, "skipped": 0,
        "records": [],
    }
    for page_id, page in candidate.items():
        for index, element in enumerate(page["elements"]):
            old_text = copy.deepcopy(element.get("text"))
            record: dict[str, Any] = {
                "page": page_id, "element_index": index,
                "label": element.get("label"), "crop": element.get("crop"),
                "original_text": old_text, "new_text": None,
                "applied_text": copy.deepcopy(old_text),
                "crop_sha256": None, "elapsed_seconds": 0.0,
                "model_response": None, "inference_result": None,
                "finish_reasons": [], "status": "skipped", "reason": None,
            }
            audit["records"].append(record)
            reason = skip_reason(element, include_titles)
            if reason is None and limit is not None and audit["attempted"] >= limit:
                reason = "attempt limit reached"
            if reason is not None:
                record["reason"] = reason
                audit["skipped"] += 1
                continue
            audit["attempted"] += 1
            started = time.perf_counter()
            try:
                crop = element.get("crop")
                if not isinstance(crop, str) or not crop:
                    raise ValueError("missing crop path")
                if Path(crop).is_absolute():
                    raise ValueError("crop must be relative to outputs")
                crop_path = (OUTPUTS / crop).resolve(strict=True)
                if not crop_path.is_relative_to(OUTPUTS.resolve()):
                    raise ValueError("crop escapes outputs directory")
                record["crop_sha256"] = hashlib.sha256(crop_path.read_bytes()).hexdigest()
                result = infer(crop_path, backend)
                record["inference_result"] = copy.deepcopy(result)
                record["model_response"] = copy.deepcopy(result.get("response"))
                record["new_text"] = result.get("text")
                record["model_elapsed_seconds"] = result.get("elapsed_seconds")
                failure, record["finish_reasons"] = completion_failure(result)
                if failure:
                    raise ValueError(failure)
                element["text"] = result["text"]
                record["applied_text"] = result["text"]
                record["status"] = "succeeded"
                audit["succeeded"] += 1
            except Exception as exc:
                record["status"] = "failed"
                record["reason"] = f"{type(exc).__name__}: {exc}"
                audit["failed"] += 1
            finally:
                record["elapsed_seconds"] = time.perf_counter() - started

    # Serialize before opening either file; exclusive mode also protects against
    # a destination created during inference. No schema metadata is injected into
    # the page mapping or existing elements/documents/table objects.
    candidate_json = json.dumps(candidate, ensure_ascii=False, indent=2) + "\n"
    audit_json = json.dumps(audit, ensure_ascii=False, indent=2) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as candidate_file:
        with audit_path.open("x", encoding="utf-8") as audit_file:
            candidate_file.write(candidate_json)
            audit_file.write(audit_json)
    return audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=BACKENDS, default="lighton3")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--include-titles", action="store_true")
    args = parser.parse_args(argv)
    try:
        audit = reocr_layouts(**vars(args))
    except (OSError, ValueError) as exc:
        print(f"reocr_layouts: {exc}", file=sys.stderr)
        return 2
    print(f"Candidate: {audit['output']}\nAudit: {audit['audit']}\n"
          f"Succeeded: {audit['succeeded']}; failed: {audit['failed']}; "
          f"skipped: {audit['skipped']}")
    return 1 if audit["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())