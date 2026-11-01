"""Stdlib-only tests: all inference is mocked; no torch/model imports."""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import reocr_layouts as subject


def completion(text="new text", finish_reason="stop"):
    return {
        "text": text, "elapsed_seconds": 0.25,
        "request_config": {"max_tokens": 2048},
        "response": {
            "model": "mock-model", "usage": {"completion_tokens": 3},
            "choices": [{"finish_reason": finish_reason,
                         "message": {"content": text}}],
        },
    }


class ReocrLayoutsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.outputs = self.root / "outputs"
        self.crop = self.outputs / "crops/page/0.png"
        self.crop.parent.mkdir(parents=True)
        self.crop.write_bytes(b"unchanged crop bytes")
        self.source = self.outputs / "layouts_with_texts.json"
        self.output = self.outputs / "ocr_candidates/lighton3/layouts_with_texts.json"
        self.addCleanup(patch.stopall)
        patch.object(subject, "OUTPUTS", self.outputs).start()
        patch.object(subject, "DEFAULT_SOURCE", self.source).start()
        self.infer = patch.object(subject, "infer", return_value=completion()).start()

    def element(self, label="plain text", **extra):
        return {"label": label, "text": "original", "bbox": [1, 2, 3, 4],
                "crop": "crops/page/0.png", **extra}

    def write_source(self, elements):
        self.original = {"page": {"elements": elements,
                                  "document_markdown": "outputs/layout_documents/page.md",
                                  "custom": {"nested": [1, 2]}}}
        self.source.write_text(json.dumps(self.original), encoding="utf-8")
        self.source_bytes = self.source.read_bytes()

    def run_reocr(self, **kwargs):
        audit = subject.reocr_layouts(self.source, self.output, **kwargs)
        self.assertEqual(self.source.read_bytes(), self.source_bytes)
        self.candidate = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(json.loads(Path(audit["audit"]).read_text(encoding="utf-8")), audit)
        return audit

    def test_tables_are_preserved_even_when_labeled_plain_text(self):
        table = {"tsv": "tables/page/0.tsv", "html": "tables/page/0.html",
                 "rows": [["a", "b"]]}
        self.write_source([
            self.element("table", table=table), self.element(table=table),
            self.element(text='<TABLE class="grid"><tr><td>a</td></tr></TABLE>'),
            self.element("heading", text="<tbody><tr><td>b</td></tr></tbody>"),
        ])
        audit = self.run_reocr(include_titles=True)
        self.infer.assert_not_called()
        self.assertEqual(self.candidate, self.original)
        self.assertEqual(audit["skipped"], 4)

    def test_figures_abandon_titles_and_unknown_labels_are_skipped(self):
        self.write_source([self.element(label) for label in
                           ("figure", "abandon", "title", "formula")])
        self.run_reocr()
        self.infer.assert_not_called()
        self.assertEqual(self.candidate, self.original)

    def test_titles_require_opt_in(self):
        self.write_source([self.element("title"), self.element("figure")])
        audit = self.run_reocr(include_titles=True)
        self.assertEqual(audit["succeeded"], 1)
        self.assertEqual(self.candidate["page"]["elements"][0]["text"], "new text")
        self.assertEqual(self.candidate["page"]["elements"][1], self.original["page"]["elements"][1])

    def test_text_captions_headings_and_metadata_audit(self):
        labels = ("plain text", "figure_caption", "caption", "heading", "section_header")
        self.write_source([self.element(label) for label in labels])
        audit = self.run_reocr(backend="paddle16")
        self.assertEqual(audit["succeeded"], len(labels))
        self.assertEqual(audit["backend"], "paddle16")
        self.assertEqual(audit["crop_base"], str(self.outputs))
        self.assertEqual(audit["source_sha256"], hashlib.sha256(self.source_bytes).hexdigest())
        expected = copy.deepcopy(self.original)
        for element in expected["page"]["elements"]:
            element["text"] = "new text"
        self.assertEqual(self.candidate, expected)
        record = audit["records"][0]
        self.assertEqual(record["original_text"], "original")
        self.assertEqual(record["new_text"], "new text")
        self.assertEqual(record["crop_sha256"], hashlib.sha256(self.crop.read_bytes()).hexdigest())
        self.assertEqual(record["finish_reasons"], ["stop"])
        self.assertEqual(record["model_response"], completion()["response"])
        self.assertEqual(record["inference_result"], completion())
        self.assertGreaterEqual(record["elapsed_seconds"], 0)
        self.assertEqual(record["model_elapsed_seconds"], 0.25)
        self.infer.assert_called_with(self.crop, "paddle16")

    def test_truncation_preserves_original_and_audits_attempt(self):
        self.write_source([self.element()])
        self.infer.return_value = completion("partial", "length")
        audit = self.run_reocr()
        self.assertEqual(self.candidate, self.original)
        self.assertEqual(audit["failed"], 1)
        record = audit["records"][0]
        self.assertEqual(record["new_text"], "partial")
        self.assertEqual(record["applied_text"], "original")
        self.assertEqual(record["finish_reasons"], ["length"])
        self.assertIn("incomplete", record["reason"])

    def test_http_failure_preserves_original_and_continues(self):
        self.write_source([self.element(), self.element()])
        self.infer.side_effect = [RuntimeError("HTTP 500"), completion()]
        audit = self.run_reocr()
        self.assertEqual((audit["failed"], audit["succeeded"]), (1, 1))
        self.assertEqual(self.candidate["page"]["elements"][0]["text"], "original")
        self.assertIn("HTTP 500", audit["records"][0]["reason"])
        self.assertIsNotNone(audit["records"][0]["crop_sha256"])

    def test_invalid_completions_are_rejected(self):
        cases = [completion(" "), completion("partial", None),
                 completion("partial", "content_filter"), {"text": "x"}]
        capped = completion()
        capped["response"]["usage"]["completion_tokens"] = 2048
        cases.append(capped)
        truncated = completion()
        truncated["response"]["truncated"] = True
        cases.append(truncated)
        refused = completion()
        refused["response"]["choices"][0]["message"]["refusal"] = "refused"
        cases.append(refused)
        mismatch = completion()
        mismatch["text"] = "not the model output"
        cases.append(mismatch)
        for result in cases:
            with self.subTest(result=result):
                self.assertIsNotNone(subject.completion_failure(result)[0])

    def test_missing_crop_is_failure_without_inference(self):
        self.write_source([self.element(crop="crops/missing.png"), self.element(crop=None)])
        audit = self.run_reocr()
        self.infer.assert_not_called()
        self.assertEqual(audit["failed"], 2)
        self.assertEqual(self.candidate, self.original)

    def test_crop_escape_and_absolute_paths_are_failures(self):
        outside = self.root / "outside.png"
        outside.write_bytes(b"outside")
        self.write_source([self.element(crop="../outside.png"), self.element(crop=str(outside))])
        audit = self.run_reocr()
        self.infer.assert_not_called()
        self.assertEqual(audit["failed"], 2)
        self.assertEqual(self.candidate, self.original)

    def test_same_input_output_refused_before_inference(self):
        self.write_source([self.element()])
        for output in (self.source, self.outputs / "crops/../layouts_with_texts.json"):
            with self.subTest(output=output), self.assertRaises(ValueError):
                subject.reocr_layouts(self.source, output)
        self.infer.assert_not_called()
        self.assertEqual(self.source.read_bytes(), self.source_bytes)

    def test_existing_destination_and_audit_refused(self):
        self.write_source([self.element()])
        self.output.parent.mkdir(parents=True)
        for path in (self.output, self.output.with_name("layouts_with_texts.audit.json")):
            path.write_bytes(b"protected")
            with self.assertRaises(ValueError):
                subject.reocr_layouts(self.source, self.output)
            self.assertEqual(path.read_bytes(), b"protected")
            path.unlink()
        self.infer.assert_not_called()

    def test_limit_counts_failed_attempts_not_skips(self):
        self.write_source([self.element("figure"), self.element(), self.element()])
        self.infer.side_effect = RuntimeError("failed")
        audit = self.run_reocr(limit=1)
        self.assertEqual(self.infer.call_count, 1)
        self.assertEqual((audit["attempted"], audit["failed"], audit["skipped"]), (1, 1, 2))
        self.assertEqual(self.candidate, self.original)

    def test_zero_limit_and_default_candidate_destination(self):
        self.write_source([self.element()])
        audit = subject.reocr_layouts(self.source, backend="paddle16", limit=0)
        self.assertEqual(Path(audit["output"]), self.outputs / "ocr_candidates/paddle16/layouts_with_texts.json")
        self.assertEqual(audit["attempted"], 0)
        self.infer.assert_not_called()

    def test_custom_source_still_resolves_crops_relative_to_outputs(self):
        self.write_source([self.element()])
        custom = self.root / "custom.json"
        custom.write_bytes(self.source_bytes)
        subject.reocr_layouts(custom, self.output)
        self.infer.assert_called_once_with(self.crop, "lighton3")

    def test_cli_nonzero_on_failure_and_refusal(self):
        self.write_source([self.element()])
        self.infer.return_value = completion("partial", "length")
        args = ["--source", str(self.source), "--output", str(self.output)]
        with patch("sys.stdout"), patch("sys.stderr"):
            self.assertEqual(subject.main(args), 1)
            self.assertEqual(subject.main(args), 2)
            self.assertEqual(subject.main(["--source", str(self.source), "--output", str(self.source)]), 2)

    def test_invalid_schema_and_negative_limit_are_refused(self):
        self.write_source([self.element()])
        with self.assertRaises(ValueError):
            subject.reocr_layouts(self.source, self.output, limit=-1)
        self.source.write_text('{"page": {"elements": "invalid"}}', encoding="utf-8")
        with self.assertRaises(ValueError):
            subject.reocr_layouts(self.source, self.output)
        self.infer.assert_not_called()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()