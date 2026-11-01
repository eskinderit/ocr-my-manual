import tempfile
import unittest
from pathlib import Path

from tools.build_reconstructed_pdf import load_combined, resolve_render_target


class RenderTargetTests(unittest.TestCase):
    def test_prefers_table_content_before_text_and_crop(self):
        element = {
            "bbox": [0, 0, 10, 10],
            "text": "plain text",
            "crop": "crop.png",
            "table": {"raw_text": "<table><tr><td>Cell</td></tr></table>"},
        }
        self.assertEqual(resolve_render_target(element), ("table", "<table><tr><td>Cell</td></tr></table>"))

    def test_falls_back_to_text_before_crop(self):
        element = {
            "bbox": [0, 0, 10, 10],
            "text": "plain text",
            "crop": "crop.png",
        }
        self.assertEqual(resolve_render_target(element), ("text", "plain text"))

    def test_loads_embedded_json_from_ui_html(self):
        html = '...const data = {"vespa_manual_0.png": {"elements": []}};\n'
        with tempfile.NamedTemporaryFile('w', suffix='.html', delete=False, encoding='utf-8') as tmp:
            tmp.write(html)
            tmp_path = Path(tmp.name)
        try:
            data = load_combined(tmp_path)
            self.assertIn('vespa_manual_0.png', data)
            self.assertEqual(data['vespa_manual_0.png']['elements'], [])
        finally:
            tmp_path.unlink()


if __name__ == "__main__":
    unittest.main()
