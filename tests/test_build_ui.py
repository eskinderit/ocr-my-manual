import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vespa_retrieval import build_ui


class BuildUITests(unittest.TestCase):
    def test_alternate_source_assets_and_script_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scans = root / 'custom scans'
            scans.mkdir()
            (scans / 'page.png').write_bytes(b'fixture')
            outputs = root / 'outputs'
            (outputs / 'crops').mkdir(parents=True)
            (outputs / 'crops/one.png').write_bytes(b'crop fixture')
            source = root / 'candidate.json'
            source.write_text(json.dumps({'page': {'elements': [{'text': '</script><script>bad()</script>'}]}}))
            original = source.read_bytes()
            cwd = Path.cwd()
            try:
                os.chdir(root)
                with patch.object(build_ui, 'OUT', outputs):
                    build_ui.build(source, root / 'preview', asset_root=outputs, show_abandon=True, imgs_dir=scans)
            finally:
                os.chdir(cwd)
            generated = (root / 'preview/index.html').read_text()
            self.assertIn('\\u003c/script\\u003e', generated)
            self.assertNotIn('</script><script>bad()', generated)
            self.assertNotIn('__LAYOUTS_JSON__', generated)
            self.assertIn('id="show-abandon" checked', generated)
            self.assertIn('Fresh DocLayout-YOLO', generated)
            self.assertEqual(source.read_bytes(), original)
            self.assertTrue((root / 'preview/originals/page.png').exists())
            self.assertTrue((root / 'preview/crops/one.png').exists())
            self.assertTrue((root / 'preview/.nojekyll').exists())


if __name__ == '__main__':
    unittest.main()