import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pymupdf
from PIL import Image

from main import main
from vespa_retrieval.inputs import prepare_inputs
from vespa_retrieval.pipeline import run_pipeline


class InputTests(unittest.TestCase):
    def test_photos_are_alphabetical_not_numeric(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            photos = root / 'photos'; photos.mkdir()
            for name in ('b.jpg', 'a_2.png', 'a_10.png'):
                Image.new('RGB', (30, 20), 'white').save(photos / name)
            (photos / 'notes.txt').write_text('not a photo')
            records = prepare_inputs(photos, root / 'run/originals')
            self.assertEqual([r['source_filename'] for r in records], ['a_10.png', 'a_2.png', 'b.jpg'])
            self.assertEqual([r['image'] for r in records], ['page_000001.png', 'page_000002.png', 'page_000003.png'])

    def test_pdf_pages_render_in_original_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'manual.pdf'
            with pymupdf.open() as pdf:
                pdf.new_page(width=144, height=72).insert_text((10, 30), 'FIRST')
                pdf.new_page(width=72, height=144).insert_text((10, 30), 'SECOND')
                pdf.save(source)
            records = prepare_inputs(source, root / 'run/originals', dpi=72)
            self.assertEqual([r['source_pdf_page'] for r in records], [1, 2])
            with Image.open(root/'run/originals/page_000001.png') as page:
                self.assertEqual(page.size, (144, 72))
            with Image.open(root/'run/originals/page_000002.png') as page:
                self.assertEqual(page.size, (72, 144))

    def test_invalid_inputs_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            empty = root/'empty'; empty.mkdir()
            for source in (empty, root/'missing.pdf'):
                with self.assertRaises(ValueError):
                    prepare_inputs(source, root/'staging')
            with self.assertRaises(ValueError):
                prepare_inputs(empty, root/'staging', dpi=0)


class FullPipelineTests(unittest.TestCase):
    def fake_detector(self, scans, output, model):
        detections = {p.name: [{'label':'title','bbox':[0,0,20,20],'score':0.9}] for p in scans.glob('*.png')}
        (output/'layouts.json').write_text(json.dumps(detections))

    def test_complete_folder_pipeline_builds_static_site(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            photos = root/'photos'; photos.mkdir()
            for name in ('z.png','a.png'):
                Image.new('RGB',(50,50),'white').save(photos/name)
            output = root/'result'
            result = {'text':'Vespa','response':{'choices':[{'finish_reason':'stop'}]}}
            with patch('vespa_retrieval.pipeline.detect', side_effect=self.fake_detector), patch('vespa_retrieval.run_qwen_layout_ocr.infer', return_value=result):
                site = run_pipeline(photos, output, endpoint='http://example.test/v1/chat/completions')
            self.assertTrue(site.is_file())
            self.assertIn('Vespa',site.read_text())
            self.assertTrue((output/'site/originals/page_000001.png').is_file())
            self.assertTrue((output/'site/crops/page_000001/page_000001_0.png').is_file())
            self.assertEqual(json.loads((output/'pipeline_manifest.json').read_text())['status'],'complete')
            manifest = json.loads((output/'input_manifest.json').read_text())
            self.assertEqual([r['source_filename'] for r in manifest['pages']],['a.png','z.png'])
            with self.assertRaises(FileExistsError):
                run_pipeline(photos, output, endpoint='http://example.test')

    def test_pdf_full_pipeline_and_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root/'manual.pdf'
            with pymupdf.open() as pdf:
                pdf.new_page(width=100,height=100).insert_text((10,30),'Vespa')
                pdf.save(source)
            result = {'text':'Vespa','response':{'choices':[{'finish_reason':'stop'}]}}
            with patch('vespa_retrieval.pipeline.detect', side_effect=self.fake_detector), patch('vespa_retrieval.run_qwen_layout_ocr.infer', return_value=result):
                self.assertEqual(main([str(source),'--output',str(root/'run'),'--endpoint','http://example.test/v1/chat/completions','--dpi','72']),0)
            self.assertTrue((root/'run/site/index.html').exists())

    def test_ocr_failure_recorded_and_no_successful_site(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); photos=root/'photos'; photos.mkdir()
            Image.new('RGB',(50,50)).save(photos/'a.png')
            result={'text':'partial','response':{'choices':[{'finish_reason':'length'}]}}
            with patch('vespa_retrieval.pipeline.detect',side_effect=self.fake_detector), patch('vespa_retrieval.run_qwen_layout_ocr.infer',return_value=result):
                with self.assertRaises(RuntimeError):
                    run_pipeline(photos,root/'run',endpoint='http://example.test')
            self.assertEqual(json.loads((root/'run/pipeline_manifest.json').read_text())['status'],'failed')
            self.assertFalse((root/'run/site/index.html').exists())

    def test_output_cannot_be_inside_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            photos=Path(tmp)
            with self.assertRaises(ValueError):
                run_pipeline(photos,photos/'output')