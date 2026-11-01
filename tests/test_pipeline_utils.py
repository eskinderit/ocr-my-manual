import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from vespa_retrieval.pipeline_utils import completion_text, image_sort_key, normalize_bbox, prepare_retry_image, write_json


class PipelineUtilsTests(unittest.TestCase):
    def test_arbitrary_page_names_sort_safely(self):
        self.assertEqual(sorted(['cover.png', 'page_10.jpg', 'page_2.png'], key=image_sort_key),
                         ['page_2.png', 'page_10.jpg', 'cover.png'])

    def test_bbox_clipping_and_validation(self):
        self.assertEqual(normalize_bbox([-5, 2.2, 110, 80], 100, 50), [0, 2, 100, 50])
        for box in ([1, 1, 1, 2], [100, 2, 110, 3], [1, 2, 3]):
            with self.assertRaises(ValueError):
                normalize_bbox(box, 100, 50)

    def test_retry_preserves_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'source.png'
            destination = Path(tmp) / 'retry' / 'copy.png'
            Image.new('RGB', (400, 200), 'white').save(source)
            original = source.read_bytes()
            metadata = prepare_retry_image(source, destination, 100)
            with Image.open(destination) as image:
                self.assertEqual(image.size, (100, 50))
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(metadata['method'], 'Lanczos')
            with self.assertRaises(ValueError):
                prepare_retry_image(source, source, 100)

    def test_completion_rejects_empty_and_truncated(self):
        def response(text, reason='stop'):
            return {'text': text, 'response': {'choices': [{'finish_reason': reason}]}}
        self.assertEqual(completion_text(response(' Vespa ')), 'Vespa')
        for result in (response(''), response('partial', 'length')):
            with self.assertRaises(ValueError):
                completion_text(result)

    def test_failed_checkpoint_preserves_previous_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'data.json'
            write_json(path, {'text': 'Italiano è'})
            with patch.object(Path, 'replace', side_effect=OSError('interrupted')):
                with self.assertRaises(OSError):
                    write_json(path, {'text': 'new'})
            self.assertEqual(json.loads(path.read_text()), {'text': 'Italiano è'})
            self.assertFalse(path.with_suffix('.json.tmp').exists())