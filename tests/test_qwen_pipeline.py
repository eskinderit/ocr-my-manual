import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from vespa_retrieval.run_qwen_layout_ocr import run


class QwenPipelineTests(unittest.TestCase):
    def test_title_footer_table_and_figure_routing_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'run'
            imgs = Path(tmp) / 'imgs'
            root.mkdir(); imgs.mkdir()
            Image.new('RGB', (100,100), 'white').save(imgs / 'page_0.png')
            boxes = [{'label':label,'bbox':[0,i*20,50,i*20+10],'score':0.9} for i,label in enumerate(['title','abandon','table','figure'])]
            (root / 'layouts.json').write_text(json.dumps({'page_0.png':boxes}))
            def fake(image,backend,**kwargs):
                text = '<table><tr><td>150</td></tr></table>' if kwargs['max_tokens'] == 6144 else 'Vespa 150'
                return {'text':text,'response':{'choices':[{'finish_reason':'stop'}]}}
            with patch('vespa_retrieval.run_qwen_layout_ocr.infer',side_effect=fake) as infer:
                self.assertEqual(run(root,imgs),0)
                self.assertEqual(infer.call_count,3)
                self.assertEqual(run(root,imgs),0)
                self.assertEqual(infer.call_count,3)
            elements = json.loads((root/'layouts_with_texts.json').read_text())['page_0.png']['elements']
            self.assertEqual(elements[0]['text'],'Vespa 150')
            self.assertIn('<table',elements[2]['table']['html_inline'])
            self.assertEqual(elements[3]['text'],'')

    def test_truncation_is_failure_not_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); imgs = root/'imgs'; imgs.mkdir()
            Image.new('RGB',(50,50)).save(imgs/'page_0.png')
            (root/'layouts.json').write_text(json.dumps({'page_0.png':[{'label':'title','bbox':[0,0,20,20]}]}))
            with patch('vespa_retrieval.run_qwen_layout_ocr.infer',return_value={'text':'partial','response':{'choices':[{'finish_reason':'length'}]}}):
                self.assertEqual(run(root,imgs),1)
            el = json.loads((root/'layouts_with_texts.json').read_text())['page_0.png']['elements'][0]
            self.assertIn('truncated',el['ocr_error'])

    def test_unnumbered_image_and_custom_endpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); imgs = root / 'scans'; imgs.mkdir()
            Image.new('RGB',(50,50)).save(imgs/'cover.png')
            (root/'layouts.json').write_text(json.dumps({'cover.png':[{'label':'title','bbox':[0,0,20,20]}]}))
            with patch('vespa_retrieval.run_qwen_layout_ocr.infer', return_value={'text':'Vespa','response':{'choices':[{'finish_reason':'stop'}]}}) as infer:
                self.assertEqual(run(root, imgs, endpoint='http://localhost:9000/v1/chat/completions'), 0)
            self.assertEqual(infer.call_args.kwargs['endpoint'], 'http://localhost:9000/v1/chat/completions')


if __name__ == '__main__':
    unittest.main()