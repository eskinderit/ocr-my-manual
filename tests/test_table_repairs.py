import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from vespa_retrieval.repair_qwen_tables import run

from vespa_retrieval.table_html import clean_table_html, validate_table


class TableRepairTests(unittest.TestCase):
    def test_three_columns_required(self):
        self.assertEqual(validate_table('<table><tr><td>a</td><td>b</td><td></td></tr></table>'),1)
        for html in ['<table><tr><td>a</td><td>b</td></tr></table>',
                     '<table><tr><td>a</td><td></td><td>b</td><td></td></tr></table>',
                     '<table><tr><td rowspan="2">a</td><td>b</td><td></td></tr></table>',
                     '<table><tr><td>a</td><td>b</td><td></td></tr>']:
            with self.assertRaises(ValueError):
                validate_table(html)

    def test_column_widths_survive_cleaning(self):
        html='<table><colgroup><col style="width:43%"><col style="width:42%"><col style="width:15%"></colgroup><tr><td>a</td><td>b</td><td></td></tr></table>'
        self.assertIn('<colgroup>',clean_table_html(html))
        self.assertIn('width:15%',clean_table_html(html))
        self.assertEqual(validate_table(clean_table_html(html)),1)

    def test_selected_page_and_endpoint_without_vespa_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            crop = root / 'crops' / 'table.png'
            crop.parent.mkdir()
            Image.new('RGB',(200,100)).save(crop)
            data = {'manual.jpg': {'elements': [{'label':'table','crop':'crops/table.png','table':{'raw_text':'old'}}],
                                   'document_markdown':'old/machine/path.md'}}
            (root/'layouts_with_texts.json').write_text(json.dumps(data))
            result = {'text':'<table><tr><td>A</td><td>B</td><td></td></tr></table>',
                      'response':{'choices':[{'finish_reason':'stop'}]}}
            with patch('vespa_retrieval.repair_qwen_tables.infer', return_value=result) as infer:
                run(root, pages=['manual.jpg'], endpoint='http://localhost:9000/v1/chat/completions')
                self.assertEqual(infer.call_count,1)
                self.assertEqual(infer.call_args.kwargs['endpoint'],'http://localhost:9000/v1/chat/completions')
                run(root, pages=['manual.jpg'])
                self.assertEqual(infer.call_count,1)
            self.assertTrue((root/'layout_documents/manual.md').exists())
            report=json.loads((root/'table_repairs.json').read_text())
            self.assertEqual(report['records'][0]['previous_table']['raw_text'],'old')