import unittest

from tools.benchmark_ocr import distance, normalize, score, table_rows


class BenchmarkTests(unittest.TestCase):
    def test_edit_distance_counts_insertions_deletions_substitutions(self):
        self.assertEqual(distance('kitten', 'sitting'), 3)
        self.assertEqual(distance('', 'abc'), 3)

    def test_wrapping_and_transport_markup_not_lexical_errors(self):
        self.assertEqual(normalize('**cer-\nchioni**'), 'cerchioni')
        self.assertNotEqual(normalize('carter', True), normalize('cartoncino', True))

    def test_numeric_punctuation_is_not_hidden_by_content_cer(self):
        sample = {'kind': 'text', 'reference': 'olio 3.50-8', 'numbers': ['3.50-8']}
        self.assertFalse(score(sample, 'olio 3,50-8')['numbers_exact'])

    def test_table_formats(self):
        expected = [['A', '1'], ['B', '2']]
        self.assertEqual(table_rows('<table><tr><td>A</td><td>1</td></tr><tr><td>B</td><td>2</td></tr></table>'), expected)
        self.assertEqual(table_rows('| A | 1 |\n| --- | --- |\n| B | 2 |'), expected)
        self.assertEqual(table_rows('<fcel>A<fcel>1<nl><fcel>B<fcel>2<nl>'), expected)

    def test_missing_extra_empty_and_merged_cells(self):
        self.assertEqual(table_rows('<fcel>A<ecel><nl>'), [['A', '']])
        self.assertEqual(table_rows('<fcel>A<lcel><nl>'), [])
        sample = {'kind': 'table', 'rows': [['A', '1']]}
        self.assertEqual(score(sample, 'A 1')['cell_accuracy'], 0)
        self.assertEqual(score(sample, 'A 1')['text_content_cer'], 0)
        self.assertLess(score(sample, '<fcel>A<fcel>1<fcel>extra<nl>')['cell_accuracy'], 1)


if __name__ == '__main__':
    unittest.main()