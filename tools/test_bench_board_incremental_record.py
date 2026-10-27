"""Rebuilt installed components must not be presented as a stock PyPI wheel."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('board', Path(__file__).with_name('bench_board.py'))
board = importlib.util.module_from_spec(spec)
spec.loader.exec_module(board)


class IncrementalRecord(unittest.TestCase):
    def test_retains_base_provenance_and_labels_replacements(self):
        wheel = dict(source='pypi', sha256='base', pypi={'all_digests_match': True})
        identity = {'component_sources': {'host/repair.so': {'source_commit': 'repair'}}}
        result = board.installed_artifact_record(wheel, identity)
        self.assertEqual(result['source'], 'incremental')
        self.assertEqual(result['base_distribution_source'], 'pypi')
        self.assertEqual(result['pypi'], wheel['pypi'])
        self.assertEqual(result['component_sources'], identity['component_sources'])
        self.assertEqual(wheel['source'], 'pypi')
        self.assertEqual(board.installed_artifact_record(result, identity), result)

    def test_unmodified_wheel_keeps_its_provenance(self):
        wheel = dict(source='candidate')
        self.assertEqual(board.installed_artifact_record(wheel, {}), wheel)


if __name__ == '__main__':
    unittest.main()
