import copy
import unittest
from performance_measurement_board import build


class MeasurementBoardTest(unittest.TestCase):
    def setUp(self):
        self.inventory = {'campaign': 'test', 'candidates': [
            {'id': 'A01', 'title': 'candidate', 'mode': 'identical', 'vendors': ['amd']}]}
        self.cell = {'id': 'A01', 'vendor': 'amd', 'status': 'MEASURED', 'case': 'gemm',
                     'scope': 'component', 'machine': 'owned-device', 'source_sha': 'frozen',
                     'baseline_machine': 'owned-device', 'candidate_machine': 'owned-device',
                     'baseline_source_sha': 'frozen', 'candidate_source_sha': 'frozen',
                     'artifact_hashes': {'baseline': 'a', 'candidate': 'b'}, 'evidence': '/retained/result.json',
                     'warmups': 1, 'scored_samples': 1, 'baseline_ms': 10, 'candidate_ms': 8, 'returncode': 0}

    def test_component_stays_partial_and_cannot_promote(self):
        board = build(self.inventory, {'cells': [self.cell]})
        self.assertFalse(board['promotion'])
        card = board['cards'][0]
        self.assertEqual(card['status'], 'PARTIAL_MEASUREMENTS_RETAINED')
        self.assertEqual(card['cells'][0]['candidate_over_baseline'], 0.8)
        self.assertEqual(card['cells'][0]['scope'], 'component')

    def test_failed_attempt_never_receives_ratio(self):
        cell = dict(self.cell, status='TIMEOUT', candidate_over_baseline=0.1)
        result = build(self.inventory, {'cells': [cell]})['cards'][0]['cells'][0]
        self.assertNotIn('candidate_over_baseline', result)
        self.assertNotIn('candidate_ms', result)

    def test_incomplete_or_nonfinite_measurement_refused(self):
        for changes in [{'candidate_ms': float('nan')}, {'candidate_ms': 0}, {'returncode': 1},
                        {'warmups': 0}, {'scored_samples': 3}, {'artifact_hashes': {}}, {'vendor': 'nvidia'},
                        {'candidate_machine': 'different-device'}, {'candidate_source_sha': 'different-freeze'}]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                build(self.inventory, {'cells': [dict(self.cell, **changes)]})
        cell = copy.deepcopy(self.cell)
        del cell['source_sha']
        with self.assertRaises(ValueError):
            build(self.inventory, {'cells': [cell]})


if __name__ == '__main__':
    unittest.main()
