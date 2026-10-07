import copy
import unittest
import json
import tempfile
import time
from pathlib import Path
from performance_measurement_board import build, write
from performance_measurement_watch import tick


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

    def test_watcher_preserves_manual_and_last_good_rows_on_bad_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / 'state'
            state.mkdir()
            inventory = root / 'inventory.json'
            inventory.write_text(json.dumps(self.inventory))
            source = root / 'source.json'
            source.write_text(json.dumps({'rows': [self.cell]}))
            index = root / 'index.json'
            manual = {'id': 'A01', 'vendor': 'amd', 'status': 'INTERRUPTED', 'evidence': 'older-attempt'}
            index.write_text(json.dumps({'cells': [manual]}))
            (state / 'status.json').write_text(json.dumps({'last_notified': time.time()}))
            config = {'index': str(index), 'inventory': str(inventory), 'sources': [{'path': str(source)}],
                      'out': str(root / 'board'), 'notify_seconds': 999999}
            tick(config, state)
            self.assertEqual(len(json.loads(index.read_text())['cells']), 2)
            source.write_text('{partial write')
            tick(config, state)
            self.assertEqual(len(json.loads(index.read_text())['cells']), 2)
            self.assertTrue(json.loads((state / 'status.json').read_text())['errors'])
            source.unlink()
            tick(config, state)
            self.assertEqual(len(json.loads(index.read_text())['cells']), 2)

    def test_render_rejected_pair_keeps_times_metrics_and_incumbent_defaults(self):
        cell = dict(self.cell, status='QUALITY_FAILED', execution_status='MEASURED_FULL',
                    quality='FAIL', quality_assessment='QUALITY_FAILED',
                    quality_reason='Candidate fails the saved opponent gate',
                    quality_metrics={arm: {'status': 'PENDING', 'metrics': {'ours': {'accuracy': score}}}
                                     for arm, score in [('A', 0.9), ('B', 0.95)]},
                    quality_comparisons={'candidate_vs_opponents': {'accuracy': {'status': 'WORSE'}}},
                    controls={'A': {'defines': ['MOJOLEARN_SWITCH=1']}, 'B': {'defines': []}})
        board = build(self.inventory, {'cells': [cell]})
        with tempfile.TemporaryDirectory() as directory:
            write(board, directory)
            text = (Path(directory) / 'BOARD.md').read_text()
            self.assertIn('| A (s) | B (s) |', text)
            self.assertIn('| 0.008 | 0.01 | — | 0.8000 |', text)
            self.assertIn('QUALITY_FAILED; accuracy A/B=0.9/0.95', text)
            self.assertIn('| accuracy | 0.9 | 0.95 |', text)
            self.assertIn('| define MOJOLEARN_SWITCH | 1 | incumbent default (no explicit override) |', text)
            self.assertIn('candidate_vs_opponents.accuracy.status | WORSE', text)
            self.assertIn('not proof that every requested switch was compiled or reached', text)
            self.assertNotIn('candidate_over_baseline', board['cards'][0]['cells'][0])
            self.assertFalse(board['promotion'])

    def test_render_missing_and_failed_evidence_does_not_infer_configuration_or_time(self):
        cell = dict(self.cell, status='FAILED_OR_INCOMPLETE', execution_status='CANDIDATE_FAILED',
                    failure_reasons=['worker failed | original log retained'],
                    requested_controls={'A': {'defines': ['REQUESTED_ONLY=1']}, 'B': {'defines': []}})
        board = build(self.inventory, {'cells': [cell]})
        with tempfile.TemporaryDirectory() as directory:
            write(board, directory)
            text = (Path(directory) / 'BOARD.md').read_text()
            self.assertIn('Observed A: — s; B: — s.', text)
            self.assertIn('Configuration not recorded.', text)
            self.assertIn('Requested selection differs from the recorded arm configuration', text)
            self.assertIn('Scored quality metrics not recorded', text)
            self.assertIn('worker failed \\| original log retained', text)

    def test_render_deduplicates_profiles_but_keeps_attempts_and_vendor_pages(self):
        controls = {'A': {'defines': ['SWITCH=1'], 'runtime': {'route': 'a|b'}}, 'B': {'defines': []}}
        first = dict(self.cell, controls=controls)
        second = dict(first, case='another-workload', evidence='/retained/second.json')
        board = build(self.inventory, {'cells': [first, second]})
        with tempfile.TemporaryDirectory() as directory:
            write(board, directory)
            for name in ('BOARD', 'amd'):
                text = (Path(directory) / (name + '.md')).read_text()
                self.assertEqual(text.count('<summary>Recorded toggle profile '), 1)
                self.assertEqual(text.count('<a id="attempt-'), 2)
                self.assertIn('runtime.route | a\\|b', text)
                self.assertLess(text.index('## Failed or quality-rejected attempts'),
                                text.index('## Experiments run: toggles, timing and quality'))
            self.assertEqual(json.loads((Path(directory) / 'board.json').read_text()), board)


if __name__ == '__main__':
    unittest.main()
