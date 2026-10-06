"""Stdlib orchestration checks only; never imports an ML library."""
import copy
import importlib.util
import json
import pathlib
import tempfile
import unittest

path = pathlib.Path(__file__).with_name('selective-neural-repair.py')
spec = importlib.util.spec_from_file_location('repair', path)
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)


class SelectiveNeuralRepair(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name)
        self.box = {'host': {'hostname': 'owned'}, 'gpu': {'name': 'AMD', 'driver': 'same'}}
        self.arms = ['torch-eager-fp32', 'torch-compile-fp32', 'torch-eager-bf16', 'torch-compile-bf16']
        self.race = dict(id='neural/lm-forward/bytes/shape=full', family='neural',
                         arms=self.arms, opponents=self.arms, our_arms={})
        self.old = dict(status='failed', rc=1, started='old', finished='old-end',
                        log='logs/lm.log', race_json='raw/lm-bytes.json', cells=[
                            {'arm': arm, 'status': 'ok' if 'eager' in arm else 'ERROR',
                             'median_ms': 3 if 'eager' in arm else None,
                             'source': 'original', 'device_name': 'AMD'} for arm in self.arms],
                        infer_cells=[{'arm': self.arms[0], 'status': 'ok', 'median_ms': 2}])
        self.write_board()
        (self.root / 'logs').mkdir()
        (self.root / 'raw').mkdir()
        (self.root / 'logs/lm.log').write_text('original log')
        (self.root / 'raw/lm-bytes.json').write_text('original raw')
        (self.root / 'raw/lm-bytes-torch-compile-fp32.log').write_text('original worker failure')
        self.ctx = dict(out=str(self.root), box=copy.deepcopy(self.box))
        self.marker = {'arms_by_race': {self.race['id']: self.arms[1::2]}, 'source_sha': 'repair-sha'}
        self.calls = []

    def write_board(self):
        (self.root / 'board.json').write_text(json.dumps({'box': self.box, 'races': {self.race['id']: self.old}}))

    def fake_run(self, ctx, race):
        self.calls.append(copy.deepcopy((ctx, race)))
        self.assertTrue(ctx['retime'])
        # Simulate the harness replacing the raw result and worker log.
        (self.root / 'raw/lm-bytes.json').write_text('new raw')
        (self.root / 'raw/lm-bytes-torch-compile-fp32.log').write_text('new log')
        return dict(status='done', rc=0, started='new', finished='new-end',
                    cells=[{'arm': a, 'status': 'ok', 'median_ms': 1, 'source': 'new'} for a in race['arms']],
                    infer_cells=[])

    def test_only_selected_failures_run_and_original_cells_and_files_survive(self):
        result = repair.repair_neural(self.ctx, self.race, self.fake_run, self.marker)
        self.assertEqual(self.calls[0][1]['arms'], self.arms[1::2])
        self.assertEqual(result['cells'][:2], self.old['cells'][::2])
        self.assertEqual(result['infer_cells'], self.old['infer_cells'])
        self.assertEqual(result['status'], 'done')
        retained = result['isolated_neural_repair']['preserved_evidence']
        contents = {x['original']: (self.root / x['archive']).read_text() for x in retained}
        self.assertEqual(contents['raw/lm-bytes.json'], 'original raw')
        self.assertEqual(contents['raw/lm-bytes-torch-compile-fp32.log'], 'original worker failure')
        self.assertEqual(contents['logs/lm.log'], 'original log')
        self.assertIn('board.json', contents)

    def test_unrelated_failure_remains_failed(self):
        self.marker['arms_by_race'][self.race['id']] = [self.arms[1]]
        result = repair.repair_neural(self.ctx, self.race, self.fake_run, self.marker)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(next(c for c in result['cells'] if c['arm'] == self.arms[3]), self.old['cells'][3])

    def test_successful_selected_arm_is_never_retimed(self):
        self.marker['arms_by_race'][self.race['id']] = [self.arms[0]]
        self.assertEqual(repair.repair_neural(self.ctx, self.race, self.fake_run, self.marker), self.old)
        self.assertFalse(self.calls)

    def test_hardware_mismatch_is_refused_before_execution(self):
        self.ctx['box']['host']['hostname'] = 'different'
        with self.assertRaisesRegex(ValueError, 'hardware'):
            repair.repair_neural(self.ctx, self.race, self.fake_run, self.marker)
        self.assertFalse(self.calls)

    def test_own_arm_is_refused_before_execution(self):
        self.race['our_arms'] = {'ours': 'identical'}
        with self.assertRaisesRegex(ValueError, 'opponents-only'):
            repair.repair_neural(self.ctx, self.race, self.fake_run, self.marker)
        self.assertFalse(self.calls)


if __name__ == '__main__':
    unittest.main()
