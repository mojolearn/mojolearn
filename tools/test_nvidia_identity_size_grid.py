"""Authored file-only grid tests; root alone executes. No NumPy/native imports."""
from argparse import Namespace
import json
from pathlib import Path
import tempfile
import unittest
import nvidia_identity_size_grid as grid


class GridPolicy(unittest.TestCase):
    def args(self, out, all=False):
        return Namespace(out=out, all=all, lanes=None, sizes=None, external_modes=None,
                         job_seconds=180, total_seconds=2400)

    def test_default_is_one_identical_only_job(self):
        record = grid.plan(self.args(Path('/unused-output')))
        self.assertEqual(record['status'], 'PLAN_ONLY')
        self.assertEqual(len(record['jobs']), 1)
        job = record['jobs'][0]
        self.assertEqual(job['shape'], {'dimension': 512})
        self.assertEqual(job['ours_mode'], 'identical-only')
        self.assertEqual(job['command'][job['command'].index('--ours-mode') + 1], 'identical-only')
        self.assertNotIn('legacy-fast-and-identical', job['command'])

    def test_full_grid_does_not_launch_unproved_cells(self):
        jobs = grid.plan(self.args(Path('/unused-output'), all=True))['jobs']
        self.assertEqual(len(jobs), 36)
        index = {(j['lane'], j['size'], j['external_mode']): j for j in jobs}
        self.assertEqual(index['knn', 'large', 'fast']['status'], 'MEMORY_ADMISSION_REFUSED')
        for size in grid.SIZES:
            self.assertEqual(index['knn', size, 'deterministic']['status'], 'DETERMINISTIC_SUPPORT_UNVERIFIED')
            self.assertEqual(index['gbdt', size, 'deterministic']['status'], 'DOCUMENTED_UNSUPPORTED')
        self.assertEqual(index['umap', 'large', 'fast']['status'], 'ADAPTER_REQUIRED')
        self.assertEqual(index['gbdt', 'medium', 'fast']['status'], 'ADAPTER_REQUIRED')
        self.assertEqual(len({j['command'][j['command'].index('--out') + 1] for j in jobs}), 36)

    def result(self):
        return dict(status='PASSED', ours_mode='identical-only', active_arms=['identical', 'external'],
            external_mode='fast', args={'lane': 'gemv'},
            metadata={'identical': {'mode': 'identical', 'cuda': 'fixture', 'nvidia_smi': 'fixture'},
                      'external': {'cuda': 'fixture', 'nvidia_smi': 'fixture'}},
            records=[dict(arm=arm, round=r, warmup=r == 0, ms=1., hashes=['fixture'])
                     for arm in ('identical', 'external') for r in range(8)],
            accuracy=[{'passed': True}], summary={'identical': {}, 'external': {}})

    def test_retained_fast_arm_missing_witness_and_repeat_divergence_refuse(self):
        for defect in (None, 'fast', 'witness', 'repeat', 'sample', 'accuracy'):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as tmp:
                record = self.result()
                if defect == 'fast': record['active_arms'].insert(0, 'fast')
                elif defect == 'witness': record['metadata']['identical']['mode'] = 'fast'
                elif defect == 'repeat': record['records'][1]['hashes'] = ['different']
                elif defect == 'sample': record['records'].pop()
                elif defect == 'accuracy': record['accuracy'][0]['passed'] = False
                path = Path(tmp) / 'results.json'
                path.write_text(json.dumps(record))
                job = {'external_mode': 'fast', 'lane': 'gemv'}
                if defect:
                    with self.assertRaises(ValueError): grid.admit_result(path, job)
                else:
                    self.assertIn('results_sha256', grid.admit_result(path, job))


if __name__ == '__main__':
    unittest.main()
