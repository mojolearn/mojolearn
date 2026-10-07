"""Tests for tools/six_lane_grid_run.py (metadata glue only; no compile, import of estimators or GPU)."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import six_lane_grid_run as G  # noqa: E402


def cfg(cid, algorithm, bindings, a_keys, priority=0):
    return dict(id=cid, priority=priority, A=dict(defines=['X=1'], environment={}, runtime={}),
                B=dict(defines=[], environment={}, runtime={}),
                grid=dict(algorithm=algorithm, bindings=bindings,
                          builds=dict(nvidia=dict(A=a_keys, B=['b-' + b for b in bindings]))))


def cell(cid, wid, algorithm, vendor='nvidia'):
    return dict(key=G.sha_value([cid, vendor, wid])[:20], configuration=cid, vendor=vendor, mode='identical',
                workload_id=wid, workload=dict(grid_algorithm=algorithm), blockers=['gap'])


class OrderTests(unittest.TestCase):
    def setUp(self):
        self.matrix = dict(configurations=[cfg('G.big.x=on', 'big', ['_m'], ['a1'], 0),
                                           cfg('G.small.y=on', 'small', ['_m'], ['a2'], 1),
                                           cfg('G.small.all_on', 'small', ['_m'], ['a3'], 0)],
                           cells=[cell('G.big.x=on', 'w:big@dataset=t', 'big'),
                                  cell('G.small.y=on', 'w:small@dataset=t', 'small'),
                                  cell('G.small.all_on', 'w:small@dataset=t', 'small'),
                                  cell('G.small.y=on', 'w:small@dataset=t', 'small', vendor='amd')])
        self.plan = dict(algorithms=dict(big=dict(regime='pairwise'), small=dict(regime='factorial')))

    def test_factorial_first_then_priority(self):
        rows = G.ordered_cells(self.matrix, self.plan, 'nvidia')
        self.assertEqual([r['cell']['configuration'] for r in rows], ['G.small.all_on', 'G.small.y=on', 'G.big.x=on'])
        self.assertEqual([r['regime'] for r in rows], ['factorial', 'factorial', 'pairwise'])

    def test_phase_filter(self):
        self.assertEqual([r['cell']['configuration'] for r in G.ordered_cells(self.matrix, self.plan, 'nvidia', 'pairwise')],
                         ['G.big.x=on'])
        self.assertEqual(len(G.ordered_cells(self.matrix, self.plan, 'amd', 'factorial')), 1)


class ArtifactTests(unittest.TestCase):
    def index(self, *failed):
        keys = ['a1', 'b-_m', 'b-_p']
        return {k: dict(status='FAILED' if k in failed else 'COMPILED', receipt='/r/' + k, path='/p/' + k,
                        binding=k, defines=[], artifact_sha256='0' * 64) for k in keys}

    def test_changed_and_prerequisite(self):
        c = cfg('G.a.x=on', 'a', ['_m'], ['a1'])
        arms = G.cell_artifacts(c, 'nvidia', 'w', dict(workload_bindings=['_m', '_p']), {'w': ['_m', '_p']},
                                self.index(), {'_m': 'b-_m', '_p': 'b-_p'})
        self.assertEqual([(a['stem'], a['key'], a['prerequisite']) for a in arms['A']], [('_m', 'a1', False), ('_p', 'b-_p', True)])
        self.assertEqual([(a['stem'], a['key'], a['prerequisite']) for a in arms['B']], [('_m', 'b-_m', False), ('_p', 'b-_p', True)])

    def test_unreached_and_missing_builds_block(self):
        c = cfg('G.a.x=on', 'a', ['_m'], ['a1'])
        with self.assertRaisesRegex(ValueError, 'No changed binding'):
            G.cell_artifacts(c, 'nvidia', 'w', dict(workload_bindings=['_p']), {'w': ['_p']}, self.index(), {'_p': 'b-_p'})
        with self.assertRaisesRegex(ValueError, 'Missing compiled A artifact'):
            G.cell_artifacts(c, 'nvidia', 'w', {}, {'w': ['_m']}, self.index('a1'), {'_m': 'b-_m'})

    def test_build_index_prefers_compiled(self):
        with tempfile.TemporaryDirectory() as tmp:
            for shard, status in (('s1', 'FAILED'), ('s2', 'COMPILED')):
                G.write(Path(tmp) / shard / 'k' / 'receipt.json', dict(key='k', status=status, artifact='/x'))
            self.assertEqual(G.build_index([tmp])['k']['status'], 'COMPILED')


FAKE_RUNNER = r'''
import json, sys, pathlib
args = sys.argv[1:]
config = json.loads(pathlib.Path(args[args.index('--config') + 1]).read_text())
out = pathlib.Path(args[args.index('--output') + 1]); out.mkdir(parents=True, exist_ok=True)
key = config['jobs'][0]['key']
log = pathlib.Path(config['log']); log.write_text(log.read_text() + key + (' retry' if '--retry-failed' in args else '') + '\n')
status = 'MEASUREMENT_FAILED' if config.get('fail') else 'MEASURED_FULL'
(out / 'results.json').write_text(json.dumps({key: dict(status=status)}))
sys.exit(2 if config.get('fail') else 0)
'''


class RunTests(unittest.TestCase):
    def test_skip_measured_keep_going_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            runner = run / 'runner.py'
            runner.write_text(FAKE_RUNNER)
            calls = run / 'calls.txt'
            calls.write_text('')
            cells = []
            for key, extra in (('done', {}), ('bad', dict(fail=True)), ('cut', {}), ('new', {})):
                q = run / 'q' / (key + '.json')
                G.write(q, dict(jobs=[dict(key=key)], log=str(calls), **extra))
                cells.append(dict(key=key, state='READY', queue=str(q)))
            cells.append(dict(key='blocked', state='BLOCKED'))
            G.write(run / 'stage' / 'cells-all.json', dict(vendor='nvidia', cells=cells))
            G.write(run / 'results' / 'done' / 'results.json', dict(done=dict(status='MEASURED_FULL')))
            (run / 'results' / 'cut' / 'cut' / 'attempts' / 'attempt-0001').mkdir(parents=True)  # interrupted
            argv = ['run', '--run', str(run), '--runner', str(runner), '--python', sys.executable]
            self.assertEqual(G.main(argv), 0)
            self.assertEqual(calls.read_text().split('\n')[:-1], ['bad', 'cut retry', 'new'])
            self.assertIn('measured=3 failed=1', (run / 'status.txt').read_text())
            calls.write_text('')
            G.main(argv)  # second pass: nothing to do, the failed cell is not retried
            self.assertEqual(calls.read_text(), '')
            G.main(argv + ['--retry-failed'])
            self.assertEqual(calls.read_text().split('\n')[:-1], ['bad retry'])


def receipt(cid, wid, vendor, a_metrics, b_metrics):
    def result(arm, metrics):
        return dict(vendor=vendor, arm=arm, dataset_sha256='d' * 64, dataset_version='v', dataset_split='s', seed=7,
                    dimensions={'X': [2, 2]}, estimator_settings={'p': 1}, harness_sha256='e' * 64, timed_boundary='b',
                    configuration=dict(defines=['X=1'] if arm == 'A' else [], environment={}, runtime={}),
                    outputs=dict(manifest=[dict(path='$.pred')]), model_state=dict(status='UNAVAILABLE', reason='r'),
                    repeated_use=[], task_quality=dict(metrics=dict(ours=metrics)))
    return dict(status='MEASURED_FULL', source_sha='1' * 40,
                workload=dict(workload_id=wid, mode='identical', implementation_ids=['i'], master_selection=dict(id=cid)),
                runs=[dict(phase='scored', arm=arm, result=result(arm, m)) for arm, m in (('A', a_metrics), ('B', b_metrics))])


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for vendor, auc in (('nvidia', 0.90), ('amd', 0.80)):
            path = self.root / vendor / 'k' / 'k' / 'attempts' / 'attempt-0001' / 'receipt.json'
            G.write(path, receipt('G.a.x=on', 'w@dataset=t', vendor, dict(auc=auc), dict(auc=0.85)))

    def test_quality_rows(self):
        out = self.root / 'quality.json'
        G.main(['quality', str(self.root / 'nvidia'), '--out', str(out)])
        row, = G.read(out)['rows']
        self.assertEqual((row['configuration'], row['workload_id'], row['candidate_vs_baseline']['verdict']),
                         ('G.a.x=on', 'w@dataset=t', 'BETTER'))
        G.main(['quality', str(self.root / 'amd'), '--out', str(out)])
        self.assertEqual(G.read(out)['rows'][0]['candidate_vs_baseline']['verdict'], 'WORSE')

    def test_manifest_pairs_vendors(self):
        out = self.root / 'm.json'
        G.main(['manifest', '--nvidia', str(self.root / 'nvidia'), '--amd', str(self.root / 'amd'), '--out', str(out)])
        case, = G.read(out)['cases']
        self.assertEqual(set(case['columns']), {'nvidia-native', 'amd'})
        self.assertEqual(case['expected']['capture_paths'], dict(outputs=['$.pred']))
        self.assertEqual(case['expected']['configurations']['A']['defines'], ['X=1'])
        # The comparator accepts the scope this builds.
        from six_lane_compare_results import expected_scope
        expected_scope(case)


class ScriptTests(unittest.TestCase):
    def test_shell_scripts_parse(self):
        for name in ('six_lane_grid_run.sh', 'six_lane_grid_decide.sh'):
            subprocess.run(['bash', '-n', str(Path(__file__).resolve().parent / name)], check=True)

    def test_stage_requires_authorization(self):
        with self.assertRaises(SystemExit):
            G.main(['stage', '--vendor', 'nvidia', '--grid', 'g', '--kit', 'k', '--builds', 'b', '--run', '/tmp/x'])


if __name__ == '__main__':
    unittest.main()
