"""Invented metadata and fake runners only: no estimator, tensor, compiler or GPU."""
import json
import math
from pathlib import Path
import tempfile
import time
import unittest

import six_lane_timing as T


def result(vendor, scored, fit=None, inference=0.0, artifacts=None, legacy=False):
    fit = scored - inference if fit is None else fit
    timings = dict(fit_or_training_seconds=fit, inference_seconds=inference, preparation_seconds=5.0,
                   full_operation_seconds=fit + inference + 5.0)
    if not legacy:
        timings.update(scored_seconds=scored, scored_fit_seconds=fit, scored_inference_seconds=inference)
    return dict(vendor=vendor, timings=timings, loaded_artifacts=artifacts or {'/p/a.so': 'a' * 64})


def receipt(workload, vendor, a, b, aa=False, config='cfg'):
    job = dict(key='k-' + workload + vendor, workload_id=workload, master_selection=dict(id=config))
    if aa:
        job['aa_noise_floor'] = dict(source_key='k')
    return dict(status='MEASURED_FULL', mode='identical', source_sha='1' * 40, workload=job,
                runs=[dict(phase='scored', arm='A', returncode=0, result=a),
                      dict(phase='scored', arm='B', returncode=0, result=b)])


class TimingTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def dump(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value))
        return str(path)

    def test_scored_excludes_preparation_and_legacy_fallback(self):
        r = result('nvidia', 1.5, inference=0.1)
        self.assertEqual(T.scored(r['timings'])['scored'], 1.5)
        legacy = T.scored(result('nvidia', 1.5, inference=0.1, legacy=True)['timings'])
        self.assertAlmostEqual(legacy['scored'], 1.5)
        self.assertTrue(legacy['source'].startswith('legacy'))
        # The LDA istella case: preparation 5.22 vs 2.15 s, fit 1.412 vs 1.428 s.
        a = dict(timings=dict(fit_or_training_seconds=1.412, inference_seconds=0.106, preparation_seconds=5.224,
                              full_operation_seconds=6.75))
        b = dict(timings=dict(fit_or_training_seconds=1.428, inference_seconds=0.104, preparation_seconds=2.153,
                              full_operation_seconds=3.693))
        self.assertLess(abs(T.log_ratios(a, b)['scored']), 0.02)

    def floors(self, ratios):
        files = []
        for (workload, vendor), ratio in ratios.items():
            files.append(self.dump('aa-%s-%s.json' % (workload, vendor),
                                   receipt(workload, vendor, result(vendor, ratio), result(vendor, 1.0), aa=True)))
        return T.floors(files)

    def test_floor_rejects_non_identical_builds_and_second_pair(self):
        good = self.dump('aa1.json', receipt('w', 'nvidia', result('nvidia', 1.05), result('nvidia', 1.0), aa=True))
        dup = self.dump('aa2.json', receipt('w', 'nvidia', result('nvidia', 1.2), result('nvidia', 1.0), aa=True))
        bad = self.dump('aa3.json', receipt('w', 'amd', result('amd', 1.0, artifacts={'/x': 'b' * 64}),
                                            result('amd', 1.0), aa=True))
        doc = T.floors([good, dup, bad])
        self.assertAlmostEqual(doc['floors']['w|nvidia']['floor']['scored'], math.log(1.05))
        self.assertNotIn('w|amd', doc['floors'])
        self.assertEqual(len(doc['rejected']), 2)

    def verdict(self, nv, amd, floors=None):
        floors = floors or self.floors({('w', 'nvidia'): 1.05, ('w', 'amd'): 1.05})
        files = [self.dump('ab-nv.json', receipt('w', 'nvidia', result('nvidia', nv), result('nvidia', 1.0))),
                 self.dump('ab-amd.json', receipt('w', 'amd', result('amd', amd), result('amd', 1.0)))]
        return T.verdicts(files, floors)['cases'][0]

    def test_verdict_is_the_vendor_average_against_the_mean_floor(self):
        self.assertEqual(self.verdict(0.8, 0.85)['verdict'], 'FASTER')
        self.assertEqual(self.verdict(1.3, 1.2)['verdict'], 'SLOWER')
        self.assertEqual(self.verdict(0.8, 1.02)['verdict'], 'FASTER')         # AMD inside its floor; the average is faster
        split = self.verdict(0.8, 1.3)                                         # vendors disagree: average ~1.02 = within floor, split flagged
        self.assertEqual(split['verdict'], 'NO_VERDICT')
        self.assertTrue(any('vendor split' in r for r in split['phases']['scored']['reasons']))
        self.assertEqual(self.verdict(0.5, 1.3)['verdict'], 'FASTER')          # average 0.81x wins despite the AMD split
        none = self.verdict(0.5, 0.5, floors=dict(floors={}))
        self.assertEqual(none['verdict'], 'NO_VERDICT')
        self.assertIn('nvidia: no A/A floor', none['phases']['scored']['reasons'])

    def test_aa_job_runs_incumbent_on_both_arms(self):
        job = dict(key='orig', workload_id='w', mode='identical', implementation_ids=['I.X'],
                   master_selection=dict(id='c', A=dict(defines=['MOJOLEARN_X=1'], environment={}, runtime={}),
                                         B=dict(defines=[], environment={}, runtime={})),
                   artifact_provenance=dict(A=[dict(path='/a/x.so', sha256='a' * 64)], B=[dict(path='/b/x.so', sha256='b' * 64)]),
                   arms=dict(A=dict(configuration='A', argv=['w', '--recipe', 'r', '--arm', 'A'], environment={'E': '1'}),
                             B=dict(configuration='B', argv=['w', '--recipe', 'r', '--arm', 'B'], environment={})))
        worker = dict(vendor='nvidia', job=json.loads(json.dumps(job)), packages=dict(A='/a', B='/b'),
                      artifact_provenance=job['artifact_provenance'], execution_authorized=True)
        q, w = T.aa_job(job, worker)
        self.assertNotEqual(q['key'], 'orig')
        self.assertEqual(w['job']['key'], 'orig')
        for doc in (q, w['job']):
            self.assertEqual(doc['master_selection']['A'], doc['master_selection']['B'])
            self.assertEqual(doc['artifact_provenance']['A'], doc['artifact_provenance']['B'])
            self.assertEqual(doc['arms']['A']['environment'], {})
            self.assertEqual(doc['arms']['A']['argv'][-1], 'A')
            self.assertEqual(doc['implementation_ids'], ['I.X'])
        self.assertEqual(w['packages'], dict(A='/b', B='/b'))
        self.assertFalse(w['execution_authorized'])
        self.assertEqual(job['master_selection']['A']['defines'], ['MOJOLEARN_X=1'])  # input untouched

    def test_aa_queue_one_pair_per_workload(self):
        workers = self.root / 'q-workers'
        workers.mkdir()
        jobs = []
        for key in ('k1', 'k2'):
            job = dict(key=key, workload_id='same', mode='identical', implementation_ids=['I'], blocked=[],
                       master_selection=dict(id=key, A=dict(defines=['D']), B=dict(defines=[])),
                       artifact_provenance=dict(A=[], B=[]),
                       arms=dict(A=dict(argv=['w', '--recipe', str(workers / (key + '.json')), '--arm', 'A']),
                                 B=dict(argv=['w', '--recipe', str(workers / (key + '.json')), '--arm', 'B'])))
            (workers / (key + '.json')).write_text(json.dumps(dict(vendor='amd', job=job, packages=dict(A='a', B='b'),
                                                                   artifact_provenance=job['artifact_provenance'])))
            jobs.append(job)
        doc = T.aa_queue(dict(jobs=jobs, vendor='amd'), workers, self.root / 'aa.json')
        self.assertEqual(len(doc['jobs']), 1)
        recipe = doc['jobs'][0]['arms']['A']['argv'][2]
        self.assertTrue(Path(recipe).is_file())
        self.assertEqual(json.loads(Path(recipe).read_text())['job']['key'], 'k1')


class WorkerBoundaryTest(unittest.TestCase):
    """complete_operation keeps capture and inference setup outside the scored clock."""

    def test_classical_and_forest_boundaries(self):
        from six_lane_ab_worker import complete_operation

        class Inf:
            def call(self): time.sleep(0.02)
            def sync(self): pass
            def outputs(self): time.sleep(0.05); return {'p': 1}

        class Module:
            def infer_runner(self, lane, runner, data): time.sleep(0.05); return Inf()

        class Runner:
            def call(self): time.sleep(0.02)
            def sync(self): pass
            def outputs(self): time.sleep(0.05); return {'o': 1}

        _, _, t = complete_operation(Module(), 'classical', dict(inference='separate', lane='x'), {}, Runner())
        self.assertLess(t['scored_seconds'], 0.09)
        self.assertGreater(t['scored_inference_seconds'], 0.015)
        self.assertGreater(t['inference_seconds'], 0.1)  # old field keeps setup + capture

        class Forest:
            out = None
            predicted = 0
            def call(self): pass
            def sync(self): pass
            def infer(self): self.predicted += 1; time.sleep(0.03); self.out = 1; return True
            def outputs(self):
                if self.out is None: self.infer()
                return {'prediction': self.out}

        forest = Forest()
        _, _, t = complete_operation(None, 'forest', dict(inference='included_in_operation'), {}, forest)
        self.assertEqual(forest.predicted, 1)
        self.assertGreater(t['scored_inference_seconds'], 0.025)


class ComparatorTimingTest(unittest.TestCase):
    def test_timing_verdict_from_columns(self):
        from six_lane_compare_results import timing_verdict
        case = dict(workload_id='w', mode='identical', columns={
            'nvidia-native': dict(arms=dict(A=dict(result=result('nvidia', 0.8)), B=dict(result=result('nvidia', 1.0)))),
            'amd': dict(arms=dict(A=dict(result=result('amd', 0.7)), B=dict(result=result('amd', 1.0))))})
        floors = dict(floors={'w|nvidia': dict(floor=dict(scored=0.1, fit=0.1, inference=0.1), evidence='e'),
                              'w|amd': dict(floor=dict(scored=0.1, fit=0.1, inference=0.1), evidence='e')})
        self.assertEqual(timing_verdict(case, floors)['verdict'], 'FASTER')
        self.assertEqual(timing_verdict(case, None)['verdict'], 'NO_VERDICT')


if __name__ == '__main__':
    unittest.main()
