#!/usr/bin/env python3
"""Unit tests for tools/six_lane_grid_decide.py (pure Python; synthetic matrix and evidence, no runs)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import six_lane_grid_decide as D  # noqa: E402


def cfg(algo, assignment, tier, workloads=('istella', 'taxi')):
    cid = 'G.%s.%s' % (algo, tier if tier == 'all_on' else ','.join('%s=%s' % kv for kv in sorted(assignment.items())))
    return dict(id=cid, grid=dict(algorithm=algo, assignment=assignment, tier=tier),
                workloads=['%s@dataset=%s' % (algo, w) for w in workloads])


def tcase(config, wid, verdict, nv, amd):
    return dict(configuration=config, workload_id=wid, verdict=verdict,
                vendors={'nvidia': dict(candidate_over_baseline=dict(scored=nv), evidence='nv'),
                         'amd': dict(candidate_over_baseline=dict(scored=amd), evidence='amd')})


def icase(config, wid, status):
    return dict(configuration_id=config, workload_id=wid, arms={'A': dict(status=status), 'B': dict(status='MATCH')})


class DecideTests(unittest.TestCase):
    def setUp(self):
        A1, A2, A3 = 'classical:a1', 'classical:a2', 'classical:a3'
        self.configs = [
            cfg(A1, {'x': 'on'}, 'single'), cfg(A2, {'x': 'on'}, 'single'),          # PROMOTE
            cfg(A1, {'y': 'on'}, 'single'), cfg(A3, {'y': 'on'}, 'single'),          # SPLIT (A1 faster, A3 slower)
            cfg(A1, {'z': '2'}, 'single'), cfg(A1, {'z': '4'}, 'single'),            # arm 2 noise, arm 4 faster
            cfg(A1, {'w': 'on'}, 'single'),                                           # identity mismatch
            cfg(A1, {'v': 'on'}, 'single'),                                           # quality worse
            cfg('classical:a4', {'u': 'on'}, 'single'),                               # unmeasured (its own algorithm, so a1 stays complete)
            cfg(A1, {'x': 'on', 'y': 'on'}, 'cross'),                                 # interaction: slower together
        ]
        matrix = dict(schema='m', base_main='abc', configurations=self.configs)
        ids = {c['id']: c for c in self.configs}
        t, ident, q = [], [], []

        def measure(cid, verdicts, ratios, identity='MATCH', quality='SAME'):
            for wid, v, r in zip(ids[cid]['workloads'], verdicts, ratios):
                t.append(tcase(cid, wid, v, r, r))
                ident.append(icase(cid, wid, identity))
                lane, ds = wid.split('@dataset=')
                q.append(dict(configuration=cid, lane=lane.split(':')[1], dataset=ds, vendor='nvidia',
                              candidate_vs_baseline=dict(verdict=quality)))

        measure('G.classical:a1.x=on', ['FASTER', 'FASTER'], [0.5, 0.6])
        measure('G.classical:a2.x=on', ['FASTER', 'NO_VERDICT'], [0.8, 1.0])
        measure('G.classical:a1.y=on', ['FASTER', 'FASTER'], [0.7, 0.7])
        measure('G.classical:a3.y=on', ['SLOWER', 'NO_VERDICT'], [1.5, 1.0])
        measure('G.classical:a1.z=2', ['NO_VERDICT', 'NO_VERDICT'], [1.0, 1.01])
        measure('G.classical:a1.z=4', ['FASTER', 'FASTER'], [0.3, 0.4])
        measure('G.classical:a1.w=on', ['FASTER', 'FASTER'], [0.5, 0.5], identity='MISMATCH')
        measure('G.classical:a1.v=on', ['FASTER', 'FASTER'], [0.5, 0.5], quality='WORSE')
        measure('G.classical:a1.x=on,y=on', ['SLOWER', 'SLOWER'], [1.4, 1.3])
        self.dec = D.decide(matrix, D.timing_index([dict(cases=t)]), D.identity_index([dict(cases=ident)]),
                            D.quality_index([dict(rows=q)]))

    def test_arm_recommendations(self):
        c = self.dec['controls']
        self.assertEqual(c['x']['recommendation'], 'PROMOTE')
        self.assertEqual(c['y']['recommendation'], 'SPLIT')
        self.assertEqual(c['y']['arms']['on']['promote_for'], ['classical:a1'])
        self.assertEqual(c['y']['arms']['on']['refuse_for'], ['classical:a3'])
        self.assertEqual(c['w']['recommendation'], 'HOLD_IDENTITY')
        self.assertEqual(c['v']['recommendation'], 'HOLD_QUALITY')
        self.assertEqual(c['u']['recommendation'], 'NOT_MEASURED')

    def test_best_arm_and_noise_is_a_loser(self):
        z = self.dec['controls']['z']
        self.assertEqual(z['best_arm'], '4')
        self.assertEqual(z['recommendation'], 'PROMOTE')
        self.assertEqual(z['arms']['2']['recommendation'], 'DELETE')

    def test_interaction_and_recommended_configuration(self):
        a1 = self.dec['algorithms']['classical:a1']
        self.assertEqual(len(a1['interactions']), 1)
        self.assertEqual(a1['interactions'][0]['predicted'], 'NOT_SLOWER')
        self.assertEqual(a1['recommended']['assignment'], {'z': '4'})
        self.assertIn('G.classical:a1.x=on,y=on', a1['losers'])
        self.assertIsNone(self.dec['algorithms']['classical:a3']['recommended'])

    def test_no_evidence_is_not_measured(self):
        matrix = dict(schema='m', configurations=self.configs)
        dec = D.decide(matrix, {}, {}, {})
        self.assertTrue(all(c['recommendation'] == 'NOT_MEASURED' for c in dec['controls'].values()))
        self.assertTrue(all(a['recommended'] is None for a in dec['algorithms'].values()))

    def test_outputs_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            D.write_outputs(self.dec, tmp)
            doc = json.loads((Path(tmp) / 'grid-decisions.json').read_text())
            self.assertEqual(doc['schema'], D.SCHEMA)
            md = (Path(tmp) / 'GRID_DECISIONS.md').read_text()
            self.assertIn('| y | SPLIT |', md)
            self.assertIn('on for classical:a1; off for classical:a3', md)


def acase(config, wid, verdict, ratio, nv_verdict=None):
    """An M3 FAST case: apple is the one voter; a stray nvidia row must not count."""
    vendors = {'apple': dict(candidate_over_baseline=dict(scored=ratio), verdict=verdict, evidence='m3')}
    if nv_verdict:
        vendors['nvidia'] = dict(candidate_over_baseline=dict(scored=0.01), verdict=nv_verdict, evidence='nv')
    return dict(configuration=config, workload_id=wid, verdict=nv_verdict or verdict, vendors=vendors)


class FastDecideTests(unittest.TestCase):
    def setUp(self):
        A1, A2 = 'algos:a1', 'trees:a2'
        self.configs = [cfg(A1, {'x': 'on'}, 'single'), cfg(A2, {'x': 'on'}, 'single'),
                        cfg(A1, {'q': 'on'}, 'single'), cfg(A1, {'s': 'on'}, 'single'), cfg(A1, {'n': 'on'}, 'single')]
        t, q = [], []

        def measure(cid, verdicts, ratios, quality='SAME', nv=None):
            c = [x for x in self.configs if x['id'] == cid][0]
            for wid, v, r in zip(c['workloads'], verdicts, ratios):
                t.append(acase(cid, wid, v, r, nv))
                lane, ds = wid.split('@dataset=')
                q.append(dict(configuration=cid, lane=lane.split(':')[1], dataset=ds, vendor='apple',
                              candidate_vs_baseline=dict(verdict=quality)))

        measure('G.algos:a1.x=on', ['FASTER', 'NO_VERDICT'], [0.6, 1.0], nv='SLOWER')   # nvidia SLOWER must not vote
        measure('G.trees:a2.x=on', ['FASTER', 'FASTER'], [0.5, 0.7])
        measure('G.algos:a1.q=on', ['FASTER', 'FASTER'], [0.5, 0.5], quality='WORSE')
        measure('G.algos:a1.s=on', ['SLOWER', 'NO_VERDICT'], [1.3, 1.0])
        measure('G.algos:a1.n=on', ['NO_VERDICT', 'NO_VERDICT'], [1.0, 1.0])
        self.matrix = dict(schema='m', mode='fast', configurations=self.configs)
        self.dec = D.decide(self.matrix, D.timing_index([dict(cases=t)], voters=D.FAST_VOTERS), {},
                            D.quality_index([dict(rows=q)]), mode='fast')

    def test_no_identity_needed(self):
        c = self.dec['controls']
        self.assertEqual(c['x']['recommendation'], 'PROMOTE')        # promoted with no identity evidence at all
        rows = self.dec['configurations']['G.algos:a1.x=on']['rows']
        self.assertEqual({r['identity'] for r in rows}, {'NOT_REQUIRED'})
        self.assertEqual(rows[0]['ratios'], {'apple': 0.6})          # one voter: only the M3 ratio counts
        self.assertTrue(self.dec['rule']['identity'].startswith('NOT_REQUIRED'))

    def test_quality_gate_and_losers(self):
        c = self.dec['controls']
        self.assertEqual(c['q']['recommendation'], 'HOLD_QUALITY')
        self.assertEqual(c['s']['recommendation'], 'DELETE')
        self.assertEqual(c['n']['recommendation'], 'DELETE')         # noise is a loser
        self.assertNotIn('HOLD_IDENTITY', {v['recommendation'] for v in c.values()})

    def test_identical_mode_still_requires_identity(self):
        dec = D.decide(self.matrix, D.timing_index([dict(cases=[acase('G.trees:a2.x=on', w, 'FASTER', 0.5) for w in self.configs[1]['workloads']])]),
                       {}, {}, mode='identical')
        self.assertEqual(dec['configurations']['G.trees:a2.x=on']['verdict'], 'IDENTITY_INCOMPLETE')

    def test_fast_markdown(self):
        md = D.render_md(self.dec)
        self.assertIn('# FAST switch grid decisions (Apple M3)', md)
        self.assertIn('| x | PROMOTE |', md)


class Phase2MergeTests(unittest.TestCase):
    def test_phase2_crosses_judged_against_phase1_singles(self):
        A = 'classical:a1'
        p1 = dict(schema='m', base_main='abc', configurations=[cfg(A, {'x': 'on'}, 'single'), cfg(A, {'y': 'on'}, 'single')])
        p2 = dict(schema='m', configurations=[cfg(A, {'x': 'on', 'y': 'on'}, 'cross'), cfg(A, {'x': 'on'}, 'single')])
        m = D.merge_matrices(p1, p2)
        self.assertEqual(len(m['configurations']), 3)  # the phase-1 single is not duplicated
        self.assertEqual(m['merged_phase2']['configurations'], 1)
        t = [tcase(c['id'], w, v, r, r) for c, v, r in ((p1['configurations'][0], 'FASTER', 0.8), (p1['configurations'][1], 'FASTER', 0.9),
                                                         (p2['configurations'][0], 'SLOWER', 1.2)) for w in c['workloads']]
        ident = [icase(x['configuration'], x['workload_id'], 'MATCH') for x in t]
        dec = D.decide(m, D.timing_index([dict(cases=t)]), D.identity_index([dict(cases=ident)]), {})
        inter = dec['algorithms'][A]['interactions']
        self.assertEqual([i['assignment'] for i in inter], [{'x': 'on', 'y': 'on'}])
        self.assertEqual(inter[0]['predicted'], 'NOT_SLOWER')
        self.assertIn('merged_phase2', dec)
        self.assertNotIn('merged_phase2', D.decide(p1, {}, {}, {}))  # unchanged schema without phase 2

    def test_id_collision_with_other_assignment_refused(self):
        A = 'classical:a1'
        c = cfg(A, {'x': 'on'}, 'single')
        bad = dict(c, grid=dict(c['grid'], assignment={'x': 'off'}))
        with self.assertRaises(ValueError):
            D.merge_matrices(dict(configurations=[c]), dict(configurations=[bad]))

    def test_new_tiers_are_interactions(self):
        A = 'classical:a1'
        cs = [cfg(A, {'x': 'on'}, 'single'), cfg(A, {'y': 'on'}, 'single')]
        for tier in ('factorial', 'triple', 'cross_across', 'all_survivors'):
            c = cfg(A, {'x': 'on', 'y': 'on'}, 'cross')
            c['id'] += '.' + tier
            c['grid']['tier'] = tier
            m = dict(configurations=cs + [c])
            t = [tcase(x['id'], w, 'FASTER' if x is not c else 'SLOWER', 1, 1) for x in m['configurations'] for w in x['workloads']]
            ident = [icase(x['configuration'], x['workload_id'], 'MATCH') for x in t]
            dec = D.decide(m, D.timing_index([dict(cases=t)]), D.identity_index([dict(cases=ident)]), {})
            self.assertEqual(len(dec['algorithms'][A]['interactions']), 1, tier)



class IncompleteAndPartialTests(unittest.TestCase):
    """A missing pair is not a neutral result, and nothing is promoted or deleted before the algorithm grid is complete."""

    def test_missing_pair_is_incomplete_not_neutral(self):
        case = dict(configuration='G.classical:a1.x=on', workload_id='classical:a1@dataset=taxi', verdict='NO_VERDICT',
                    phases=dict(scored=dict(verdict='NO_VERDICT', reasons=['nvidia: no A/B pair', 'amd: no A/B pair'])),
                    vendors={})
        idx = D.timing_index([dict(cases=[case])])
        self.assertEqual(idx[('G.classical:a1.x=on', 'classical:a1@dataset=taxi')]['verdict'], 'INCOMPLETE')
        within = dict(case, phases=dict(scored=dict(verdict='NO_VERDICT', reasons=['nvidia: |log ratio| 0.01 within floor 0.05',
                                                                                     'amd: |log ratio| 0.02 within floor 0.05'])))
        idx = D.timing_index([dict(cases=[within])])
        self.assertEqual(idx[('G.classical:a1.x=on', 'classical:a1@dataset=taxi')]['verdict'], 'NO_VERDICT')

    def test_partial_algorithm_grid_defers_the_flip(self):
        A1 = 'classical:a1'
        configs = [cfg(A1, {'x': 'on'}, 'single'), cfg(A1, {'y': 'on'}, 'single'), cfg(A1, {'x': 'on', 'y': 'on'}, 'cross')]
        matrix = dict(schema='m', configurations=configs)
        t, ident = [], []
        for wid in configs[0]['workloads']:  # x alone is measured FASTER on both; y and the cross are not measured yet
            t.append(tcase('G.classical:a1.x=on', wid, 'FASTER', 0.5, 0.5)); ident.append(icase('G.classical:a1.x=on', wid, 'MATCH'))
        dec = D.decide(matrix, D.timing_index([dict(cases=t)]), D.identity_index([dict(cases=ident)]), {})
        self.assertEqual(dec['controls']['x']['arms']['on']['recommendation'], 'PARTIAL_PROMOTE')
        self.assertEqual(dec['controls']['x']['recommendation'], 'PARTIAL_PROMOTE')
        self.assertEqual(dec['controls']['y']['recommendation'], 'NOT_MEASURED')

if __name__ == '__main__':
    unittest.main()
