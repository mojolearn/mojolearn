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
            cfg(A1, {'u': 'on'}, 'single'),                                           # unmeasured
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


if __name__ == '__main__':
    unittest.main()
