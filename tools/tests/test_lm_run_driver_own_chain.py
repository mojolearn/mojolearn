"""tools/lm_run_driver.py: a route named in the spec's "own_chain" starts each
segment after its first from ITS OWN previous checkpoint and waits for its
own previous segment, still held to route A's chain for the same segment; a
spec without the key plans exactly as before (B hangs off A's checkpoints).
CPU only; nothing is rendered or rented."""
import importlib.util
from pathlib import Path
import sys
import unittest

sys.argv = [sys.argv[0]]
SPEC = importlib.util.spec_from_file_location(
    'lm_run_driver', Path(__file__).resolve().parents[1] / 'lm_run_driver.py')
drv = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(drv)


def _spec(own=None):
    spec = dict(routes=dict(
        A=[dict(segment='1', vendor='nvidia', steps=1000), dict(segment='2', vendor='nvidia', steps=1000),
           dict(segment='3', vendor='live', steps=400, first='nvidia', shards=[44, 20])],
        B=[dict(segment='1', vendor='amd', steps=1000), dict(segment='2', vendor='live', steps=1000, first='amd'),
           dict(segment='3', vendor='nvidia', steps=400)],
        C=[dict(segment='1', vendor='nvidia', steps=1000), dict(segment='2', vendor='nvidia', steps=1000)]))
    if own is not None:
        spec['own_chain'] = own
    for segs in spec['routes'].values():
        for i, s in enumerate(segs):
            s['index'] = i + 1
    return spec


def _plan(spec):
    return {'%s/%s' % (e['route'], e['segment']): e for e in drv.segment_plan(spec)}


class OwnChain(unittest.TestCase):
    def test_default_hangs_off_route_a(self):
        p = _plan(_spec())
        self.assertEqual((p['B/3']['from_route'], p['B/3']['from_segment']), ('A', '2'))
        self.assertEqual(p['B/3']['depends'], [('A', '2'), ('A', '3')])

    def test_own_chain_starts_from_its_own_checkpoints(self):
        p = _plan(_spec(['B']))
        # the first segment still starts from the shared seed A drew
        self.assertEqual((p['B/1']['from_route'], p['B/1']['from_segment'], p['B/1']['from_ckpt']),
                         ('A', '1', 'ckpt_00000000.blm'))
        for k, prev in (('2', '1'), ('3', '2')):
            e = p['B/' + k]
            self.assertEqual((e['from_route'], e['from_segment']), ('B', prev))
            self.assertEqual(e['depends'], [('B', prev), ('A', k)])
            self.assertEqual(e['expect'], 'A/%s/chain.jsonl' % k)
            self.assertEqual(e['replay_ckpt'], 'ckpt_%08d.blm' % (e['first'] - 2))
        self.assertEqual(p['B/3']['from_ckpt'], 'ckpt_00002000.blm')

    def test_other_routes_unchanged(self):
        p = _plan(_spec(['B']))
        self.assertEqual((p['C/2']['from_route'], p['C/2']['from_segment']), ('A', '1'))
        self.assertEqual((p['A/2']['from_route'], p['A/2']['from_segment']), ('A', '1'))

    def test_plan_shape_sees_the_change(self):
        self.assertNotEqual(drv._plan_shape(drv.segment_plan(_spec())),
                            drv._plan_shape(drv.segment_plan(_spec(['B']))))


if __name__ == '__main__':
    unittest.main()
