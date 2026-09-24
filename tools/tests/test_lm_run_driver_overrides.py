"""tools/lm_run_driver.py: a segment's own nvidia_devices and nvidia_gpus win
over the run's for that segment (its rendered --devices, the GPU types its
rental walks and the GPU count it rents), a route shorter than route A plans
and lands, and a spec without the keys renders and rents exactly as before.
CPU only; every subprocess is mocked, nothing is rendered or rented."""
import argparse
import importlib.util
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.argv = [sys.argv[0]]
SPEC = importlib.util.spec_from_file_location(
    'lm_run_driver', Path(__file__).resolve().parents[1] / 'lm_run_driver.py')
drv = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(drv)

H100 = 'NVIDIA H100 80GB HBM3'
WALK = 'NVIDIA H100 80GB HBM3|NVIDIA H200|NVIDIA H100 NVL|NVIDIA H100 PCIe'


def _spec(with_c=True):
    spec = dict(run='runs/t3/x', recipe='r.json', recipe_key='runs/t3/x/recipe.json', tokens_stage='tok',
                lease_minutes=120, dollar_cap=10, nvidia_devices='0,1', nvidia_gpus=WALK, amd_devices='0',
                routes=dict(A=[dict(segment='1', vendor='nvidia', steps=1000),
                               dict(segment='2', vendor='amd', steps=1000),
                               dict(segment='3', vendor='live', steps=400, first='nvidia', shards=[44, 20])],
                            B=[dict(segment='1', vendor='amd', steps=1000),
                               dict(segment='2', vendor='nvidia', steps=1000)]))
    if with_c:
        spec['routes']['C'] = [dict(segment='1', vendor='nvidia', steps=1000, nvidia_devices='0', nvidia_gpus=H100)]
    return spec


def _entry(spec, key):
    return next(e for e in drv.segment_plan(spec) if '%s/%s' % (e['route'], e['segment']) == key)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / 'out'
        self.out.mkdir()
        self.spec_path = Path(self.tmp.name) / 'spec.json'
        self.ledger = drv.Ledger(self.out)
        ck = {'ckpt_%08d.blm' % n: ('%02x' % (n % 251)) * 32 for n in (0, 998, 1000, 1998, 2000)}
        self.ledger.land(dict(route='A', segment='1'), dict(verdict='PASS', checkpoints=ck))
        self.ledger.land(dict(route='A', segment='2'), dict(verdict='PASS', checkpoints=ck))

    def tearDown(self):
        self.tmp.cleanup()

    def load(self, spec):
        self.spec_path.write_text(json.dumps(spec))
        return drv.load_spec(self.spec_path)

    def plan_lines(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            drv.cmd_plan(argparse.Namespace(spec=str(self.spec_path)))
        return buf.getvalue().splitlines()

    def render_argv(self, spec, key, role=None):
        with mock.patch.object(drv.subprocess, 'run') as run:
            drv.render(spec, _entry(spec, key), self.out, self.ledger, role=role)
        return run.call_args.args[0]

    def rent_calls(self, spec, key, env):
        with mock.patch.object(drv.subprocess, 'run', return_value=mock.Mock(returncode=0)) as run, \
                mock.patch.dict(drv.os.environ, env):
            drv.rent_one(spec, _entry(spec, key), self.out / 'body.sh', self.out)
        return [(c.args[0], c.kwargs['env']) for c in run.call_args_list]


class Plan(Base):
    def test_route_c_plans_from_a_and_prints_its_override(self):
        spec = self.load(_spec())
        c = _entry(spec, 'C/1')
        self.assertEqual((c['from_route'], c['from_segment'], c['from_ckpt']), ('A', '1', 'ckpt_00000000.blm'))
        self.assertEqual(c['expect'], 'A/1/chain.jsonl')
        self.assertEqual(c['depends'], [('A', '1')])
        self.assertIsNone(c['replay_ckpt'])
        self.assertIn('C/1 nvidia  steps 0..1000 from A/1/ckpt_00000000.blm expect A/1/chain.jsonl  devices 0  gpus ' + H100,
                      self.plan_lines())
        # nothing waits on route C
        self.assertFalse(any(('C', '1') in e['depends'] for e in drv.segment_plan(spec)))

    def test_route_c_starts_after_ready_a_and_b_segments(self):
        spec = self.load(_spec())
        ready = [_entry(spec, k) for k in ('C/1', 'B/2', 'A/2', 'B/1')]
        self.assertEqual(['%s/%s' % (e['route'], e['segment']) for e in sorted(ready, key=drv._ready_order)],
                         ['B/1', 'A/2', 'B/2', 'C/1'])

    def test_plan_without_overrides_prints_as_before(self):
        self.load(_spec(with_c=False))
        self.assertEqual(self.plan_lines(), [
            'A/1 nvidia  steps 0..1000 from init',
            'A/2 amd     steps 1000..2000 from A/1/ckpt_00001000.blm replay ckpt_00000998.blm',
            'A/3 live    steps 2000..2400 from A/2/ckpt_00002000.blm replay ckpt_00001998.blm',
            'B/1 amd     steps 0..1000 from A/1/ckpt_00000000.blm expect A/1/chain.jsonl',
            'B/2 nvidia  steps 1000..2000 from A/1/ckpt_00001000.blm replay ckpt_00000998.blm expect A/2/chain.jsonl'])


class Render(Base):
    def _expected_one(self, route, seg, arm, devices, frm, extra=()):
        return [sys.executable, str(drv.REPO / 'tools' / 'lm_segment_leg.py'), 'render', '--run', 'runs/t3/x',
                '--recipe', 'r.json', '--recipe-key', 'runs/t3/x/recipe.json', '--arm', arm, '--mode', 'one',
                '--devices', devices, '--tokens-key', 'tok', '--route', route, '--segment', seg,
                '--label', '%s-%s-%s' % (arm, route, seg), '--steps', '1000', '--from', frm, '--seconds', str(8 * 3600),
                '--boundary', str(1000 * int(seg)), *extra, '--out', str(self.out / 'bodies' / ('%s-%s.sh' % (route, seg)))]

    def test_route_c_renders_its_own_devices(self):
        spec = self.load(_spec())
        argv = self.render_argv(spec, 'C/1')
        a0 = self.ledger.landed(dict(route='A', segment='1'))['checkpoints']['ckpt_00000000.blm']
        self.assertEqual(argv, self._expected_one('C', '1', 'nvidia', '0', 'ckpt_00000000.blm', (
            '--from-sha', a0, '--from-key', 'runs/t3/x/A/1/ckpt_00000000.blm',
            '--expect-key', 'runs/t3/x/A/1/chain.jsonl')))

    def test_a_seeded_first_segment_renders_without_a_replay(self):
        # route B's first segment starts from route A's seed: no previous
        # segment of its own, so no arrival replay (this raised KeyError(None))
        spec = self.load(_spec(with_c=False))
        argv = self.render_argv(spec, 'B/1')
        a0 = self.ledger.landed(dict(route='A', segment='1'))['checkpoints']['ckpt_00000000.blm']
        self.assertEqual(argv, self._expected_one('B', '1', 'amd', '0', 'ckpt_00000000.blm', (
            '--from-sha', a0, '--from-key', 'runs/t3/x/A/1/ckpt_00000000.blm',
            '--expect-key', 'runs/t3/x/A/1/chain.jsonl')))

    def test_live_nvidia_side_follows_the_segment(self):
        spec = _spec(with_c=False)
        spec['amd_devices'] = '0,1,2,3'
        spec['routes']['A'][2]['nvidia_devices'] = '0'
        spec = self.load(spec)
        nv = self.render_argv(spec, 'A/3', role='nvidia')
        amd = self.render_argv(spec, 'A/3', role='amd')
        self.assertEqual(nv[nv.index('--devices') + 1], '0')
        self.assertEqual(amd[amd.index('--devices') + 1], '0,1,2,3')

    def test_without_overrides_the_body_is_unchanged(self):
        spec = self.load(_spec(with_c=False))
        self.assertEqual(self.render_argv(spec, 'A/1'), self._expected_one('A', '1', 'nvidia', '0,1', 'init'))
        ck = self.ledger.landed(dict(route='A', segment='1'))['checkpoints']
        self.assertEqual(self.render_argv(spec, 'B/2'), self._expected_one('B', '2', 'nvidia', '0,1', 'ckpt_00001000.blm', (
            '--from-sha', ck['ckpt_00001000.blm'], '--from-key', 'runs/t3/x/A/1/ckpt_00001000.blm',
            '--replay', 'ckpt_00000998.blm', '--replay-sha', ck['ckpt_00000998.blm'],
            '--replay-key', 'runs/t3/x/A/1/ckpt_00000998.blm', '--replay-chain-key', 'runs/t3/x/A/1/chain.jsonl',
            '--expect-key', 'runs/t3/x/A/2/chain.jsonl')))
        live = self.render_argv(spec, 'A/3', role='nvidia')
        self.assertEqual(live[live.index('--devices') + 1], '0,1')


class Rent(Base):
    def test_route_c_rents_one_gpu_of_its_own_type(self):
        spec = self.load(_spec())
        # a GPU count left in the caller's environment does not leak into the segment's rental
        calls = self.rent_calls(spec, 'C/1', {'MOJOLEARN_GEMM_LEG_GPU_COUNT': '2'})
        self.assertEqual(len(calls), 1)
        argv, env = calls[0]
        self.assertEqual(argv, ['sh', 'tools/gemm_remote_leg.sh', 'nvidia', '--rent', '--allow-concurrent',
                                '--segment-lease', '120', '--dollar-cap', '10', '--gpu', H100])
        self.assertEqual(env['MOJOLEARN_GEMM_LEG_GPU_COUNT'], '1')

    def test_without_overrides_the_rental_is_unchanged(self):
        spec = self.load(_spec(with_c=False))
        argv, env = self.rent_calls(spec, 'A/1', {})[0]
        self.assertEqual(argv, ['sh', 'tools/gemm_remote_leg.sh', 'nvidia', '--rent', '--allow-concurrent',
                                '--segment-lease', '120', '--dollar-cap', '10', '--gpu', 'NVIDIA H100 80GB HBM3'])
        self.assertEqual(env['MOJOLEARN_GEMM_LEG_GPU_COUNT'], '2')

    def test_live_coordinator_walks_the_segment_types(self):
        spec = _spec(with_c=False)
        spec['routes']['A'][2].update(nvidia_devices='0', nvidia_gpus=H100)
        spec = self.load(spec)
        with mock.patch.object(drv.subprocess, 'run', return_value=mock.Mock(returncode=0)) as run:
            drv.rent_live(spec, _entry(spec, 'A/3'), 'nv.sh', 'amd.sh', self.out)
        env = run.call_args.kwargs['env']
        self.assertEqual(env['MOJOLEARN_LIVE_NVIDIA_GPUS'], H100)
        self.assertEqual(env['MOJOLEARN_LIVE_NVIDIA_GPU_COUNT'], '1')

    def test_live_without_overrides_sets_no_count(self):
        spec = self.load(_spec(with_c=False))
        with mock.patch.object(drv.subprocess, 'run', return_value=mock.Mock(returncode=0)) as run, \
                mock.patch.dict(drv.os.environ, {}):
            drv.os.environ.pop('MOJOLEARN_LIVE_NVIDIA_GPU_COUNT', None)
            drv.rent_live(spec, _entry(spec, 'A/3'), 'nv.sh', 'amd.sh', self.out)
        env = run.call_args.kwargs['env']
        self.assertEqual(env['MOJOLEARN_LIVE_NVIDIA_GPUS'], WALK)
        self.assertNotIn('MOJOLEARN_LIVE_NVIDIA_GPU_COUNT', env)


class Land(Base):
    def _results(self, name, sha):
        seg = self.out / 'legs' / name / 'remote' / 'segment'
        seg.mkdir(parents=True)
        (seg / 'segment.json').write_text(json.dumps(dict(
            verdict='PASS', steps_completed=1000, disagreements=[], utc_start='t',
            checkpoints=[dict(file='ckpt_00001000.blm', sha256=sha)])))
        return seg.parent.parent

    def test_route_c_is_held_to_a_boundary(self):
        spec = self.load(_spec())
        c = _entry(spec, 'C/1')
        a = self.ledger.landed(dict(route='A', segment='1'))['checkpoints']['ckpt_00001000.blm']
        self.assertTrue(drv.land(spec, c, self._results('C-1', a), self.out, self.ledger))
        self.assertFalse(drv.land(spec, c, self._results('C-1b', 'ff' * 32), self.out, self.ledger))
        self.assertEqual(self.ledger.landed(c)['verdict'], 'FAIL-BOUNDARY')


if __name__ == '__main__':
    unittest.main()
