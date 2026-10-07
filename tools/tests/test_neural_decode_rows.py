"""The board's GPU inference rows (*-decode, mlp-predict): tables, planning and
the ours runners' decode loops over fake public classes. CPU only; no
bindings, no torch, no GPU launches."""
import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).parents[1]
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location('bench_board_neural', TOOLS / 'bench_board_neural.py')
n = importlib.util.module_from_spec(spec)
spec.loader.exec_module(n)

GPU_INFER = n.DECODE_LANES + ('mlp-predict',)


class Tables(unittest.TestCase):
    def test_gpu_inference_rows_are_gpu_lanes(self):
        for lane in GPU_INFER:
            self.assertIn(lane, n.LANES)
            self.assertEqual(n.DEVICE_OF[lane], 'gpu')
            self.assertIn(lane, n.LANE_TEXT)
            self.assertNotIn('CPU', n.LANE_TEXT[lane][0])

    def test_cpu_infer_rows_stay_cpu(self):
        for lane in n.LANES:
            if lane.endswith('-infer'):
                self.assertEqual(n.DEVICE_OF[lane], 'cpu')

    def test_opponents(self):
        for v in n.VENDORS:
            for lane in ('mamba1-decode', 'mamba2-decode', 'mamba3-decode', 'samba-decode'):
                self.assertEqual(n.opponents(v, lane), ())
            td = n.opponents(v, 'transformer-decode')
            self.assertTrue(td and all('-compile-' not in a and '-cpu-' not in a for a in td))
            self.assertTrue(n.opponents(v, 'mlp-predict'))
            self.assertTrue(all('-cpu-' not in a for a in n.opponents(v, 'mlp-predict')))

    def test_decode_quality_has_host_reference_or_nll(self):
        for lane in ('transformer-decode', 'mamba1-decode', 'mamba2-decode', 'mamba3-decode'):
            self.assertIn(lane, n.FORWARD_REFERENCE_LANES)
        self.assertIn('mean_nll', n.lane_settings('samba-decode')['quality'])
        self.assertIn('decode calls', n.lane_settings('mamba2-decode')['clock'])


class Planning(unittest.TestCase):
    def test_board_plans_gpu_rows_and_never_cpu(self):
        bb_spec = importlib.util.spec_from_file_location('bench_board', TOOLS / 'bench_board.py')
        bb = importlib.util.module_from_spec(bb_spec)
        bb_spec.loader.exec_module(bb)
        for vendor, modes in (('nvidia', ['identical']), ('amd', ['identical']),
                              ('apple', ['fast', 'identical'])):
            races = bb.enforce_gpu_only(bb.plan_races(vendor, modes, families=('neural',)))
            lanes = {r['lane'] for r in races}
            self.assertTrue(set(GPU_INFER) <= lanes, vendor)
            self.assertFalse({l for l in lanes if l.endswith('-infer')}, vendor)


class _Session:
    def __init__(self, log, state):
        self.log, self.state = log, state

    def load_state(self):
        self.log.append(('load_state', self.state.cached_tokens))

    def step(self, x):
        self.state.cached_tokens += 1
        return np.asarray(x) * 2


class _Block:
    log = None

    def __init__(self, w, **kw):
        self.w = w

    def numeric_mode_used(self):
        return 'identical'

    def allocate_state(self, b, *rest):
        self.log.append(('allocate', b) + rest)
        return types.SimpleNamespace(cached_tokens=0)

    def decode_session(self, state):
        self.log.append(('open',))
        return _Session(self.log, state)

    def step(self, x, state):
        return np.asarray(x) * 3

    def forward(self, x):
        raise AssertionError('a decode lane must not call forward')


def fake_module(log):
    blk = type('Blk', (_Block,), {'log': log})
    m2 = type('Mamba2Block', (blk,), {'decode_session': None})
    mod = types.SimpleNamespace(TransformerBlock=blk, Mamba1Block=blk, Mamba2Block=m2,
                                Mamba3Block=m2, numeric_mode=lambda: 'identical',
                                vendor=lambda: 'cuda', __version__='test')
    sys.modules.setdefault(blk.__module__, types.ModuleType(blk.__module__))
    return mod


class DecodeRunners(unittest.TestCase):
    def setUp(self):
        os.environ['MOJOLEARN_NUMERIC_MODE'] = 'identical'
        self.log = []
        self._orig = n._ours_module
        n._ours_module = lambda: fake_module(self.log)

    def tearDown(self):
        n._ours_module = self._orig

    def data(self, model):
        x = np.arange(2 * 5 * 3, dtype=np.float32).reshape(2, 5, 3)
        return {'x': x, 'w:weight': np.ones(1, np.float32)}

    def test_resident_session_decode(self):
        for lane in ('transformer-decode', 'mamba1-decode'):
            self.log.clear()
            d = self.data(lane)
            r = n.OursBlock(lane, 'small', d)
            for _ in range(2):
                r.call()
            y = r.outputs()['y']
            self.assertEqual(y.shape, d['x'].shape)
            np.testing.assert_array_equal(y, d['x'] * 2)
            self.assertEqual(sum(e[0] == 'open' for e in self.log), 1)   # weights once
            self.assertEqual([e for e in self.log if e[0] == 'load_state'],
                             [('load_state', 0), ('load_state', 0)])    # zero reset per round
            self.assertEqual(len(r.digest()), 16)

    def test_per_call_step_decode(self):
        for lane in ('mamba2-decode', 'mamba3-decode'):
            self.log.clear()
            d = self.data(lane)
            r = n.OursBlock(lane, 'small', d)
            r.call()
            r.call()
            np.testing.assert_array_equal(r.outputs()['y'], d['x'] * 3)
            self.assertEqual([e for e in self.log if e[0] == 'allocate'], [('allocate', 2)] * 2)


class SambaDecode(unittest.TestCase):
    def test_stacks_per_token_logits(self):
        os.environ['MOJOLEARN_NUMERIC_MODE'] = 'identical'
        d = n._dims_of('samba-decode', 'small')
        calls = []

        class Cfg:
            def __init__(self, *a, **k):
                pass

            def registry(self):
                return n.samba_registry(d)

            def to_dict(self):
                return {}

        class Stack:
            _blocks = []

            def __init__(self, cfg, **k):
                pass

            def allocate_state(self, b, smax):
                calls.append(('allocate', b, smax))
                return object()

            def step(self, ids, state):
                return np.tile(np.asarray(ids, np.float32)[:, None], (1, d['vocab']))

            def forward(self, ids):
                raise AssertionError('samba-decode must not call forward')

        mod = types.SimpleNamespace(SambaConfig=Cfg, SambaStack=Stack, numeric_mode=lambda: 'identical',
                                    vendor=lambda: 'cuda', __version__='test')
        orig = n._ours_module
        n._ours_module = lambda: mod
        try:
            batches = np.arange(3 * d['batch'] * (d['length'] + 1)).reshape(3, d['batch'], -1) % 256
            r = n.OursSamba('samba-decode', 'small', {'batches': batches.astype(np.int32)})
            r.call()
        finally:
            n._ours_module = orig
        y = r.outputs()['y']
        self.assertEqual(y.shape, (d['batch'], d['length'], d['vocab']))
        np.testing.assert_array_equal(y[:, :, 0], batches[0][:, :-1])
        self.assertEqual(calls, [('allocate', d['batch'], d['length'])])


if __name__ == '__main__':
    unittest.main()
