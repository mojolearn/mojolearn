"""The negative controls of tools/lm_segment.py `run --control`: parsing, the
K override, the swap, the split step and its one-ulp edit, and the stamp every
control line carries. CPU only: the trainer is a NumPy stand-in whose fold is
the same ordered float32 left fold, so a plain run and a control compare
through the real ChainWriter and the real `--expect-chain` verdict."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

import numpy as np

sys.argv = [sys.argv[0]]
SPEC = importlib.util.spec_from_file_location(
    'lm_segment', Path(__file__).resolve().parents[1] / 'lm_segment.py')
seg = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(seg)

_SAVED_MODULES = {}


def setUpModule():
    """`_bytes_of` asks `mojolearn._array` whether a buffer is the package's own
    Array; importing the real package loads a binding, which a checkout without
    built bindings lacks. Nothing here is an Array, so a stub answers."""
    for name in ('mojolearn', 'mojolearn._array'):
        _SAVED_MODULES[name] = sys.modules.get(name)
    if _SAVED_MODULES['mojolearn._array'] is None:
        pkg = _SAVED_MODULES['mojolearn'] or types.ModuleType('mojolearn')
        arr = types.ModuleType('mojolearn._array')
        arr.Array = type('Array', (), {})
        sys.modules.setdefault('mojolearn', pkg)
        sys.modules['mojolearn._array'] = arr


def tearDownModule():
    for name, mod in _SAVED_MODULES.items():
        if mod is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = mod


N = 4096      # parameters of the stand-in model
K = 8         # logical shards of the stand-in recipe


def _grad(ids):
    """A shard's gradient, a pure function of its ids, spread over many
    binades so a reordered float32 sum rounds differently."""
    rng = np.random.default_rng(int(np.asarray(ids).ravel()[0]) + 1)
    return (rng.standard_normal(N) * np.exp2(rng.integers(-12, 12, N))).astype(np.float32)


def _loss(ids):
    return float(np.float32(2.5 + (int(np.asarray(ids).ravel()[0]) % 97) / 64.0))


class FakePar:
    """ParallelByteLanguageModelTrainer's surface, in NumPy."""
    made = []

    def __init__(self, state, *, devices=(0,), logical_shards=1, pool_optimizer=True):
        self.devices, self.logical_shards, self.pool_optimizer = tuple(devices), logical_shards, pool_optimizer
        self.p = np.array(state['parameters'], dtype=np.float32)
        self.m = np.array(state['m'], dtype=np.float32)
        self.v = np.array(state['v'], dtype=np.float32)
        self.flags = np.array(state['flags'], dtype=np.int32)
        self.step_ = int(state['completed_steps'])
        self.lr = np.float32(1e-3)
        self.total = None
        self.seen = []
        FakePar.made.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def set_lr(self, lr):
        self.lr = np.float32(lr)
        return float(self.lr)

    def _update(self, total):
        self.m = (np.float32(0.9) * self.m + np.float32(0.1) * total).astype(np.float32)
        self.v = (np.float32(0.95) * self.v + np.float32(0.05) * total * total).astype(np.float32)
        self.p = (self.p - self.lr * self.m / (np.sqrt(self.v) + np.float32(1e-8))).astype(np.float32)
        self.step_ += 1

    def train_step(self, shards):
        assert len(shards) == self.logical_shards
        self.seen.append([int(np.asarray(x).ravel()[0]) for x in shards])
        total = _grad(shards[0]).copy()
        for x in shards[1:]:
            total = (total + _grad(x)).astype(np.float32)
        self.total = total
        self._update(total)
        return dict(losses=tuple(_loss(x) for x in shards))

    def export_raw(self):
        return dict(parameters=self.p.copy(), m=self.m.copy(), v=self.v.copy(), flags=self.flags.copy())

    def export_gradients(self):
        return self.total.copy()

    # the split step
    def fold_reset(self, prefix=None):
        self.fold = None if prefix is None else np.frombuffer(prefix, dtype=np.float32).copy()

    def _add(self, g):
        self.fold = g.copy() if self.fold is None else (self.fold + g).astype(np.float32)

    def shard_gradient_fold(self, ids):
        self._add(_grad(ids))
        return _loss(ids)

    def shard_gradient(self, ids):
        return _loss(ids), _grad(ids).copy()

    def fold_add(self, g):
        self._add(np.asarray(g, dtype=np.float32))

    def fold_export(self):
        return self.fold.tobytes()

    def apply_gradient(self, total):
        total = np.frombuffer(bytes(memoryview(total).cast('B')), dtype=np.float32).copy()
        self.total = total
        self._update(total)
        return self.step_


class FakeBatches:
    sha256 = 'ab' * 32

    def ids(self, i):
        return np.full((2, 3), i, dtype=np.int32)


def _recipe(path):
    table = ['%08x' % seg._f32_bits(1e-3 * (t + 1)) for t in range(10)]
    path.write_text(json.dumps(dict(schema=seg.RECIPE_SCHEMA, shape=[1] * 9, logical_shards=K, steps=10, seed=1,
                                    schedule=dict(table_f32_hex=table), checkpoint_every=5, boundaries=[],
                                    hash_scheme=seg.SCHEME_V2)))


class Harness(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.recipe = self.dir / 'recipe.json'
        _recipe(self.recipe)
        rng = np.random.default_rng(7)
        self.state = dict(parameters=rng.standard_normal(N).astype(np.float32),
                          m=(rng.standard_normal(N) * 1e-3).astype(np.float32),
                          v=(rng.random(N) * 1e-6).astype(np.float32),
                          flags=np.zeros(3, dtype=np.int32), completed_steps=3, data_schedule={'x': 1})
        self.saved = (seg.open_batches, seg.data_schedule, seg.load_checkpoint)
        seg.open_batches = lambda recipe, tokens: FakeBatches()
        seg.data_schedule = lambda recipe, batches: {'x': 1}
        seg.load_checkpoint = lambda path, zero_moments=False: (
            {k: (v.copy() if hasattr(v, 'copy') else v) for k, v in self.state.items()}, 'cd' * 32)
        fake = types.ModuleType('mojolearn.parallel_training')
        fake.ParallelByteLanguageModelTrainer = FakePar
        self.saved_mod = sys.modules.get('mojolearn.parallel_training')
        sys.modules['mojolearn.parallel_training'] = fake
        FakePar.made = []
        # the reference: a plain two-step run from step 3
        self.reference = self.run_seg('ref')
        self.assertEqual(self.reference[0], 0)

    def tearDown(self):
        seg.open_batches, seg.data_schedule, seg.load_checkpoint = self.saved
        if self.saved_mod is None:
            sys.modules.pop('mojolearn.parallel_training', None)
        else:
            sys.modules['mojolearn.parallel_training'] = self.saved_mod
        self.tmp.cleanup()

    def run_seg(self, name, *extra, expect=None):
        out = self.dir / name
        argv = ['run', '--recipe', str(self.recipe), '--tokens', str(self.dir), '--from', str(self.dir / 'c.blm'),
                '--steps', '2', '--out', str(out), '--no-checkpoints', *extra]
        if expect:
            argv += ['--expect-chain', str(expect)]
        rc = seg.main(argv)
        lines = [json.loads(l) for l in (out / 'chain.jsonl').read_text().splitlines()]
        return rc, lines, json.loads((out / 'segment.json').read_text())

    def ref_chain(self):
        return self.dir / 'ref' / 'chain.jsonl'

    # ---- the harness's own checks: these must PASS

    def test_control_none_passes_and_is_stamped(self):
        rc, lines, segment = self.run_seg('none', '--control', 'none', expect=self.ref_chain())
        self.assertEqual(rc, 0)
        self.assertEqual(segment['verdict'], 'PASS')
        self.assertEqual([l['control'] for l in lines], ['none', 'none'])
        self.assertEqual(segment['control']['stamp'], 'none')
        self.assertNotIn('control', self.reference[1][0])
        self.assertEqual(segment['checkpoints'], [])

    def test_split_without_an_edit_passes(self):
        rc, lines, segment = self.run_seg('split', '--control', 'split', expect=self.ref_chain())
        self.assertEqual(rc, 0, segment.get('disagreements'))
        self.assertEqual(segment['control']['split_shard'], K - 1)
        made = FakePar.made[-1]
        self.assertEqual((made.devices, made.logical_shards, made.pool_optimizer), ((0,), 1, False))

    def test_split_at_a_middle_shard_passes(self):
        rc, _, segment = self.run_seg('split3', '--control', 'split=3', expect=self.ref_chain())
        self.assertEqual(rc, 0, segment.get('disagreements'))

    def test_the_checkpoint_state_is_checked_against_the_expected_line(self):
        # the expected chain has no line at step 3 here, so nothing is claimed
        _, _, segment = self.run_seg('none2', '--control', 'none', expect=self.ref_chain())
        self.assertNotIn('from_state', segment)

    # ---- the negative controls: these must FAIL, at the first step

    def test_shards_63_style_override_fails_and_runs_fewer_shards(self):
        rc, lines, segment = self.run_seg('k', '--control', 'shards=%d' % (K - 1), expect=self.ref_chain())
        self.assertEqual(rc, 1)
        self.assertEqual(len(lines), 1)
        self.assertEqual(len(lines[0]['losses_f32_hex']), K - 1)
        self.assertEqual(FakePar.made[-1].logical_shards, K - 1)
        self.assertEqual(FakePar.made[-1].seen[0], [3 * K + k for k in range(K - 1)])
        d = segment['disagreements'][0]
        self.assertEqual(d['step'], 4)
        self.assertEqual(set(d['fields']), {'state_sha256', 'gradient_sha256', 'losses_f32_hex'})
        self.assertEqual(lines[0]['control'], 'shards=%d' % (K - 1))

    def test_swap_fails_on_losses_and_the_sum(self):
        rc, lines, segment = self.run_seg('swap', '--control', 'swap=5,2', expect=self.ref_chain())
        self.assertEqual(rc, 1)
        self.assertEqual(lines[0]['control'], 'swap=2,5')
        order = [3 * K + k for k in range(K)]
        order[2], order[5] = order[5], order[2]
        self.assertEqual(FakePar.made[-1].seen[0], order)
        self.assertEqual(lines[0]['shard_slots'][2], 5)
        self.assertEqual(set(segment['disagreements'][0]['fields']),
                         {'state_sha256', 'gradient_sha256', 'losses_f32_hex'})

    def test_swapping_the_first_two_moves_the_losses_only(self):
        # g0 + g1 == g1 + g0 in float32: the sum and the state agree
        rc, _, segment = self.run_seg('swap01', '--control', 'swap=0,1', expect=self.ref_chain())
        self.assertEqual(rc, 1)
        self.assertEqual(segment['disagreements'][0]['fields'], ['losses_f32_hex'])

    def test_one_ulp_fails_on_the_sum_and_the_state_not_the_losses(self):
        rc, lines, segment = self.run_seg('ulp', '--control', 'ulp=%d,auto' % (K - 1), expect=self.ref_chain())
        self.assertEqual(rc, 1)
        self.assertEqual(set(segment['disagreements'][0]['fields']), {'state_sha256', 'gradient_sha256'})
        edit = segment['control']['edit']
        self.assertTrue(edit['survives_fold'])
        self.assertEqual(edit['observed_total_bits'], edit['predicted_total_bits_edited'])
        self.assertNotEqual(edit['predicted_total_bits_unedited'], edit['predicted_total_bits_edited'])
        self.assertEqual(int(edit['gradient_bits_after'], 16), int(edit['gradient_bits_before'], 16) + 1)
        self.assertEqual(lines[0]['control'], 'ulp=%d,auto' % (K - 1))

    def test_zero_moments_is_stamped_and_fails(self):
        rc, lines, segment = self.run_seg('zm', '--zero-moments', expect=self.ref_chain())
        self.assertEqual(rc, 1)
        self.assertEqual(lines[0]['control'], 'zero-moments')
        self.assertIn('state_sha256', segment['disagreements'][0]['fields'])

    # ---- refusals

    def test_a_control_never_uploads(self):
        urls = self.dir / 'u.json'
        urls.write_text('{}')
        with self.assertRaises(SystemExit):
            self.run_seg('up', '--control', 'none', '--upload-urls', str(urls))

    def test_the_split_step_takes_one_device(self):
        with self.assertRaises(SystemExit):
            self.run_seg('two', '--control', 'split', '--devices', '0,1')

    def test_a_control_writes_no_checkpoint_even_when_asked(self):
        out = self.dir / 'ck'
        seg.main(['run', '--recipe', str(self.recipe), '--tokens', str(self.dir), '--from', str(self.dir / 'c.blm'),
                  '--steps', '2', '--out', str(out), '--control', 'none'])
        self.assertEqual(sorted(p.name for p in out.iterdir()), ['chain.jsonl', 'log.txt', 'manifest.tsv', 'segment.json'])


class Parse(unittest.TestCase):
    def test_forms(self):
        self.assertEqual(seg.parse_control('none')['text'], 'none')
        self.assertEqual(seg.parse_control('k63'), dict(kind='shards', shards=63, text='shards=63'))
        self.assertEqual(seg.parse_control('shards=63')['shards'], 63)
        self.assertEqual(seg.parse_control('swap=40,5')['text'], 'swap=5,40')
        self.assertEqual(seg.parse_control('split')['shard'], None)
        self.assertEqual(seg.parse_control('split=7')['text'], 'split=7')
        self.assertEqual(seg.parse_control('ulp=63,auto')['index'], 'auto')
        self.assertEqual(seg.parse_control('ulp=63,12')['text'], 'ulp=63,12')

    def test_refusals(self):
        for bad in ('', 'bogus', 'none=1', 'shards=0', 'shards=x', 'swap=3,3', 'swap=1', 'swap=-1,2',
                    'ulp=3', 'ulp=3,-1', 'ulp=a,1'):
            with self.assertRaises(SystemExit, msg=bad):
                seg.parse_control(bad)

    def test_plans_against_k(self):
        self.assertEqual(seg.control_plan(seg.parse_control('shards=63'), 64)[:2], (63, list(range(63))))
        with self.assertRaises(SystemExit):
            seg.control_plan(seg.parse_control('shards=64'), 64)
        with self.assertRaises(SystemExit):
            seg.control_plan(seg.parse_control('swap=5,64'), 64)
        with self.assertRaises(SystemExit):
            seg.control_plan(seg.parse_control('ulp=0,1'), 64)
        self.assertEqual(seg.control_plan(seg.parse_control('split'), 64)[2], 63)
        self.assertEqual(seg.control_plan(None, 64), (64, list(range(64)), None))

    def test_stamp(self):
        self.assertIsNone(seg.control_stamp(None, False))
        self.assertEqual(seg.control_stamp(None, True), 'zero-moments')
        self.assertEqual(seg.control_stamp(seg.parse_control('swap=1,2'), True), 'zero-moments+swap=1,2')


class UlpEdit(unittest.TestCase):
    def test_an_explicit_index_that_the_fold_absorbs_is_reported(self):
        g = np.array([1e-9, 1.0], dtype=np.float32)
        prefix = np.array([1.0, 1.0], dtype=np.float32).tobytes()
        d = seg.ulp_edit(g, prefix, 0)
        self.assertFalse(d['survives_fold'])
        self.assertEqual(g.view(np.uint32)[0], int(d['gradient_bits_after'], 16))

    def test_auto_picks_a_survivor(self):
        g = np.array([1e-9, 3.0, -2.0], dtype=np.float32)
        prefix = np.array([1.0, 1e6, 1.0], dtype=np.float32).tobytes()
        d = seg.ulp_edit(g, prefix, 'auto')
        self.assertEqual(d['index'], 2)  # 3.0 is swallowed by 1e6; -2.0 survives
        self.assertTrue(d['survives_fold'])


if __name__ == '__main__':
    unittest.main()
