"""The overlapped per-step digests of tools/lm_segment.py `run`: the threaded
digests equal the definition (`_hash_arrays`, `_hash_gradient`, and an
independent sequential sha256 written here) on odd sizes, the empty buffer,
buffers shorter than the slice count and single-bit buffers; the snapshot
reuses its host buffers and makes the export call `export_raw` makes; the
overlapped loop writes the same chain lines in the same order as the
synchronous loop, stops at the same first differing step when a disagreement
is injected, and hands the checkpoint writer the same state. CPU only, no
model data: the trainer is a NumPy stand-in."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
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
    """`_bytes_of` asks `mojolearn._array` whether a buffer is the package's
    own Array; nothing here is one, so a stub answers when the real package
    (which loads a binding) is not importable."""
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


# ---------------------------------------------------------------- an independent definition

def ref_sliced(data, slices=8):
    """sliced-sha256-8.v2 of one buffer, written from the scheme's words:
    the sha256 of the concatenated hex sha256s of eight equal byte ranges."""
    data = bytes(data)
    n = len(data)
    hexes = ''.join(hashlib.sha256(data[n * i // slices:n * (i + 1) // slices]).hexdigest() for i in range(slices))
    return hashlib.sha256(hexes.encode()).hexdigest()


def ref_state(arrays, scheme):
    keys = ('parameters', 'm', 'v', 'flags')
    if scheme == seg.SCHEME_V1:
        return hashlib.sha256(b''.join(bytes(arrays[k]) for k in keys)).hexdigest()
    return hashlib.sha256(''.join(ref_sliced(arrays[k]) for k in keys).encode()).hexdigest()


def ref_gradient(g, scheme):
    return hashlib.sha256(bytes(g)).hexdigest() if scheme == seg.SCHEME_V1 else ref_sliced(g)


SIZES = (0, 1, 3, 7, 8, 9, 15, 17, 1023, 4096 + 5, 65536 * 3 + 7, (1 << 20) + 13)
SCHEMES = (seg.SCHEME_V1, seg.SCHEME_V2)


class DigestsEqualTheDefinition(unittest.TestCase):
    def test_the_reference_is_the_definition(self):
        rng = np.random.default_rng(1)
        for n in SIZES:
            b = rng.integers(0, 256, n, dtype=np.uint8).tobytes()
            self.assertEqual(ref_sliced(b), seg._sha_sliced(b), n)

    def test_random_buffers_of_odd_sizes(self):
        rng = np.random.default_rng(2)
        for scheme in SCHEMES:
            for n in SIZES:
                arrays = {k: rng.integers(0, 256, n + d, dtype=np.uint8).tobytes()
                          for k, d in (('parameters', 0), ('m', 1), ('v', 2), ('flags', 3))}
                g = rng.integers(0, 256, n + 5, dtype=np.uint8)
                got = seg.Digests(arrays, g, scheme).result()
                want = (seg._hash_arrays(arrays, scheme), seg._hash_gradient(g, scheme))
                self.assertEqual(got, want, (scheme, n))
                self.assertEqual(got, (ref_state(arrays, scheme), ref_gradient(g, scheme)), (scheme, n))

    def test_float_and_int_arrays_as_the_trainer_hands_them_out(self):
        rng = np.random.default_rng(3)
        for scheme in SCHEMES:
            arrays = dict(parameters=rng.standard_normal(100003).astype(np.float32),
                          m=rng.standard_normal(100003).astype(np.float32),
                          v=rng.random(100003).astype(np.float32), flags=np.array([0, 1, 1], dtype=np.int32))
            g = rng.standard_normal(100003).astype(np.float32)
            self.assertEqual(seg.Digests(arrays, g, scheme).result(),
                             (seg._hash_arrays(arrays, scheme), seg._hash_gradient(g, scheme)))

    def test_empty_everything(self):
        for scheme in SCHEMES:
            arrays = {k: b'' for k in seg.ARRAYS}
            self.assertEqual(seg.Digests(arrays, b'', scheme).result(),
                             (seg._hash_arrays(arrays, scheme), seg._hash_gradient(b'', scheme)))

    def test_no_gradient(self):
        arrays = {k: bytes(range(11)) for k in seg.ARRAYS}
        for scheme in SCHEMES:
            self.assertEqual(seg.Digests(arrays, None, scheme).result(), (seg._hash_arrays(arrays, scheme), None))

    def test_one_bit_in_each_slice(self):
        """All-zero buffers with ONE bit set, in every slice in turn, at both
        edges of the slice: each digest equals the definition and every one
        differs from the all-zero digest and from every other."""
        n = 8 * 4096 + 3
        zeros = bytes(n)
        seen = set()
        for scheme in SCHEMES:
            base = seg.Digests({k: zeros for k in seg.ARRAYS}, zeros, scheme).result()
            self.assertEqual(base, (ref_state({k: zeros for k in seg.ARRAYS}, scheme), ref_gradient(zeros, scheme)))
            seen.add(base)
            for a, b in seg._slice_bounds(n):
                for pos, bit in ((a, 0), (b - 1, 7)):
                    buf = bytearray(n)
                    buf[pos] = 1 << bit
                    buf = bytes(buf)
                    for key in seg.ARRAYS:
                        arrays = {k: (buf if k == key else zeros) for k in seg.ARRAYS}
                        got = seg.Digests(arrays, zeros, scheme).result()
                        self.assertEqual(got[0], seg._hash_arrays(arrays, scheme))
                        self.assertNotIn(got, seen)
                        seen.add(got)
                    got = seg.Digests({k: zeros for k in seg.ARRAYS}, buf, scheme).result()
                    self.assertEqual(got[1], seg._hash_gradient(buf, scheme))
                    self.assertNotEqual(got[1], base[1])

    def test_the_slice_bounds_are_the_schemes(self):
        for n in SIZES:
            bounds = seg._slice_bounds(n)
            self.assertEqual(len(bounds), 8)
            self.assertEqual(bounds[0][0], 0)
            self.assertEqual(bounds[-1][1], n)
            self.assertTrue(all(bounds[i][1] == bounds[i + 1][0] for i in range(7)))


# ---------------------------------------------------------------- the snapshot

class FakeBinding:
    def __init__(self, trainer):
        self.t = trainer
        self.calls = []

    def byte_lm_parallel_export(self, session, bufs, rank, gradients):
        self.calls.append((rank, gradients, [id(b) for b in bufs]))
        if gradients:
            bufs[0][:] = self.t.g.tobytes()
        else:
            for b, arr in zip(bufs, (self.t.p, self.t.m, self.t.v, self.t.flags)):
                b[:] = arr.tobytes()
        return self.t.step_ + self.t.skew


class BindingTrainer:
    """The surface `Snapshot.read` uses of the real parallel trainer."""

    def __init__(self, n=37, tensors=3):
        self._shape = types.SimpleNamespace(n_total=n, n_tensors=tensors)
        self._lock = threading.RLock()
        self._session = object()
        self._binding = FakeBinding(self)
        self.step_ = 5
        self.skew = 0
        self.opened = 0
        self.fill(0)

    def fill(self, seed):
        rng = np.random.default_rng(seed)
        n = self._shape.n_total
        self.p, self.m, self.v, self.g = (rng.standard_normal(n).astype(np.float32) for _ in range(4))
        self.flags = rng.integers(0, 2, self._shape.n_tensors).astype(np.int32)

    def _open(self):
        self.opened += 1


class SnapshotReads(unittest.TestCase):
    def setUp(self):
        buf = types.ModuleType('mojolearn._buffer')
        buf.empty = lambda shape, dtype: bytearray(4 * shape[0])
        buf.addr = lambda obj, name=None: obj
        self.saved = sys.modules.get('mojolearn._buffer')
        sys.modules['mojolearn._buffer'] = buf

    def tearDown(self):
        if self.saved is None:
            sys.modules.pop('mojolearn._buffer', None)
        else:
            sys.modules['mojolearn._buffer'] = self.saved

    def test_the_export_call_into_reused_buffers(self):
        t = BindingTrainer()
        snap = seg.Snapshot()
        snap.read(t)
        first = {k: id(v) for k, v in snap.arrays.items()}, id(snap.gradient)
        self.assertEqual(bytes(snap.arrays['parameters']), t.p.tobytes())
        self.assertEqual(bytes(snap.arrays['flags']), t.flags.tobytes())
        self.assertEqual(bytes(snap.gradient), t.g.tobytes())
        self.assertEqual([(c[0], c[1], len(c[2])) for c in t._binding.calls], [(0, False, 4), (0, True, 1)])
        t.fill(1)
        t.step_ += 1
        snap.read(t)
        self.assertEqual(({k: id(v) for k, v in snap.arrays.items()}, id(snap.gradient)), first)
        self.assertEqual(bytes(snap.arrays['v']), t.v.tobytes())
        self.assertEqual(bytes(snap.gradient), t.g.tobytes())
        raw = dict(parameters=t.p, m=t.m, v=t.v, flags=t.flags)
        self.assertEqual(seg.Digests(snap.arrays, snap.gradient, seg.SCHEME_V2).result(),
                         (seg._hash_arrays(raw, seg.SCHEME_V2), seg._hash_gradient(t.g, seg.SCHEME_V2)))

    def test_no_gradient_for_the_split_step(self):
        t = BindingTrainer()
        snap = seg.Snapshot().read(t, gradient=False)
        self.assertIsNone(snap.gradient)
        self.assertEqual(len(t._binding.calls), 1)

    def test_a_step_mismatch_is_refused(self):
        t = BindingTrainer()
        t.skew = 1
        with self.assertRaises(RuntimeError):
            seg.Snapshot().read(t)

    def test_a_stand_in_is_read_through_its_methods(self):
        class Plain:
            def export_raw(self):
                return dict(parameters=b'a', m=b'b', v=b'c', flags=b'd')

            def export_gradients(self):
                return b'e'
        snap = seg.Snapshot().read(Plain())
        self.assertEqual(snap.arrays['v'], b'c')
        self.assertEqual(snap.gradient, b'e')


# ---------------------------------------------------------------- the loop, synchronous and overlapped

N = 2048
K = 4


def _grad(ids):
    rng = np.random.default_rng(int(np.asarray(ids).ravel()[0]) + 1)
    return (rng.standard_normal(N) * np.exp2(rng.integers(-8, 8, N))).astype(np.float32)


class FakePar:
    """ParallelByteLanguageModelTrainer's surface in NumPy, with the
    metadata (`_state`) a checkpoint is written from."""
    made = []

    def __init__(self, state, *, devices=(0,), logical_shards=1, pool_optimizer=True):
        self.logical_shards = logical_shards
        self.p = np.array(state['parameters'], dtype=np.float32)
        self.m = np.array(state['m'], dtype=np.float32)
        self.v = np.array(state['v'], dtype=np.float32)
        self.flags = np.array(state['flags'], dtype=np.int32)
        self._state = dict(state, parameters=None, m=None, v=None)
        self.lr = np.float32(1e-3)
        self.total = None
        self.steps_run = 0
        FakePar.made.append(self)

    @property
    def step_(self):
        return self._state['completed_steps']

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def set_lr(self, lr):
        self.lr = np.float32(lr)
        self._state['config'] = dict(self._state['config'], lr=float(self.lr))
        return float(self.lr)

    def train_step(self, shards):
        total = _grad(shards[0]).copy()
        for x in shards[1:]:
            total = (total + _grad(x)).astype(np.float32)
        self.total = total
        self.m = (np.float32(0.9) * self.m + np.float32(0.1) * total).astype(np.float32)
        self.v = (np.float32(0.95) * self.v + np.float32(0.05) * total * total).astype(np.float32)
        self.p = (self.p - self.lr * self.m / (np.sqrt(self.v) + np.float32(1e-8))).astype(np.float32)
        self.flags = np.ones_like(self.flags)
        self._state['completed_steps'] += 1
        self._state['next_batch_index'] = self._state['completed_steps']
        self.steps_run += 1
        return dict(losses=tuple(float(np.float32(2 + int(np.asarray(x).ravel()[0]) % 7 / 8)) for x in shards))

    def export_raw(self):
        return dict(parameters=self.p.copy(), m=self.m.copy(), v=self.v.copy(), flags=self.flags.copy())

    def export_gradients(self):
        return self.total.copy()


class FakeBatches:
    sha256 = 'ab' * 32

    def ids(self, i):
        return np.full((2, 3), i, dtype=np.int32)


def fake_ck_module(record):
    """A stand-in for `mojolearn._byte_lm_checkpoint.save` that writes the
    metadata canonically and every array's bytes, so two files are equal
    exactly when the state dicts handed to it are."""
    mod = types.ModuleType('mojolearn._byte_lm_checkpoint')

    def save(path, state):
        meta = json.dumps({k: v for k, v in state.items() if k not in seg.ARRAYS}, sort_keys=True).encode()
        blob = b'%08d' % len(meta) + meta + b''.join(
            np.ascontiguousarray(state[k]).tobytes() for k in seg.ARRAYS)
        Path(path).write_bytes(blob)
        record.append(Path(path).name)
        return hashlib.sha256(blob).hexdigest()
    mod.save = save
    return mod


TIMING = ('seconds', 'hash_seconds', 'prev')


def _strip(lines):
    return [{k: v for k, v in l.items() if k not in TIMING} for l in lines]


class Loop(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.recipe = self.dir / 'recipe.json'
        table = ['%08x' % seg._f32_bits(1e-3 * (t + 1)) for t in range(20)]
        self.recipe.write_text(json.dumps(dict(schema=seg.RECIPE_SCHEMA, shape=[1] * 9, logical_shards=K, steps=20,
                                               seed=1, schedule=dict(table_f32_hex=table), checkpoint_every=2,
                                               boundaries=[], hash_scheme=seg.SCHEME_V2)))
        rng = np.random.default_rng(7)
        self.state = dict(parameters=rng.standard_normal(N).astype(np.float32),
                          m=(rng.standard_normal(N) * 1e-3).astype(np.float32),
                          v=(rng.random(N) * 1e-6).astype(np.float32),
                          flags=np.zeros(3, dtype=np.int32), completed_steps=3, next_batch_index=3,
                          config=dict(lr=1e-3), data_schedule={'x': 1})
        self.saved = (seg.open_batches, seg.data_schedule, seg.load_checkpoint)
        seg.open_batches = lambda recipe, tokens: FakeBatches()
        seg.data_schedule = lambda recipe, batches: {'x': 1}
        seg.load_checkpoint = lambda path, zero_moments=False: (
            {k: (v.copy() if hasattr(v, 'copy') else v) for k, v in self.state.items()}, 'cd' * 32)
        self.saved_mods = {n: sys.modules.get(n) for n in ('mojolearn.parallel_training',
                                                            'mojolearn._byte_lm_checkpoint')}
        fake = types.ModuleType('mojolearn.parallel_training')
        fake.ParallelByteLanguageModelTrainer = FakePar
        sys.modules['mojolearn.parallel_training'] = fake
        self.saves = []
        sys.modules['mojolearn._byte_lm_checkpoint'] = fake_ck_module(self.saves)
        # `from mojolearn import X` reads the package's attribute first: a
        # package imported by an earlier test carries the real submodules
        self.pkg = sys.modules.get('mojolearn')
        self.saved_attrs = {n: getattr(self.pkg, n, None) for n in ('parallel_training', '_byte_lm_checkpoint')}
        if self.pkg is not None:
            self.pkg.parallel_training = fake
            self.pkg._byte_lm_checkpoint = sys.modules['mojolearn._byte_lm_checkpoint']
        FakePar.made = []

    def tearDown(self):
        seg.open_batches, seg.data_schedule, seg.load_checkpoint = self.saved
        if self.pkg is not None:
            for n, mod in self.saved_attrs.items():
                if mod is None:
                    self.pkg.__dict__.pop(n, None)
                else:
                    setattr(self.pkg, n, mod)
        for n, mod in self.saved_mods.items():
            if mod is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = mod
        self.tmp.cleanup()

    def run_seg(self, name, *extra, steps=4, expect=None, checkpoints=False):
        out = self.dir / name
        argv = ['run', '--recipe', str(self.recipe), '--tokens', str(self.dir), '--from', str(self.dir / 'c.blm'),
                '--steps', str(steps), '--out', str(out), *extra]
        if not checkpoints:
            argv.append('--no-checkpoints')
        if expect:
            argv += ['--expect-chain', str(expect)]
        rc = seg.main(argv)
        lines = [json.loads(l) for l in (out / 'chain.jsonl').read_text().splitlines()]
        return rc, lines, json.loads((out / 'segment.json').read_text()), out

    def test_the_same_lines_in_the_same_order(self):
        rc_s, sync, seg_s, out_s = self.run_seg('sync', '--sync-hash', steps=5)
        rc_o, over, seg_o, out_o = self.run_seg('over', steps=5)
        self.assertEqual((rc_s, rc_o), (0, 0))
        self.assertEqual([l['step'] for l in over], [4, 5, 6, 7, 8])
        self.assertEqual(_strip(over), _strip(sync))
        self.assertEqual([sorted(l) for l in over], [sorted(l) for l in sync])  # no field added or lost
        # the chain links each line to the one before it
        text = (out_o / 'chain.jsonl').read_text().splitlines()
        for a, b in zip(text, text[1:]):
            self.assertEqual(json.loads(b)['prev'], hashlib.sha256(a.encode()).hexdigest())
        self.assertEqual((seg_s['digests'], seg_o['digests']), ('synchronous', 'overlapped'))
        self.assertEqual([t['step'] for t in seg_o['digest_timings']], [4, 5, 6, 7, 8])
        # and both equal what the definition says of the stand-in's states
        self.assertEqual(over[-1]['state_sha256'], seg._hash_arrays(FakePar.made[-1].export_raw(), seg.SCHEME_V2))

    def test_an_expected_chain_passes_both_ways(self):
        self.run_seg('ref', '--sync-hash')
        ref = self.dir / 'ref' / 'chain.jsonl'
        for name, extra in (('s', ('--sync-hash',)), ('o', ())):
            rc, lines, segment, _ = self.run_seg(name, *extra, expect=ref)
            self.assertEqual((rc, segment['verdict'], len(lines)), (0, 'PASS', 4), name)

    def _inject(self, at, field='state_sha256'):
        self.run_seg('ref%d' % at, '--sync-hash', steps=6)
        ref = self.dir / ('ref%d' % at) / 'chain.jsonl'
        rows = [json.loads(l) for l in ref.read_text().splitlines()]
        for r in rows:
            if r['step'] == at:
                r[field] = ('0' if r[field][0] != '0' else '1') + r[field][1:]
        bad = self.dir / ('bad-%d.jsonl' % at)
        bad.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        return bad

    def test_a_disagreement_stops_at_the_step_it_is_at(self):
        for at in (4, 6, 9):  # the first step, a middle step, the last step
            bad = self._inject(at)
            rc_s, sync, seg_s, _ = self.run_seg('s%d' % at, '--sync-hash', steps=6, expect=bad)
            computed_sync = FakePar.made[-1].steps_run
            rc_o, over, seg_o, _ = self.run_seg('o%d' % at, steps=6, expect=bad)
            computed_over = FakePar.made[-1].steps_run
            self.assertEqual((rc_s, rc_o), (1, 1), at)
            self.assertEqual([l['step'] for l in over], list(range(4, at + 1)), at)
            self.assertEqual(_strip(over), _strip(sync), at)
            for s in (seg_s, seg_o):
                self.assertEqual(s['verdict'], 'FAIL')
                self.assertEqual(s['disagreements'][0]['step'], at)
                self.assertEqual(s['disagreements'][0]['fields'], ['state_sha256'])
                self.assertEqual((s['last_completed'], s['steps_completed']), (at, at - 3))
            self.assertEqual(len(seg_o['disagreements']), 1)
            # the overlapped loop learns of it one step late at most, and records nothing after it
            self.assertEqual(computed_sync, at - 3)
            self.assertEqual(computed_over, min(at - 3 + 1, 6), at)

    def test_checkpoints_are_the_same_state(self):
        rc_s, _, seg_s, out_s = self.run_seg('cs', '--sync-hash', steps=5, checkpoints=True)
        saved_sync = list(self.saves)
        rc_o, _, seg_o, out_o = self.run_seg('co', steps=5, checkpoints=True)
        self.assertEqual((rc_s, rc_o), (0, 0))
        names = ['ckpt_00000004.blm', 'ckpt_00000006.blm', 'ckpt_00000008.blm']
        self.assertEqual(saved_sync, names)
        self.assertEqual(self.saves[len(saved_sync):], names)
        for name in names:
            self.assertEqual((out_s / name).read_bytes(), (out_o / name).read_bytes(), name)
        strip = lambda cks: [{k: v for k, v in c.items() if k != 'save_seconds'} for c in cks]  # noqa: E731
        self.assertEqual(strip(seg_o['checkpoints']), strip(seg_s['checkpoints']))
        self.assertEqual((out_s / 'manifest.tsv').read_text().splitlines()[1:],
                         (out_o / 'manifest.tsv').read_text().splitlines()[1:])  # all but chain.jsonl's row
        blob = (out_o / names[1]).read_bytes()
        meta = json.loads(blob[8:8 + int(blob[:8])])
        self.assertEqual((meta['completed_steps'], meta['next_batch_index']), (6, 6))
        self.assertEqual(meta['config']['lr'], float(np.float32(seg._bits_f32(int(json.loads(
            self.recipe.read_text())['schedule']['table_f32_hex'][5], 16)))))

    def test_no_checkpoint_after_a_disagreement(self):
        bad = self._inject(6)
        self.saves.clear()
        self.run_seg('cso', steps=5, expect=bad, checkpoints=True)
        self.assertEqual(self.saves, ['ckpt_00000004.blm'])
        self.saves.clear()
        self.run_seg('css', '--sync-hash', steps=5, expect=bad, checkpoints=True)
        self.assertEqual(self.saves, ['ckpt_00000004.blm'])


class Pipeline(unittest.TestCase):
    def test_a_job_error_reaches_the_main_loop(self):
        class Boom:
            def write(self, row):
                raise OSError('disk full')
        pipe = seg.StepPipeline(Boom(), seg.SCHEME_V2, lambda *_: None, '.')
        pipe.submit(dict(step=1, seconds=0.0), {k: b'x' for k in seg.ARRAYS}, b'y', t_readback=0.0,
                    readback_seconds=0.0)
        with self.assertRaises(OSError):
            pipe.after_step()
        pipe.close()

    def test_an_upload_error_reaches_the_main_loop(self):
        class Chain:
            def write(self, row):
                return True
        pipe = seg.StepPipeline(Chain(), seg.SCHEME_V2, lambda *_: None, '.',
                                save=lambda meta, arrays, step, out, manifest, upload: (
                                    upload('f', 'p'), dict(step=step, file='f', bytes=0, sha256='0', save_seconds=0))[1])

        def upload(name, path):
            raise SystemExit('upload of f failed three times')
        pipe.submit(dict(step=1, seconds=0.0), {k: b'x' for k in seg.ARRAYS}, b'y', t_readback=0.0,
                    readback_seconds=0.0, checkpoint_meta={}, upload=upload)
        with self.assertRaises(SystemExit):
            pipe.close()

    def test_one_job_at_a_time(self):
        class Chain:
            def write(self, row):
                return True
        pipe = seg.StepPipeline(Chain(), seg.SCHEME_V2, lambda *_: None, '.')
        args = (dict(step=1, seconds=0.0), {k: b'x' for k in seg.ARRAYS}, b'y')
        pipe.submit(*args, t_readback=0.0, readback_seconds=0.0)
        with self.assertRaises(RuntimeError):
            pipe.submit(*args, t_readback=0.0, readback_seconds=0.0)
        pipe.close()


if __name__ == '__main__':
    unittest.main()
