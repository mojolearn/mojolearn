"""tools/lm_cpu_witness.py without a binding: the schedule (which ids a shard
reads), the byte ranges fetched from the stream's parts, the manifest pins,
the fold, the hashing reuse and the verdicts. CPU only, no model data."""
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np

sys.argv = [sys.argv[0]]
TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location('lm_cpu_witness', TOOLS / 'lm_cpu_witness.py')
w = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(w)
_CV = importlib.util.spec_from_file_location(
    'cross_vendor_for_test', TOOLS.parent / 'python/mojolearn/cross_vendor.py')
cross_vendor = importlib.util.module_from_spec(_CV)
_CV.loader.exec_module(cross_vendor)  # the fold needs only the standard library


def _package():
    """The installed or built package, or None on a checkout with no bindings."""
    try:
        from mojolearn import lm_corpus
        return lm_corpus
    except ImportError:
        return None


class _NoArrayPackage:
    """`lm_segment._bytes_of` asks `mojolearn._array.Array` whether a value is
    the package's own array; on a checkout with no bindings built the package
    cannot import, and a NumPy array never is one, so a stand-in answers."""

    def __enter__(self):
        import types
        self.saved = {k: sys.modules.get(k) for k in ('mojolearn', 'mojolearn._array')}
        if _package() is None:
            pkg, arr = types.ModuleType('mojolearn'), types.ModuleType('mojolearn._array')
            arr.Array = type('Array', (), {})
            pkg._array = arr
            sys.modules['mojolearn'], sys.modules['mojolearn._array'] = pkg, arr
        return self

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def _recipe(manifest_bytes, manifest, shape=(4, 8, 16, 2, 2, 8, 32, 2, 64), K=4):
    return dict(schema=w.seg.RECIPE_SCHEMA, shape=list(shape), logical_shards=K, steps=3, seed=1,
                optimizer=dict(betas=[0.9, 0.95], eps=1e-8, weight_decay=0.1),
                schedule=dict(table_f32_hex=['3a83126f'] * 3),
                data=dict(schedule=w.seg.SCHEDULE, tokens_sha256=manifest['sha256'],
                          tokens_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                          train_range=manifest['train_range']),
                hash_scheme=w.seg.SCHEME_V2)


def _stream(tmp, n_tokens=4000, train_hi=3000, vocab=64):
    ids = (np.arange(n_tokens, dtype=np.int64) * 7919 % vocab).astype('<i4')
    raw = ids.tobytes()
    (tmp / 'tokens.i32').write_bytes(raw)
    manifest = dict(schema='mojolearn.byte-lm.tokens.v1', sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw),
                    tokens=n_tokens, train_range=[0, train_hi], validation_range=[train_hi, n_tokens],
                    source=dict(sha256='00' * 32),
                    vocabulary=dict(schema='mojolearn.bpe-vocabulary.v1', sha256='11' * 32, n_vocab=vocab))
    mb = json.dumps(manifest).encode()
    (tmp / 'manifest.json').write_bytes(mb)
    return ids, manifest, mb


class Schedule(unittest.TestCase):
    def test_rows_are_token_batches_rows(self):
        """row_starts is TokenBatches.ids: row b of index i starts at
        lo + (i*B*L + b*L) % (hi - lo - L - 1)."""
        lm_corpus = _package()  # asked when the test runs, not at collection
        if lm_corpus is None:
            self.skipTest('mojolearn does not import here (no bindings built)')
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            ids, manifest, _ = _stream(tmp)
            tb = lm_corpus.TokenBatches(tmp, 4, 8)
            for index in (0, 1, 5, 91, 400):
                want = bytes(memoryview(tb.ids(index)._mv).cast('B'))
                starts = w.row_starts(index, 4, 8, 0, 3000)
                got = b''.join(ids[s:s + 9].tobytes() for s in starts)
                self.assertEqual(got, want, index)

    def test_step_and_shard_map_to_the_chain_batch_index(self):
        # chain line 101 of the T3 run: batch_index [6400, 6464)
        self.assertEqual(w.train_shard_index(101, 0, 64), 6400)
        self.assertEqual(w.train_shard_index(101, 63, 64), 6463)
        self.assertEqual(w.train_shard_index(1, 0, 64), 0)
        with self.assertRaises(ValueError):
            w.train_shard_index(0, 0, 64)
        with self.assertRaises(ValueError):
            w.train_shard_index(5, 64, 64)

    def test_the_t3_shard_and_heldout_rows(self):
        """The real stream's numbers (recipe train range, manifest validation
        range, B=4, L=2048): step 101 shard 0 starts at token 52,428,800 and
        held-out batch 0 at the first id of shard 013."""
        self.assertEqual(w.row_starts(6400, 4, 2048, 0, 2926502182),
                         [52428800, 52430848, 52432896, 52434944])
        self.assertEqual(w.row_starts(0, 4, 2048, 2926502182, 3110556447),
                         [2926502182, 2926504230, 2926506278, 2926508326])
        self.assertEqual(w.byte_segments(2926502182, 2049), [(5, 1706008728, 1706016924)])

    def test_heldout_uses_the_validation_range(self):
        with tempfile.TemporaryDirectory() as d:
            _, manifest, mb = _stream(Path(d))
            recipe = _recipe(mb, manifest)
            self.assertEqual(w.heldout_rows(recipe, manifest, 0), [3000, 3008, 3016, 3024])
            del manifest['validation_range']
            with self.assertRaises(SystemExit):
                w.heldout_rows(recipe, manifest, 0)


class Ranges(unittest.TestCase):
    def test_a_row_inside_one_part(self):
        self.assertEqual(w.byte_segments(10, 5, part_bytes=400), [(0, 40, 60)])

    def test_a_row_across_a_part_boundary(self):
        # tokens 95..105 with 100 tokens a part: bytes [380, 400) of part 0 and [0, 24) of part 1
        self.assertEqual(w.byte_segments(95, 11, part_bytes=400), [(0, 380, 400), (1, 0, 24)])

    def test_segments_cover_exactly_the_bytes(self):
        for first, count in ((0, 1), (99, 2), (250, 400), (7, 2049)):
            segs = w.byte_segments(first, count, part_bytes=400)
            self.assertEqual(sum(b - a for _, a, b in segs), 4 * count)
            self.assertEqual(segs[0][0] * 400 + segs[0][1], 4 * first)

    def test_local_parts_read_the_same_ids_as_the_whole_file(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            ids, _, _ = _stream(tmp)
            parts = tmp / 'parts'
            parts.mkdir()
            raw = (tmp / 'tokens.i32').read_bytes()
            for i in range(0, len(raw), 1000):
                (parts / ('tokens.i32.part%02d' % (i // 1000))).write_bytes(raw[i:i + 1000])
            old = w.PART_BYTES
            w.PART_BYTES = 1000
            try:
                starts = [240, 1500, 3000]
                whole = w.Tokens(str(tmp)).rows(starts, 8)
                split = w.Tokens(str(parts)).rows(starts, 8)
            finally:
                w.PART_BYTES = old
            self.assertEqual(whole, split)
            self.assertEqual(whole, b''.join(ids[s:s + 9].tobytes() for s in starts))


class Pins(unittest.TestCase):
    def test_manifest_must_be_the_recipes(self):
        with tempfile.TemporaryDirectory() as d:
            _, manifest, mb = _stream(Path(d))
            recipe = _recipe(mb, manifest)
            self.assertEqual(w.check_manifest(recipe, mb)['sha256'], manifest['sha256'])
            with self.assertRaises(SystemExit):
                w.check_manifest(recipe, mb + b' ')
            other = dict(recipe, data=dict(recipe['data'], train_range=[0, 2999]))
            with self.assertRaises(SystemExit):
                w.check_manifest(other, mb)

    def test_the_repo_manifest_is_the_t3_recipes(self):
        """bench/results/fineweb_tokens_2026-09-22/run3/manifest.json is the
        stream the T3 recipe pins (tokens_manifest_sha256 1db7fadd...)."""
        path = TOOLS.parent / 'bench/results/fineweb_tokens_2026-09-22/run3/manifest.json'
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                         '1db7fadd6dec742d5dbf6c8ec3d8d95cb3aaf809fb4a8080038598240ff81b71')
        m = json.loads(path.read_text())
        self.assertEqual(m['validation_range'], [2926502182, 3110556447])


class Fold(unittest.TestCase):
    def _gradients(self):
        rng = np.random.default_rng(3)
        gs = [rng.normal(0, 1, 257).astype(np.float32) for _ in range(5)]
        tiny = struct.unpack('<f', struct.pack('<I', 0x00000003))[0]  # a subnormal
        gs[1][:4] = [tiny, -tiny, 1e-39, 3.0]
        gs[2][:4] = [-tiny, tiny, -1e-39, -3.0]
        gs[0][5] = 1e-38
        gs[3][5] = -1.00000001e-38
        return gs

    def test_equals_the_coordinators_fold(self):
        """cross_vendor.ordered_fold (the host emulation the live segment
        uses) and this numpy fold give the same bytes, subnormals included."""
        ordered_fold = cross_vendor.ordered_fold
        gs = self._gradients()
        total = None
        for g in gs:
            total = w.ordered_fold_step(total, g, np)
        self.assertEqual(total.tobytes(), ordered_fold([g.tobytes() for g in gs]))

    def test_the_first_gradient_is_copied_unflushed(self):
        g = np.array([struct.unpack('<f', struct.pack('<I', 1))[0]], dtype=np.float32)
        self.assertEqual(w.ordered_fold_step(None, g, np).view(np.uint32)[0], 1)

    def test_order_matters(self):
        gs = [np.array([1e8, 1.0, -1e8], dtype=np.float32)[i:i + 1] for i in range(3)]
        a = b = None
        for g in gs:
            a = w.ordered_fold_step(a, g, np)
        for g in (gs[0], gs[2], gs[1]):
            b = w.ordered_fold_step(b, g, np)
        self.assertNotEqual(a.tobytes(), b.tobytes())


class Hashing(unittest.TestCase):
    def test_gradient_hash_is_lm_segments(self):
        g = np.arange(1000, dtype=np.float32)
        raw = g.tobytes()
        n = len(raw)
        parts = [hashlib.sha256(raw[n * i // 8:n * (i + 1) // 8]).hexdigest() for i in range(8)]
        want = hashlib.sha256(''.join(parts).encode()).hexdigest()
        with _NoArrayPackage():
            self.assertEqual(w.seg._hash_gradient(g, w.seg.SCHEME_V2), want)
            self.assertEqual(w.seg._hash_gradient(g, w.seg.SCHEME_V1), hashlib.sha256(raw).hexdigest())

    def test_chain_reader_and_hex(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'chain.jsonl'
            line = dict(schema=w.seg.CHAIN_SCHEMA, step=101, losses_f32_hex=['40de1f6a'], hash_scheme=w.seg.SCHEME_V2,
                        state_sha256='ab' * 32, gradient_sha256='cd' * 32, lr_f32_hex='397e2cc1')
            p.write_text(json.dumps(line) + '\n')
            chain = w.load_chain(p)
            self.assertEqual(chain[101]['losses_f32_hex'][0], '40de1f6a')
        self.assertEqual(w.f32_hex(w.seg._bits_f32(0x40de1f6a)), '40de1f6a')


class FoldCommand(unittest.TestCase):
    """`fold` over saved shard gradients: whole, chained through a prefix,
    and a swapped pair that must FAIL."""

    def _setup(self, tmp):
        _, manifest, mb = _stream(tmp)
        recipe = _recipe(mb, manifest)
        (tmp / 'recipe.json').write_text(json.dumps(recipe))
        rng = np.random.default_rng(7)
        grads = tmp / 'grads'
        grads.mkdir()
        gs = [rng.normal(0, 1, 999).astype(np.float32) for _ in range(4)]
        for k, g in enumerate(gs):
            g.tofile(str(grads / ('grad_%02d.f32' % k)))
        total = None
        for g in gs:
            total = w.ordered_fold_step(total, g, np)
        with _NoArrayPackage():
            digest = w.seg._hash_gradient(total, w.seg.SCHEME_V2)
        line = dict(schema=w.seg.CHAIN_SCHEMA, step=2, losses_f32_hex=['0'] * 4, hash_scheme=w.seg.SCHEME_V2,
                    state_sha256='ab' * 32, gradient_sha256=digest, lr_f32_hex='3a83126f')
        (tmp / 'chain.jsonl').write_text(json.dumps(line) + '\n')
        return grads, gs

    def _fold(self, tmp, *extra):
        base = ['fold', '--recipe', str(tmp / 'recipe.json'), '--chain', str(tmp / 'chain.jsonl'), '--step', '2',
                '--grads', str(tmp / 'grads'), '--out', str(tmp / 'fold.json')]
        with _NoArrayPackage():
            rc = w.main(base + list(extra))
        return rc, json.loads((tmp / 'fold.json').read_text())

    def test_whole_and_chained_folds_pass(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            self._setup(tmp)
            rc, rec = self._fold(tmp, '--shards', '0:4')
            self.assertEqual((rc, rec['verdict']), (0, 'PASS'))
            rc, rec = self._fold(tmp, '--shards', '0:2', '--save-prefix', str(tmp / 'prefix.f32'))
            self.assertTrue(rec['verdict'].startswith('PREFIX'))
            rc, rec = self._fold(tmp, '--shards', '2:4', '--prefix', str(tmp / 'prefix.f32'))
            self.assertEqual((rc, rec['verdict']), (0, 'PASS'))

    def test_two_shards_swapped_fail(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            grads, gs = self._setup(tmp)
            gs[2].tofile(str(grads / 'grad_01.f32'))
            gs[1].tofile(str(grads / 'grad_02.f32'))
            rc, rec = self._fold(tmp, '--shards', '0:4')
            self.assertEqual((rc, rec['verdict']), (1, 'FAIL'))

    def test_a_fold_from_the_middle_needs_its_prefix(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            self._setup(tmp)
            with self.assertRaises(SystemExit):
                self._fold(tmp, '--shards', '2:4')


class Verdicts(unittest.TestCase):
    def test_bits_decide(self):
        self.assertEqual(w.verdict_of('40de1f6a', '40de1f6a', False), 'PASS')
        self.assertEqual(w.verdict_of('40de1f6b', '40de1f6a', False), 'FAIL')

    def test_a_control_must_fail(self):
        self.assertTrue(w.verdict_of('40de1f6b', '40de1f6a', True).startswith('FAIL (EXPECTED'))
        self.assertIn('BLIND', w.verdict_of('40de1f6a', '40de1f6a', True).upper())


if __name__ == '__main__':
    unittest.main()
