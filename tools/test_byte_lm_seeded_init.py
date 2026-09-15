"""The seeded initializer, as a test that runs rather than a script nobody calls.

`tools/byte_lm_seeded_init_gate.py` needs a retained capture tree to compare
against, and CI sparse-checkout fetches at most one of the three. These tests
need NONE of them: they pin the arithmetic itself, which is the part that must
not drift. The tree comparison is the workflow's job and is a separate step.

WHAT IS PINNED HERE, and why each one would otherwise rot silently:

  * the recorded digest. `initial_parameters_sha256` is `b87a6075...` in all
    three retained trees. Regenerating the vector and hashing it is the whole
    claim in one line, and it needs no capture on disk.
  * the identifier string, because the captures record it and a generator that
    changed while the string did not would be undetectable from the trees.
  * the grid. 257 distinct values on a dyadic step of 2^-10. A draw that moved
    onto a finer grid would still look plausible in a histogram.
  * the norm vectors. 128 exact ones, which a draw-only implementation would miss in
    precisely four tensors.
  * that a WRONG seed does not reproduce the digest, because a check that
    passes under any input is not reading what it claims to.
"""
import hashlib
import importlib.util
import struct
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tools' / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


seeded = _load('_seeded_init', 'byte_lm_seeded_init_gate.py')
shapes = _load('_byte_lm_shape', 'byte_lm_shape.py')

#: What all three retained trees record for the b2-l32 shape.
RECORDED_SHA = 'b87a6075597a82c94b68b810bef74f6331b86579d4d434d95455866ada3ba424'
RECORDED_ID = 'u32-avalanche-index-xor-42595445-top8-centered128-div1024-norm1.v1'


class SeededInitTests(unittest.TestCase):
    def setUp(self):
        self.shape = shapes.Shape()

    def test_regenerates_the_recorded_digest(self):
        raw = seeded.regenerate(self.shape)
        self.assertEqual(len(raw), self.shape.n_total * 4)
        self.assertEqual(
            hashlib.sha256(raw).hexdigest(), RECORDED_SHA,
            'the seeded initializer no longer reproduces the initialization the '
            'three retained vendor captures were taken from, so every CPU and '
            'cross-vendor training result that starts from it is no longer '
            'reachable from a seed')

    def test_identifier_matches_what_the_captures_record(self):
        self.assertEqual(seeded.INIT_ID, RECORDED_ID)

    def test_the_grid_is_257_dyadic_values(self):
        raw = seeded.regenerate(self.shape)
        values = struct.unpack('<%df' % self.shape.n_total, raw)
        distinct = sorted(set(values))
        self.assertEqual(len(distinct), 257, 'the draw left its 257-point grid')
        self.assertEqual(min(distinct), -0.125)
        self.assertEqual(max(distinct), 1.0)
        # Every drawn value is an exact multiple of 2^-10.
        for value in distinct:
            if value == 1.0:
                continue
            self.assertEqual(value * 1024.0, round(value * 1024.0),
                             f'{value!r} is not on the 2^-10 grid')

    def test_norm_vectors_are_exactly_one(self):
        raw = seeded.regenerate(self.shape)
        values = struct.unpack('<%df' % self.shape.n_total, raw)
        spans = [(e['offset'], e['count']) for e in self.shape.registry()
                 if e['name'].endswith(('norm1_w', 'norm2_w'))]
        self.assertEqual(len(spans), 4)
        for offset, count in spans:
            for index in range(offset, offset + count):
                self.assertEqual(values[index], 1.0)
        self.assertEqual(sum(1 for v in values if v == 1.0),
                         sum(c for _, c in spans))

    def test_a_wrong_seed_does_not_reproduce_it(self):
        """The control. A generator that matched under any constant would mean
        this suite is comparing something other than the draw."""
        raw = seeded.regenerate(self.shape, seed_xor=seeded.SEED_XOR + 1)
        self.assertNotEqual(hashlib.sha256(raw).hexdigest(), RECORDED_SHA)

    def test_draw_is_exact_integer_arithmetic(self):
        """Each drawn value is an 8-bit integer over 2^-10, so no rounding can
        differ between one machine and another."""
        for index in (0, 1, 17, 1023, 34943):
            top = (seeded.fmix32((index + 1) ^ seeded.SEED_XOR) >> 24)
            self.assertTrue(0 <= top <= 255)
            self.assertEqual(seeded.draw(index), (top - 128) / 1024.0)


if __name__ == '__main__':
    unittest.main()
