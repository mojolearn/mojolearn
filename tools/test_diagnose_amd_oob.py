"""CPU-only diagnostic/reference tests; these do not qualify any GPU."""
import ctypes
import importlib.util
import math
from pathlib import Path
import unittest
import numpy as np

SPEC = importlib.util.spec_from_file_location('diag', Path(__file__).with_name('diagnose_amd_oob.py'))
diag = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(diag)


class Reference(unittest.TestCase):
    def test_uncovered_rows_are_zero_not_nan(self):
        pred, words = diag.reference([0, 4, -6], [0, 2, 3], [1, 2, -2])
        self.assertEqual(pred, [0, 2, -2])
        self.assertEqual(words[0], 1)
        self.assertEqual(words[2], 1)
        self.assertTrue(all(math.isfinite(v) for v in words))

    def test_exact_sum_cancellation(self):
        _, words = diag.reference([0, 0, 0], [0, 0, 0], [2.0**53, 1.0, -(2.0**53)])
        self.assertEqual(words[0], 1)
        self.assertEqual(words[3], 1 / 3)

    def test_zero_target(self):
        self.assertEqual(diag.reference([0] * 65, [0] * 65, [0] * 65), ([0] * 65, [0] * 4))


class FakeBinding:
    """Exercise address transport using host arrays, never a real backend."""
    def x_trees_oob_r2(self, addresses, params):
        n = params[0]
        def view(addr, ctype, size):
            return np.ctypeslib.as_array((ctype * size).from_address(addr))
        acc, counts, y = [view(addresses[i], typ, n) for i, typ in enumerate([ctypes.c_double, ctypes.c_int32, ctypes.c_float])]
        p, w = diag.reference(acc, counts, y)
        view(addresses[3], ctypes.c_double, n)[:] = p
        view(addresses[4], ctypes.c_double, 4)[:] = w
        view(addresses[5], ctypes.c_int32, 4)[:] = 0


class Capture(unittest.TestCase):
    def test_fake_binding_complete_writes(self):
        arrays, row = diag.direct(FakeBinding(), [0, 2, -6], [0, 1, 3], [1, 2, -2])
        self.assertEqual(row['flags'], [0] * 4)
        self.assertTrue(row['pred_bits_match'] and row['words_bits_match'])
        self.assertTrue(row['inputs_unchanged'])
        self.assertEqual(arrays['counts'].dtype, np.dtype('<i4'))
        self.assertEqual(row['arrays']['counts']['zero_count'], 1)

    def test_missing_writes_remain_detectable(self):
        class NoWrite:
            def x_trees_oob_r2(self, addresses, params): pass
        arrays, row = diag.direct(NoWrite(), [0], [0], [1])
        self.assertEqual(row['flags'], [0x13579BDF] * 4)
        self.assertFalse(row['pred_bits_match']); self.assertFalse(row['words_bits_match'])
        self.assertEqual(row['words_bits'], ['0x7ff8abcdef123456'] * 4)

    def test_binding_error_retains_all_sentinels(self):
        class Broken:
            def x_trees_oob_r2(self, addresses, params): raise ValueError('fixture error')
        arrays, row = diag.direct(Broken(), [0], [0], [1])
        self.assertIn('fixture error', row['binding_error'])
        self.assertEqual(row['nonfinite_flag'], 0x13579BDF)
        self.assertEqual(row['overflow_flag'], 0x13579BDF)

    def test_raw_nan_payload_and_signed_zero_retained(self):
        a = np.array([0x8000000000000000, 0x7FF8000000000042], dtype='<u8').view('<f8')
        row = diag.describe(a)
        self.assertEqual(row['sample_bits'], ['0x8000000000000000', '0x7ff8000000000042'])
        self.assertEqual(row['nonfinite_indices'], [1])
        self.assertEqual(row['sha256'], diag.digest(a.tobytes()))


if __name__ == '__main__': unittest.main()
