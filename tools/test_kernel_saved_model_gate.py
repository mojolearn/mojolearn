"""Kernel saved-model probes must preserve the harness input transformation."""
import unittest

import numpy as np
import classical_host_gate as gate


class KernelProbeTests(unittest.TestCase):
    def test_all_variants_use_scaled_contiguous_float32_rows(self):
        # A strided input catches accidental shape-only/contiguity assumptions.
        source = np.arange(160 * 12, dtype=np.float32).reshape(160, 12)[:, ::2]
        expected = np.ascontiguousarray(source[:64, :4] * np.float32(0.125))
        class Recorder:
            def predict(self, x):
                self.seen = x
                return x[:, 0]
            transform = predict
        for family, estimator, method in (("kernel-ridge", "KernelRidge", "predict"),
                                           ("nystroem", "Nystroem", "transform")):
            for kernel in ("poly", "sigmoid", "laplacian"):
                with self.subTest(family=family, kernel=kernel):
                    lane = f"{family}-{kernel}"
                    name, probe, extras = gate.LANES[lane]
                    model = Recorder()
                    result = probe(model, source)
                    self.assertEqual(name, estimator)
                    self.assertEqual(gate.PROBE_NAMES[lane], method)
                    self.assertEqual(extras, {})
                    self.assertEqual(model.seen.dtype, np.dtype("float32"))
                    self.assertTrue(model.seen.flags.c_contiguous)
                    self.assertEqual(model.seen.tobytes(), expected.tobytes())
                    self.assertEqual(len(result), 1)
                    np.testing.assert_array_equal(result[0], expected[:, 0])

    def test_original_rbf_probe_is_not_scaled(self):
        source = np.ones((256, 16), dtype=np.float32)
        class Model:
            def predict(self, x):
                return x
            transform = predict
        for lane in ("kernel-ridge", "nystroem"):
            np.testing.assert_array_equal(gate.LANES[lane][1](Model(), source)[0], source[:64, :4])


if __name__ == "__main__":
    unittest.main()
