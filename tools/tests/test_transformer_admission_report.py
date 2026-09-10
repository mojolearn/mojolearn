"""CPU-only checks of diagnostic reporting, not numerical GPU qualification."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import unittest
import numpy as np

path = Path(__file__).resolve().parents[1] / 'transformer_admission_diagnose.py'
spec = importlib.util.spec_from_file_location('admission_report_test_target', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class AdmissionReport(unittest.TestCase):
    def test_tolerance_and_position_error_are_independent(self):
        ref = np.zeros((1, 2, 2), np.float32)
        ours = ref.copy()
        ours[0, 1, 1] = .01
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            module.report('witness', ours, ref)
        row = json.loads(output.getvalue())
        self.assertFalse(row['passed'])
        self.assertEqual(row['outside'], 1)
        self.assertEqual(row['total'], 4)
        self.assertEqual(row['max_index'], [0, 1, 1])
        self.assertEqual(row['position_max_abs']['0'], 0)
        self.assertAlmostEqual(row['max_tolerance_multiple'], 1000, places=3)
        self.assertEqual((row['rtol'], row['atol']), (.0005, .00001))

    def test_nonfinite_cannot_look_like_zero_outside(self):
        ref = np.zeros((1, 2, 2), np.float32)
        for value in (np.nan, np.inf, -np.inf):
            ours = ref.copy()
            ours[0, 0, 0] = value
            with self.assertRaisesRegex(ValueError, 'nonfinite'):
                module.report('bad', ours, ref)
            with self.assertRaisesRegex(ValueError, 'nonfinite'):
                module.report('bad_reference', ref, ours)

    def test_shape_mismatch_cannot_broadcast(self):
        with self.assertRaisesRegex(ValueError, 'equal nonempty'):
            module.report('bad', np.zeros((1, 2, 2)), np.zeros((1, 1, 2)))


if __name__ == '__main__':
    unittest.main()
