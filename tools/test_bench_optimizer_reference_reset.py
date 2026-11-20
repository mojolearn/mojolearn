"""CPU quality references must preserve the optimizer's immutable reset input."""
import unittest
from unittest.mock import patch

import bench_board_algos as bench


class OptimizerReferenceReset(unittest.TestCase):
    def test_cpu_reference_repeats_from_original_parameter(self):
        import numpy as np
        import torch

        for lane in ("adagrad", "adam"):
            with self.subTest(lane=lane):
                inputs = bench._make_optim_inputs(64, 3)
                initial = inputs[0].copy()
                with patch.object(bench, "_optim_size", return_value=(64, 3)), \
                     patch.object(bench, "_optim_inputs", return_value=inputs), \
                     patch.object(bench, "_torch_device", return_value="cpu"):
                    runner = bench._build_optim(lane, "torch-eager-fp32", {})
                    runner.fit()
                    first = runner.outputs()["param"].copy()
                    runner.fit()
                    second = runner.outputs()["param"].copy()
                self.assertFalse(np.array_equal(first, initial))
                np.testing.assert_array_equal(inputs[0], initial)
                np.testing.assert_array_equal(second, first)


if __name__ == "__main__":
    unittest.main()
