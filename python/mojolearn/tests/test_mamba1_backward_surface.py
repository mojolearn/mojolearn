# SPDX-License-Identifier: Apache-2.0
"""Mamba1 arbitrary-cotangent API checks against independent float64 autograd.

Root operator only: build IDENTICAL, then execute this module serially with
BLAS/OpenMP threads set to one. Native dump certificates remain separate.
Set MOJOLEARN_MAMBA1_BACKWARD_NATIVE_DUMP to a freshly generated base fixture
native dump to require public Python/native bit equality as well.
"""
import importlib.util
import os
from pathlib import Path
import unittest

import numpy as np

from mojolearn import Mamba1Block
from mojolearn.tests.test_mamba_surface import corpus_root, m1_weights, f32


class Mamba1BackwardSurface(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        root = Path(corpus_root()).parent.parent
        path = root / "tools" / "mamba_gradient_oracle.py"
        spec = importlib.util.spec_from_file_location("m1_api_oracle", path)
        cls.oracle = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.oracle)
        cls.case_dir = Path(corpus_root()) / "base_b2_l4_d8"

    def setUp(self):
        self.weights = m1_weights(str(self.case_dir), 8)
        self.x = f32(str(self.case_dir / "x.f32"), (2, 4, 8))
        self.block = Mamba1Block(self.weights, numeric_mode="identical")
        # Dense asymmetric cotangent different from the native fixture's.
        i = np.arange(self.x.size).reshape(self.x.shape)
        self.dy = (((i * 19 + 5) % 43 - 21) / 32).astype(np.float32)

    def reference(self):
        torch = self.torch
        x = torch.tensor(self.x, dtype=torch.float64, requires_grad=True)
        params = {name: torch.tensor(value, dtype=torch.float64, requires_grad=True)
                  for name, value in self.weights.items()}
        stages = self.oracle.GEN.block_forward(params, x, torch.float64)
        loss = (stages["block.out"] * torch.tensor(self.dy, dtype=torch.float64)).sum()
        values = torch.autograd.grad(loss, [x] + list(params.values()))
        return dict(zip(["x"] + list(params),
                        [v.detach().numpy() for v in values]))

    def assert_reference(self, got):
        expected = self.reference()
        self.assertEqual(set(got), set(expected))
        for name, value in got.items():
            with self.subTest(gradient=name):
                self.assertEqual(value.dtype, np.float32)
                self.assertTrue(value.flags.c_contiguous)
                np.testing.assert_allclose(value.reshape(expected[name].shape),
                                           expected[name], rtol=1e-5, atol=1e-6)

    def test_arbitrary_cotangent_and_independent_storage(self):
        got = self.block.backward(self.x, self.dy)
        self.assert_reference(got)
        buffers = [self.x, self.dy] + list(self.weights.values())
        for name, value in got.items():
            for other in buffers:
                self.assertFalse(np.shares_memory(value, other), name)
            buffers.append(value)
        again = self.block.backward(self.x, self.dy)
        for name in got:
            np.testing.assert_array_equal(got[name].view(np.uint32),
                                          again[name].view(np.uint32))
            self.assertFalse(np.shares_memory(got[name], again[name]))

    def test_recomputes_after_weights_change(self):
        before = self.block.backward(self.x, self.dy)
        self.weights["out_proj.weight"] *= np.float32(0.75)
        after = self.block.backward(self.x, self.dy)
        self.assert_reference(after)
        self.assertFalse(np.array_equal(before["x"], after["x"]))

    def test_zero_cotangent_and_noncontiguous_input(self):
        zero = self.block.backward(self.x, np.zeros_like(self.x))
        for value in zero.values():
            self.assertTrue(np.all(value == 0))
        self.x = self.x[:, ::-1, :]
        self.dy = self.dy[:, ::-1, :]
        self.assert_reference(self.block.backward(self.x, self.dy))

    def test_boundary_refusals(self):
        for mode in ("fast", "deterministic"):
            block = Mamba1Block(self.weights, numeric_mode=mode)
            with self.assertRaises(NotImplementedError):
                block.backward(self.x, self.dy)
        with self.assertRaises(TypeError):
            self.block.backward(self.x, self.dy.astype(np.float64))
        with self.assertRaises(TypeError):
            self.block.backward(self.x.astype(np.float64), self.dy)
        with self.assertRaises(ValueError):
            self.block.backward(self.x, self.dy[:, :1])
        for invalid in (np.nan, np.inf, -np.inf):
            dy = self.dy.copy()
            dy.flat[3] = invalid
            with self.assertRaises(ValueError):
                self.block.backward(self.x, dy)
        with self.assertRaises(ValueError):
            self.block.backward(self.x[:, :0], self.dy[:, :0])
        with self.assertRaises(TypeError):
            self.block.backward(self.x, self.dy, state=self.block.allocate_state(2))
        with self.assertRaises(ValueError):
            self.block.backward(self.x[:, 0], self.dy[:, 0])

    def test_native_fixture_bytes_when_requested(self):
        directory = os.environ.get("MOJOLEARN_MAMBA1_BACKWARD_NATIVE_DUMP")
        if not directory:
            self.skipTest("native fixture dump not requested; no byte certificate claimed")
        i = np.arange(self.x.size).reshape(self.x.shape)
        numer = (i * 37 + 11) % 31 - 15
        dy = (np.where(numer == 0, 1, numer) / 16).astype(np.float32)
        got = self.block.backward(self.x, dy)
        for name, value in got.items():
            expected = np.fromfile(Path(directory) / f"grad.{name}.f32", dtype="<f4")
            np.testing.assert_array_equal(value.reshape(-1).view(np.uint32),
                                          expected.view(np.uint32))


if __name__ == "__main__":
    unittest.main()
