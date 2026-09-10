# SPDX-License-Identifier: Apache-2.0
"""Root-only Mamba2/3 backward qualification.

Default execution is host-only ABI/refusal validation with a fake extension.
Set MOJOLEARN_MAMBA23_BACKWARD_NVIDIA=1 only on a RunPod NVIDIA host to run
actual tiny VJPs against float64 autograd, and optionally exact native dumps.
Fixture dependency: only x.f32 and the nine constructor weight files in
mamba/corpus/mamba{2,3}/m{2,3}_base_b2_l4_d32; no ref32/ref64 files or full
corpus generation is required. The NVIDIA oracle additionally imports the
repository gen_corpus.py and requires CUDA PyTorch plus NumPy.
No test in this file provisions hardware or spawns another process.
"""
import ctypes
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from mojolearn import Mamba2Block, Mamba3Block
from mojolearn.tests.test_mamba_surface import (
    corpus_root, f32, m2_weights, m3_weights_corpus,
)


def fixture(family):
    directory = Path(corpus_root()) / f"mamba{family}" / f"m{family}_base_b2_l4_d32"
    loader = m2_weights if family == 2 else m3_weights_corpus
    return loader(str(directory), 32), f32(str(directory / "x.f32"), (2, 4, 32))


def cotangent(x, fixture_objective=False):
    i = np.arange(x.size).reshape(x.shape)
    if fixture_objective:
        numerator = (i * 37 + 11) % 31 - 15
        return (np.where(numerator == 0, 1, numerator) / 16).astype(np.float32)
    return (((i * 19 + 5) % 43 - 21) / 32).astype(np.float32)


class Mamba23BackwardHostSurface(unittest.TestCase):
    def test_folded_abi_names_layout_and_ownership(self):
        for family, cls in ((2, Mamba2Block), (3, Mamba3Block)):
            with self.subTest(family=family):
                weights, x = fixture(family)
                if family == 2:
                    weights["conv1d.weight"] = weights["conv1d.weight"].reshape(-1, 4)
                block = cls(weights, numeric_mode="identical")
                if family == 2:
                    block.dt_limit = (0.01, 0.2)
                # Copies for noncontiguous input/cotangent must remain alive.
                x = x[:, ::-1, :]
                dy = cotangent(x)[:, ::-1, :]
                expected_inputs = [x] + list(weights.values()) + [dy]
                output_templates = [x] + list(weights.values())
                calls = []

                def native(addresses, params):
                    self.assertEqual(len(addresses), 21)
                    self.assertEqual(params, [2, 4, 32] + ([0.01, 0.2] if family == 2 else []))
                    for address, expected in zip(addresses[:11], expected_inputs):
                        observed = np.ctypeslib.as_array((ctypes.c_float * expected.size).from_address(address))
                        np.testing.assert_array_equal(observed.view(np.uint32), expected.reshape(-1).view(np.uint32))
                    for index, (address, template) in enumerate(zip(addresses[11:], output_templates)):
                        target = np.ctypeslib.as_array((ctypes.c_float * template.size).from_address(address))
                        target[:] = index + 0.25
                    calls.append(params)

                extension = SimpleNamespace(**{f"mamba{family}_backward": native})
                with patch.object(block, "_extension", return_value=extension):
                    got = block.backward(x, dy)
                    again = block.backward(x, dy)
                self.assertEqual(len(calls), 2)
                self.assertEqual(tuple(got), ("x",) + block._W_NAMES)
                buffers = expected_inputs + list(again.values())
                for index, (name, value) in enumerate(got.items()):
                    # DEVIATION 2460: gradients are mojolearn.Array; the
                    # zero-copy NumPy view carries dtype, flags and memory.
                    value = np.asarray(value)
                    self.assertEqual(value.shape, output_templates[index].shape)
                    self.assertEqual(value.dtype, np.float32)
                    self.assertTrue(value.flags['C_CONTIGUOUS'])
                    self.assertTrue(np.all(value == index + 0.25))
                    for other in buffers:
                        self.assertFalse(np.shares_memory(value, np.asarray(other)), name)
                    buffers.append(value)

    def test_refusals_precede_loading_native_extension(self):
        for family, cls in ((2, Mamba2Block), (3, Mamba3Block)):
            weights, x = fixture(family)
            dy = cotangent(x)
            for mode in ("fast", "deterministic"):
                block = cls(weights, numeric_mode=mode)
                with patch.object(block, "_extension", side_effect=AssertionError("unexpected native load")):
                    with self.assertRaisesRegex(NotImplementedError, "IDENTICAL"):
                        block.backward(x, dy)
            block = cls(weights, numeric_mode="identical")
            with patch.object(block, "_extension", side_effect=AssertionError("unexpected native load")):
                for bad_x, bad_dy, error in (
                    (x.astype(np.float64), dy, TypeError),
                    (x, dy.astype(np.float64), TypeError),
                    (x, dy[:, :1], ValueError),
                    (x[:, :0], dy[:, :0], ValueError),
                    (x[:0], dy[:0], ValueError),
                    (x[:, 0], dy[:, 0], ValueError),
                ):
                    with self.subTest(family=family, shape=bad_x.shape, error=error):
                        with self.assertRaises(error):
                            block.backward(bad_x, bad_dy)
                for bad in (np.nan, np.inf, -np.inf):
                    invalid = dy.copy()
                    invalid.flat[1] = bad
                    with self.assertRaisesRegex(ValueError, "finite"):
                        block.backward(x, invalid)
                with self.assertRaises(TypeError):
                    block.backward(x, dy, state=None)
                block._w[1] = block._w[1][:-1]
                with self.assertRaisesRegex(ValueError, "in_proj.weight"):
                    block.backward(x, dy)

    def test_stale_binary_has_actionable_error(self):
        for family, cls in ((2, Mamba2Block), (3, Mamba3Block)):
            weights, x = fixture(family)
            block = cls(weights, numeric_mode="identical")
            with patch.object(block, "_extension", return_value=SimpleNamespace()):
                with self.assertRaisesRegex(RuntimeError, f"lacks mamba{family}_backward"):
                    block.backward(x, cotangent(x))


@unittest.skipUnless(os.environ.get("MOJOLEARN_MAMBA23_BACKWARD_NVIDIA") == "1",
                     "root must opt in on RunPod NVIDIA; host ABI checks are not GPU qualification")
class Mamba23BackwardNVIDIA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        if not torch.cuda.is_available() or torch.version.hip is not None:
            raise RuntimeError("Mamba23 qualification requires NVIDIA CUDA")
        path = Path(corpus_root()) / "gen_corpus.py"
        previous_determinism = torch.are_deterministic_algorithms_enabled()
        cls.addClassCleanup(torch.use_deterministic_algorithms, previous_determinism)
        spec = importlib.util.spec_from_file_location("mamba23_api_generator", path)
        cls.gen = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.gen)
        # The external CUDA float64 oracle is compared by tolerance, not by
        # bits. Torch 2.4 rejects CUDA cumsum under the corpus generator's
        # CPU-only deterministic setting. Native IDENTICAL repeat/dump
        # equality below remains strict and independent of this setting.
        torch.use_deterministic_algorithms(False)

    def test_arbitrary_cotangent_all_public_leaves_and_repeat_bits(self):
        for family, cls in ((2, Mamba2Block), (3, Mamba3Block)):
            with self.subTest(family=family):
                weights, x = fixture(family)
                block = cls(weights, numeric_mode="identical")
                self.assertEqual(str(block._extension().mamba_vendor()), "cuda",
                                 "loaded Mojo extension must target NVIDIA")
                dy = cotangent(x)
                tx = self.torch.tensor(x, dtype=self.torch.float64, device="cuda", requires_grad=True)
                params = {name: self.torch.tensor(value, dtype=self.torch.float64, device="cuda", requires_grad=True)
                          for name, value in weights.items()}
                forward = self.gen.m2_forward if family == 2 else self.gen.m3_forward
                # The literal Mamba3 reference creates a few zero states and
                # masks without a device argument. Scope their defaults too.
                with self.torch.device("cuda"):
                    stages = forward(params, tx, self.torch.float64)
                    self.assertEqual(stages["residual.out"].device.type, "cuda")
                    loss = (stages["residual.out"].reshape(tx.shape)
                            * self.torch.tensor(dy, dtype=self.torch.float64, device="cuda")).sum()
                    grads = self.torch.autograd.grad(loss, [tx] + list(params.values()))
                self.torch.cuda.synchronize()
                got = block.backward(x, dy)
                again = block.backward(x, dy)
                self.assertEqual(tuple(got), ("x",) + tuple(weights))
                for (name, value), expected in zip(got.items(), grads):
                    np.testing.assert_allclose(value, expected.detach().cpu().numpy(), rtol=1e-5, atol=1e-6,
                                               err_msg=f"mamba{family}.{name}")
                    np.testing.assert_array_equal(np.asarray(value).view(np.uint32),
                                                  np.asarray(again[name]).view(np.uint32))  # DEVIATION 2460
                zero = block.backward(x, np.zeros_like(x))
                for name, value in zero.items():
                    self.assertTrue(np.all(value == 0), f"mamba{family}.{name}")

    def test_native_fixture_bits_when_requested(self):
        for family, cls in ((2, Mamba2Block), (3, Mamba3Block)):
            directory = os.environ.get(f"MOJOLEARN_MAMBA{family}_BACKWARD_NATIVE_DUMP")
            if not directory:
                self.skipTest("both native dumps required for Python/native byte certificate")
            weights, x = fixture(family)
            block = cls(weights, numeric_mode="identical")
            self.assertEqual(str(block._extension().mamba_vendor()), "cuda")
            got = block.backward(x, cotangent(x, fixture_objective=True))
            for name, value in got.items():
                expected = np.fromfile(Path(directory) / f"grad.{name}.f32", dtype="<f4")
                np.testing.assert_array_equal(np.asarray(value).reshape(-1).view(np.uint32), expected.view(np.uint32),
                                              err_msg=f"mamba{family}.{name}")


if __name__ == "__main__":
    unittest.main()
