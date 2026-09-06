#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tiny arithmetic-contract sabotage tests; no model or GPU execution."""
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

import mamba3_backward_arithmetic as arithmetic


class ArithmeticContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.oracle = Path(self.tmp.name) / "oracle"
        self.actual = Path(self.tmp.name) / "actual"
        self.oracle.mkdir()
        self.actual.mkdir()
        self.forward = {
            "rot.k": np.zeros((2, 1, 128), np.float32),
            "bcnorm.B": np.zeros((2, 1, 128), np.float32),
            "bcnorm.C": np.zeros((2, 1, 128), np.float32),
            "B_bias": np.zeros((1, 128), np.float32),
            "C_bias": np.zeros((1, 128), np.float32),
            "dt.out": np.array([[[2.0], [3.0]]], np.float32),
            "trap.sigma": np.array([[[0.25], [0.5]]], np.float32),
            "angle.theta": np.zeros((2, 1, 32), np.float32),
        }
        self.forward["rot.k"][:, :, 0] = 1.0
        self.forward["bcnorm.B"][:, :, 0] = 1.0
        self.gradients = {name: np.zeros(2, np.float32) for name in arithmetic.GRADIENT_OPERANDS}
        self.gradients["partial.qkdot.dt"][:] = [5, 7]
        self.gradients["partial.s16.kscale"] = np.zeros((2, 1, 128), np.float32)
        self.gradients["partial.s16.kscale"][:, 0, 0] = [4, 8]
        self.gradients["partial.s17.recur.kscale"] = np.zeros((2, 1, 128), np.float32)
        self.outputs = arithmetic.evaluate(self.gradients, self.forward)
        self.manifest = {"forward_operands": {}, "gradients": {}}
        self.dump = {"numeric_mode": "IDENTICAL", "forward_operands": list(arithmetic.FORWARD_OPERANDS),
                     "tensors": list(arithmetic.GRADIENT_OPERANDS) + list(arithmetic.OUTPUTS)}
        for name, value in self.forward.items():
            self.store(name, value, "operand", self.manifest["forward_operands"])
        for name, value in {**self.gradients, **self.outputs}.items():
            self.store(name, value, "grad", self.manifest["gradients"])

    def store(self, name, values, prefix, table):
        filename = f"{prefix}.{name}.f64"
        raw = np.asarray(values, dtype="<f8").tobytes()
        (self.oracle / filename).write_bytes(raw)
        native = np.asarray(values, dtype="<f4").tobytes()
        (self.actual / f"{prefix}.{name}.f32").write_bytes(native)
        ref32 = f"{prefix}.{name}.ref32.f32"
        (self.oracle / ref32).write_bytes(native)
        table[name] = {"file": filename, "shape": list(values.shape), "sha256": hashlib.sha256(raw).hexdigest(),
                       "ref32_file": ref32, "ref32_sha256": hashlib.sha256(native).hexdigest()}

    def audit(self):
        return arithmetic.audit(self.oracle, self.actual, self.manifest, self.dump, 1e-5, 1e-6)[0]

    def test_two_token_scale_chain_and_complete_audit(self):
        np.testing.assert_array_equal(self.outputs["partial.dt.current_total"], [6.0, 13.0])
        np.testing.assert_array_equal(self.outputs["partial.join.dt.current_total"], [6.0, 13.0])
        self.assertEqual(self.audit(), [])

    def test_fma_keeps_residual_lost_by_split_multiply_add(self):
        a = np.array([[1, 1 + 2**-23]], np.float32)
        b = np.array([[1, -(1 - 2**-23)]], np.float32)
        self.assertEqual(float(arithmetic.serial_fma_dot(a, b)[0]), 2**-46)

    def test_one_bit_sabotage_of_every_contract_output(self):
        for name, values in self.outputs.items():
            with self.subTest(name=name):
                path = self.actual / f"grad.{name}.f32"
                bad = values.copy().reshape(-1)
                bad.view(np.uint32)[0] ^= np.uint32(1)
                bad.tofile(path)
                self.assertIn(f"{name}: exact-operand arithmetic failed", self.audit())
                values.tofile(path)

    def test_operand_error_is_rejected_even_when_dot_is_unchanged(self):
        name = "partial.s16.kscale"
        values = self.gradients[name].copy()
        values[0, 0, 1] = 1.0  # Corresponding krot cell is zero.
        values.tofile(self.actual / f"grad.{name}.f32")
        # Keep downstream outputs self-consistent with the corrupt operand.
        # Only the independent operand oracle can reject this construction.
        for output, result in arithmetic.evaluate({**self.gradients, name: values}, self.forward).items():
            result.tofile(self.actual / f"grad.{output}.f32")
        failures = self.audit()
        self.assertTrue(any("operand semantics failed" in row for row in failures))
        self.assertFalse(any("arithmetic failed" in row for row in failures))

    def test_matching_float32_cannot_override_float64_operand(self):
        name = "partial.qkdot.dt"
        entry = self.manifest["gradients"][name]
        values = np.array([9.0, 7.0], np.float32)
        values.tofile(self.actual / f"grad.{name}.f32")
        raw = values.tobytes()
        (self.oracle / entry["ref32_file"]).write_bytes(raw)
        entry["ref32_sha256"] = hashlib.sha256(raw).hexdigest()
        self.assertIn(f"{name}: independent float64 operand semantics failed", self.audit())

    def test_omitted_forward_operand_and_wrong_mode_are_rejected(self):
        self.dump["numeric_mode"] = "FAST"
        with self.assertRaises(ValueError):
            self.audit()
        self.dump["numeric_mode"] = "IDENTICAL"
        self.dump["forward_operands"].remove("rot.k")
        del self.manifest["forward_operands"]["rot.k"]
        with self.assertRaises(ValueError):
            self.audit()

    def test_flattened_forward_never_shifts_beta_across_batches(self):
        forward = {**self.forward, "dt.out": self.forward["dt.out"].reshape(2, 1),
                   "trap.sigma": self.forward["trap.sigma"].reshape(2, 1)}
        gradients = {**self.gradients, "partial.s16.kscale": self.gradients["partial.s16.kscale"].reshape(2, 1, 1, 128)}
        result = arithmetic.evaluate(gradients, forward)
        np.testing.assert_array_equal(result["partial.dt.current_total"], [6, 11])

    def test_rotation_and_angle_semantics_cannot_be_bypassed(self):
        for name in ("angle.theta", "rot.k"):
            with self.subTest(name=name):
                values = self.forward[name].copy()
                values.reshape(-1)[0] += np.float32(0.5)
                values.tofile(self.actual / f"operand.{name}.f32")
                self.assertTrue(any("operand semantics failed" in row for row in self.audit()))
                self.forward[name].tofile(self.actual / f"operand.{name}.f32")


if __name__ == "__main__":
    unittest.main()
