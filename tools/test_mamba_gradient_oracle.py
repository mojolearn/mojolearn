#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Sabotage the strict backward gate using tiny synthetic dump directories.

Run with: pixi run -e skgpu python tools/test_mamba_gradient_oracle.py
These tests exercise certification policy; numerical correctness is covered
by the independent gradient oracle and native public-prefill tasks.
"""

import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import mamba_gradient_oracle as oracle


class StrictGradientPolicyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.expected = Path(self.tmp.name) / "oracle"
        self.actual = Path(self.tmp.name) / "actual"
        self.expected.mkdir()
        self.actual.mkdir()
        public = list(oracle.PUBLIC_PREFILL_LEAVES["mamba2"])
        names = public + ["initial_state", "stage.initial_state"]
        common = {
            "family": "mamba2", "case": "strict_policy_fixture",
            "public_prefill_leaves": public,
            "state_boundary_leaves": ["initial_state"],
            "state_boundary_diagnostics": ["stage.initial_state"],
            "state_boundary_policy": (
                "incoming_state_before_chunk0; block_output_objective; "
                "final_state_cotangent=zero"
            ),
        }
        self.manifest = dict(common, gradients={})
        self.dump = dict(
            common, schema="mojolearn.mamba.gradient-dump.v1",
            objective="signed_dyadic_weight_v1", tensors=names,
        )
        for name in names:
            self.manifest["gradients"][name] = {
                "file": f"grad.{name}.f64", "shape": [1],
            }
            np.array([1.0], dtype="<f8").tofile(
                self.expected / f"grad.{name}.f64"
            )
            self.write_actual(name, 1.0)

    def write_actual(self, name, value):
        np.array([value], dtype="<f4").tofile(
            self.actual / f"grad.{name}.f32"
        )

    def compare(self, public=False):
        (self.expected / "manifest.json").write_text(json.dumps(self.manifest))
        (self.actual / "dump_manifest.json").write_text(json.dumps(self.dump))
        args = argparse.Namespace(
            compare=[str(self.expected), str(self.actual)],
            require_public_prefill=public, require_state_boundary=getattr(self, "state", True),
            allow_partial=False, rtol=1e-5, atol=1e-6,
        )
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            oracle.compare(args)

    def test_valid_state_and_combined_gates(self):
        self.compare()
        self.compare(public=True)

    def test_combined_gate_checks_public_numerics(self):
        self.write_actual("x", 3.0)
        self.compare()  # The standalone state gate deliberately scopes to state.
        with self.assertRaises(SystemExit):
            self.compare(public=True)

    def test_missing_state_leaf_or_diagnostic_file_fails(self):
        for name in ("initial_state", "stage.initial_state"):
            with self.subTest(name=name):
                (self.actual / f"grad.{name}.f32").unlink()
                with self.assertRaises(SystemExit):
                    self.compare()
                self.write_actual(name, 1.0)

    def test_nonfinite_state_fails(self):
        self.write_actual("initial_state", float("nan"))
        with self.assertRaises(SystemExit):
            self.compare()

    def test_manifest_sabotage_fails(self):
        mutations = {
            "state_boundary_policy": "nonzero final cotangent",
            "state_boundary_leaves": [],
            "state_boundary_diagnostics": [],
            "family": "mamba3",
            "case": "wrong_case",
            "tensors": ["initial_state", "initial_state", "stage.initial_state"],
        }
        for key, value in mutations.items():
            with self.subTest(key=key):
                previous = self.dump[key]
                self.dump[key] = value
                with self.assertRaises(SystemExit):
                    self.compare()
                self.dump[key] = previous

    def test_missing_oracle_state_leaf_fails(self):
        del self.manifest["gradients"]["initial_state"]
        with self.assertRaises(SystemExit):
            self.compare()


class CompositionalReductionTests(unittest.TestCase):
    write_actual = StrictGradientPolicyTests.write_actual
    compare = StrictGradientPolicyTests.compare

    def setUp(self):
        StrictGradientPolicyTests.setUp(self)
        self.state = False
        self.manifest["case"] = self.dump["case"] = "m2_base_b1_l257_d64"
        self.manifest["gradients"] = {}
        self.dump["tensors"] = []
        for leaf, operand in (
            ("conv1d.bias", "diagnostic.conv.bias_operand"),
            ("block_norm.weight", "diagnostic.block_norm.weight_operand"),
        ):
            for name, shape in ((leaf, [1]), (operand, [1, 257, 1])):
                self.manifest["gradients"][name] = {
                    "file": f"grad.{name}.f64", "shape": shape,
                }
                self.dump["tensors"].append(name)
                value = 32.125 if name == leaf else 0.125
                np.full(shape, value, dtype="<f8").tofile(self.expected / f"grad.{name}.f64")
                np.full(shape, value, dtype="<f4").tofile(self.actual / f"grad.{name}.f32")

    def test_valid_reductions(self):
        self.compare()

    def test_public_leaf_sabotage(self):
        for name in ("conv1d.bias", "block_norm.weight"):
            with self.subTest(name=name):
                self.write_actual(name, np.nextafter(np.float32(32.125), np.float32(33)))
                with self.assertRaises(SystemExit):
                    self.compare()
                self.write_actual(name, 32.125)

    def test_operand_sabotage_even_when_sum_unchanged(self):
        for name in ("diagnostic.conv.bias_operand", "diagnostic.block_norm.weight_operand"):
            with self.subTest(name=name):
                values = np.full(257, 0.125, dtype="<f4")
                values[:2] = [0.25, 0.0]
                values.tofile(self.actual / f"grad.{name}.f32")
                with self.assertRaises(SystemExit):
                    self.compare()
                np.full(257, 0.125, dtype="<f4").tofile(self.actual / f"grad.{name}.f32")

    def test_missing_operand(self):
        (self.actual / "grad.diagnostic.conv.bias_operand.f32").unlink()
        with self.assertRaises(SystemExit):
            self.compare()

    def test_ref32_cannot_override_independent_float64_operand(self):
        name = "diagnostic.conv.bias_operand"
        values = np.full(257, 0.125, dtype="<f4")
        values[:2] = [0.25, 0.0]
        values.tofile(self.actual / f"grad.{name}.f32")
        filename = f"grad.{name}.ref32.f32"
        values.tofile(self.expected / filename)
        self.manifest["gradients"][name]["ref32_file"] = filename
        with self.assertRaises(SystemExit):
            self.compare()


if __name__ == "__main__":
    unittest.main()
