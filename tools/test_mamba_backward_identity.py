#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU-only fail-closed tests for retained Mamba backward certificates.

Fixtures retain every public leaf for all five production certificate cases;
small synthetic float32 tensors make corruption tests independent of a GPU.
Run: python -m unittest discover -s tools -p test_mamba_backward_identity.py
"""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import struct
import tempfile
import unittest

import mamba_backward_identity as identity


LEAVES = {
    "mamba1": ["x", "norm.weight", "in_proj.weight", "conv1d.weight",
               "conv1d.bias", "out_proj.weight", "D", "A_log",
               "dt_proj.weight", "dt_proj.bias", "x_proj.weight"],
    "mamba2": ["x", "block_norm.weight", "in_proj.weight", "conv1d.weight",
               "conv1d.bias", "dt_bias", "A_log", "D", "norm.weight",
               "out_proj.weight"],
    "mamba3": ["x", "block_norm.weight", "in_proj.weight", "dt_bias",
               "B_norm.weight", "C_norm.weight", "B_bias", "C_bias", "D",
               "out_proj.weight"],
}
CASES = {
    "mamba1": ("mamba1", "base_b2_l4_d8"),
    "mamba2": ("mamba2", "m2_base_b2_l4_d32"),
    "mamba3": ("mamba3", "m3_base_b2_l4_d32"),
    "mamba2-l257": ("mamba2", "m2_base_b1_l257_d64"),
    "mamba2-state": ("mamba2", "m2_base_b1_l257_d64"),
}
SOURCE = "a" * 64
POLICY = "incoming_state_before_chunk0; block_output_objective; final_state_cotangent=zero"


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def mutate_json(path, change):
    value = json.loads(path.read_text())
    change(value)
    write_json(path, value)


class BackwardIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.baseline = self.root / "apple"
        self.make_certificate(self.baseline, "apple")
        self.other = self.root / "nvidia"
        shutil.copytree(self.baseline, self.other)
        self.environment(self.other, vendor="nvidia")

    def environment(self, root, **updates):
        path = root / "environment.txt"
        values = dict(line.split("=", 1) for line in path.read_text().splitlines())
        values.update(updates)
        path.write_text("".join(f"{k}={v}\n" for k, v in values.items()))

    def make_certificate(self, root, vendor):
        root.mkdir()
        (root / "environment.txt").write_text(
            f"commit={'b' * 40}\nsource_sha256={SOURCE}\nmode=IDENTICAL\nvendor={vendor}\n"
        )
        (root / "device.csv").write_text("name,vendor\nsynthetic-test-device," + vendor + "\n")
        for label, (family, case) in CASES.items():
            directory = root / label
            actual = directory / "actual"
            oracle = directory / "oracle"
            actual.mkdir(parents=True)
            oracle.mkdir()
            leaves = LEAVES[family]
            state = ["initial_state"] if family == "mamba2" else []
            dump = dict(schema="mojolearn.mamba.gradient-dump.v1", family=family,
                        case=case, objective="signed_dyadic_weight_v1", partial=False,
                        public_prefill_leaves=leaves, state_boundary_leaves=state,
                        tensors=leaves + state)
            reference = dict(schema="mojolearn.mamba.gradient-oracle.v1", family=family,
                             case=case, objective="sum(flat(block_output) * signed_dyadic_weight_v1)",
                             public_prefill_leaves=leaves, state_boundary_leaves=state,
                             gradients={name: {"shape": [2, 2]} for name in leaves + state})
            if state:
                dump["state_boundary_policy"] = reference["state_boundary_policy"] = POLICY
            for i, name in enumerate(leaves + state):
                (actual / f"grad.{name}.f32").write_bytes(struct.pack("<4f", i, -.5, .25, 0.0))
            write_json(actual / "dump_manifest.json", dump)
            write_json(oracle / "manifest.json", reference)
            identity.capture(actual, oracle, directory / "native", SOURCE)
            shutil.copyfile(actual / "dump_manifest.json", directory / "dump_manifest.json")
            shutil.copyfile(oracle / "manifest.json", directory / "oracle_manifest.json")
        self.rehash_results(root)

    def rehash_results(self, root):
        rows = ["family\tverdict\texit_code\toracle_sha256\tdump_sha256"]
        for label in CASES:
            directory = root / label
            rows.append("\t".join([label, "GREEN", "0",
                                  identity.digest(directory / "oracle_manifest.json"),
                                  identity.digest(directory / "dump_manifest.json")]))
        (root / "results.tsv").write_text("\n".join(rows) + "\n")

    def assert_rejected(self):
        with self.assertRaises((ValueError, OSError, KeyError, TypeError)):
            identity.compare([self.baseline, self.other])

    def test_complete_five_case_certificate_passes(self):
        with redirect_stdout(io.StringIO()) as output:
            identity.compare([self.baseline, self.other])
        self.assertIn("BITWISE PASS", output.getvalue())
        self.assertEqual(set(identity.load_certificate(self.other)[1]), set(CASES))

    def test_changed_gradient_byte_rejected(self):
        path = self.other / "mamba2/native/grad.x.f32"
        data = bytearray(path.read_bytes())
        data[0] ^= 1
        path.write_bytes(data)
        self.assert_rejected()

    def test_rehashed_changed_gradient_rejected_by_comparison(self):
        path = self.other / "mamba2/native/grad.x.f32"
        path.write_bytes(struct.pack("<4f", 1, 2, 3, 4))
        mutate_json(path.parent / "manifest.json",
                    lambda m: m["tensors"]["x"].update(sha256=identity.digest(path)))
        self.assert_rejected()

    def test_environment_refusals(self):
        for key, value in [("mode", "FAST"), ("source_sha256", "c" * 64),
                           ("commit", "d" * 40), ("source_changed", "1"),
                           ("vendor", "unknown")]:
            with self.subTest(key=key):
                original = (self.other / "environment.txt").read_text()
                self.environment(self.other, **{key: value})
                self.assert_rejected()
                (self.other / "environment.txt").write_text(original)

    def test_missing_hardware_inventory_rejected(self):
        (self.other / "device.csv").write_text("")
        self.assert_rejected()

    def test_red_numeric_gate_rejected(self):
        path = self.other / "results.tsv"
        path.write_text(path.read_text().replace("GREEN\t0", "RED\t1", 1))
        self.assert_rejected()

    def test_missing_case_rejected(self):
        path = self.other / "results.tsv"
        path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
        self.assert_rejected()

    def test_stale_manifest_rejected(self):
        for filename in ["dump_manifest.json", "oracle_manifest.json"]:
            with self.subTest(filename=filename):
                path = self.other / "mamba2" / filename
                original = path.read_bytes()
                path.write_bytes(original + b" ")
                self.assert_rejected()
                path.write_bytes(original)

    def test_native_metadata_refusals(self):
        path = self.other / "mamba2/native/manifest.json"
        for key, value in [("mode", "FAST"), ("source_sha256", "c" * 64),
                           ("case", "wrong-case"), ("public_prefill_leaves", ["x"])]:
            with self.subTest(key=key):
                original = path.read_bytes()
                mutate_json(path, lambda m: m.update({key: value}))
                self.assert_rejected()
                path.write_bytes(original)

    def test_rehashed_oracle_semantic_mismatch_rejected(self):
        path = self.other / "mamba2/oracle_manifest.json"
        for key, value in [("family", "mamba3"), ("case", "wrong-case"),
                           ("objective", "wrong-objective"),
                           ("public_prefill_leaves", ["x"]),
                           ("state_boundary_leaves", []),
                           ("state_boundary_policy", "wrong-policy")]:
            with self.subTest(key=key):
                original = path.read_bytes()
                mutate_json(path, lambda m: m.update({key: value}))
                self.rehash_results(self.other)
                self.assert_rejected()
                path.write_bytes(original)
                self.rehash_results(self.other)

    def test_rehashed_oracle_shape_mismatch_rejected(self):
        path = self.other / "mamba2/oracle_manifest.json"
        mutate_json(path, lambda m: m["gradients"]["x"].update(shape=[4]))
        self.rehash_results(self.other)
        self.assert_rejected()

    def test_capture_rejects_mismatched_inventory_and_semantics(self):
        directory = self.baseline / "mamba2"
        path = directory / "oracle/manifest.json"
        for key, value in [("case", "wrong-case"), ("objective", "wrong-objective"),
                           ("public_prefill_leaves", ["x"]),
                           ("state_boundary_leaves", []),
                           ("state_boundary_policy", "wrong-policy")]:
            with self.subTest(key=key):
                original = path.read_bytes()
                mutate_json(path, lambda m: m.update({key: value}))
                with self.assertRaises((ValueError, KeyError, TypeError)):
                    identity.capture(directory / "actual", directory / "oracle",
                                     self.root / ("capture-" + key), SOURCE)
                path.write_bytes(original)

    def test_consistently_wrong_case_rejected_before_comparison(self):
        directory = self.other / "mamba2"
        for filename in ["dump_manifest.json", "oracle_manifest.json", "native/manifest.json"]:
            mutate_json(directory / filename, lambda m: m.update(case="m2_base_b1_l257_d64"))
        self.rehash_results(self.other)
        with self.assertRaises(ValueError):
            identity.load_certificate(self.other)

    def test_consistently_missing_leaf_rejected_before_comparison(self):
        directory = self.other / "mamba2"
        for filename in ["dump_manifest.json", "oracle_manifest.json", "native/manifest.json"]:
            def remove_leaf(manifest):
                manifest["public_prefill_leaves"].remove("x")
                if "gradients" in manifest:
                    del manifest["gradients"]["x"]
                if "tensors" in manifest:
                    if isinstance(manifest["tensors"], list):
                        manifest["tensors"].remove("x")
                    else:
                        del manifest["tensors"]["x"]
            mutate_json(directory / filename, remove_leaf)
        self.rehash_results(self.other)
        with self.assertRaises(ValueError):
            identity.load_certificate(self.other)

    def test_duplicate_case_rejected(self):
        path = self.other / "results.tsv"
        path.write_text(path.read_text() + path.read_text().splitlines()[1] + "\n")
        self.assert_rejected()

    def test_capture_rejects_undeclared_leaf(self):
        directory = self.baseline / "mamba2"
        mutate_json(directory / "actual/dump_manifest.json",
                    lambda m: m["tensors"].remove("x"))
        with self.assertRaises(ValueError):
            identity.capture(directory / "actual", directory / "oracle",
                             self.root / "undeclared", SOURCE)

    def test_capture_rejects_wrong_float32_size(self):
        directory = self.baseline / "mamba2"
        (directory / "actual/grad.x.f32").write_bytes(b"\0" * 15)
        with self.assertRaises(ValueError):
            identity.capture(directory / "actual", directory / "oracle", self.root / "bad-size", SOURCE)


if __name__ == "__main__":
    unittest.main()
