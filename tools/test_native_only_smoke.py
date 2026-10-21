"""Negative controls for the native-only release gate, using test infrastructure."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import native_only_smoke as gate


class NativeOnlyGate(unittest.TestCase):
    def test_transitive_python_import_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "main.mojo").write_text("from nested import run\ndef main():\n    run()\n")
            (root / "nested.mojo").write_text("from std.python import Python\ndef run():\n    pass\n")
            sources, errors = gate.closure(root, "main.mojo")
            self.assertEqual(set(sources), {"main.mojo", "nested.mojo"})
            self.assertTrue(any("Python dependency" in e for e in errors))

    def test_unresolved_local_import_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "main.mojo").write_text("from vanished import compute\n")
            self.assertIn("unresolved import vanished", gate.closure(root, "main.mojo")[1][0])

    def test_pytorch_prose_is_not_a_cpython_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "main.mojo"
            path.write_text('def main() raises:\n    raise Error("PyTorch shape semantics without Python. Native code.")\n')
            self.assertFalse(gate.closure(root, "main.mojo")[1])
            path.write_text('def main():\n    external_call["PyEval_EvalCode"]()\n')
            self.assertTrue(gate.closure(root, "main.mojo")[1])

    def test_source_mutation_changes_closure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "main.mojo").write_text("from nested import run\n")
            nested = root / "nested.mojo"
            nested.write_text("def run():\n    print(1)\n")
            before, _ = gate.closure(root, "main.mojo")
            nested.write_text("def run():\n    print(2)\n")
            after, _ = gate.closure(root, "main.mojo")
            self.assertNotEqual(before, after)

    def test_missing_inventory_is_not_complete(self):
        self.assertTrue(gate.coverage_debt({"cases": []}, "cpu"))

    def test_metadata_exclusion_requires_reason(self):
        manifest = {"cases": [], "inventory": [{"algorithm": "Config", "status": "not_algorithm", "reason": "configuration"}]}
        self.assertFalse(gate.coverage_debt(manifest, "cpu"))
        del manifest["inventory"][0]["reason"]
        self.assertTrue(gate.coverage_debt(manifest, "cpu"))

    def test_declared_algorithm_requires_actual_applicable_case(self):
        manifest = {"cases": [{"id": "gpu", "algorithms": ["A"], "backends": ["metal"], "whole_operation": True}],
                    "inventory": [{"algorithm": "A", "status": "ready", "cases": ["gpu"]}]}
        self.assertTrue(gate.coverage_debt(manifest, "cpu"))
        self.assertFalse(gate.coverage_debt(manifest, "metal"))
        manifest["cases"][0]["whole_operation"] = False
        self.assertTrue(gate.coverage_debt(manifest, "metal"))
        manifest["cases"][0]["whole_operation"] = True
        manifest["cases"][0]["algorithms"] = ["B"]
        self.assertTrue(gate.coverage_debt(manifest, "metal"))

    def test_timeout_is_failure_with_log_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "full.log"
            result = gate.run_command([sys.executable, "-c", "import time; print('started', flush=True); time.sleep(20)"],
                                      log, 0.2, Path(tmp))
            self.assertTrue(result["timeout"])
            self.assertNotEqual(result["exit_code"], 0)
            self.assertIn("started", log.read_text())

    def test_empty_success_without_checked_result_is_failure(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            directory = out / "empty"
            directory.mkdir()
            script = directory / "smoke"
            script.write_text("#!/bin/sh\nexit 0\n")
            script.chmod(0o755)
            record = gate.execute_case({"id": "empty", "marker": "CHECKED"},
                                       {"id": "empty", "status": "built"},
                                       SimpleNamespace(out=out, run_timeout=1), None)
            self.assertEqual(record["status"], "missing_result")

    def test_no_fake_isolation_on_nonlinux(self):
        from types import SimpleNamespace
        with patch.object(gate.sys, "platform", "darwin"):
            with self.assertRaisesRegex(RuntimeError, "no simulated interpreter absence"):
                gate.isolated_root(SimpleNamespace(), [])

    def test_manifest_path_escape_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps({"cases": [{"id": "escape", "source": "../outside.mojo",
                                                  "algorithms": ["A"], "marker": "PASS"}]}))
            with self.assertRaises(ValueError):
                gate.load_manifest(path)


if __name__ == "__main__":
    unittest.main()
