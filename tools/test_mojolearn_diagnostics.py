"""Diagnostics must describe the installed split wheels even when imports fail."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "mojolearn_diagnostics", Path(__file__).resolve().parents[1] / "python/mojolearn_diagnostics.py"
)
diagnostics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostics)


class DoctorTests(unittest.TestCase):
    def test_published_platforms_use_loaded_backend_not_gpu_utility_presence(self):
        for system, machine, vendor in [
            ("Linux", "x86_64", "cuda"), ("Linux", "x86_64", "hip"),
            ("Linux", "x86_64", "cpu"), ("Darwin", "arm64", "metal"),
        ]:
            with self.subTest(vendor=vendor), patch.object(diagnostics.platform, "system", return_value=system), \
                    patch.object(diagnostics.platform, "machine", return_value=machine):
                result = diagnostics._support_classification(
                    {"returncode": 0, "result": {"ok": True, "vendor": vendor}})
                self.assertEqual(result["level"], "imported")
                self.assertIn("wheels are published", result["reason"])
                self.assertIn("does not certify", result["reason"])

    def test_linux_import_failure_is_not_reported_as_supported(self):
        with patch.object(diagnostics.platform, "system", return_value="Linux"), \
                patch.object(diagnostics.platform, "machine", return_value="x86_64"):
            for probe in [{"returncode": 1, "result": {"ok": False, "type": "ImportError"}},
                          {"available": False}, {"returncode": 0, "result": {"ok": False}}]:
                self.assertEqual(diagnostics._support_classification(probe)["level"], "load-failed")

    def test_unreadable_backend_and_nonwheel_platform_are_not_certified(self):
        probe = {"returncode": 0, "result": {"ok": True, "vendor": "unreadable: ImportError"}}
        with patch.object(diagnostics.platform, "system", return_value="Linux"), \
                patch.object(diagnostics.platform, "machine", return_value="x86_64"):
            self.assertEqual(diagnostics._support_classification(probe)["level"], "backend-unconfirmed")
        with patch.object(diagnostics.platform, "system", return_value="Windows"), \
                patch.object(diagnostics.platform, "machine", return_value="AMD64"):
            self.assertEqual(diagnostics._support_classification(probe)["level"], "compatibility-report")

    def test_native_inventory_includes_vendor_wheels_and_excludes_missing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            members = {
                "mojolearn": ["mojolearn/host/_mojolearn_rf.so"],
                "mojolearn-nvidia": ["mojolearn/cuda_native/sm_89/identical/_mojolearn_rf.so",
                                     "mojolearn/cuda_ptx/sm_80/identical/_mojolearn_rf.so"],
                "mojolearn-amd": ["mojolearn/hip_native/gfx942/identical/_mojolearn_rf.so"],
            }
            for paths in members.values():
                for name in paths:
                    if "cuda_ptx" not in name:
                        f = root / name
                        f.parent.mkdir(parents=True, exist_ok=True)
                        f.write_bytes(b"fixture native payload")
            def distribution(name):
                return SimpleNamespace(files=members[name], locate_file=lambda p: root / p)
            with patch.object(diagnostics.importlib.metadata, "distribution", side_effect=distribution):
                records = diagnostics._native_files()
            self.assertEqual({r["distribution"] for r in records}, set(members))
            self.assertNotIn(str(root), json.dumps(records))
            self.assertEqual(diagnostics._installed_gpu_sets(records),
                             ["cuda_native/sm_89", "hip_native/gfx942"])
            tiers = diagnostics._installed_tiers(records)
            self.assertEqual(set(tiers["extensions_by_vendor_and_tier"]),
                             {"flat", "cuda_native/sm_89", "hip_native/gfx942"})

    def test_missing_distributions_are_allowed(self):
        with patch.object(diagnostics.importlib.metadata, "distribution",
                          side_effect=diagnostics.importlib.metadata.PackageNotFoundError):
            self.assertEqual(diagnostics._native_files(), [])

    def test_ptx_is_reported_only_when_carried(self):
        records = [{"relative_path": "mojolearn/cuda_ptx/sm_80/identical/_mojolearn_rf.so",
                    "exists": True}]
        self.assertEqual(diagnostics._installed_gpu_sets(records), ["cuda_ptx/sm_80"])

    def test_doctor_reports_split_package_versions_and_preserves_failure_exit(self):
        versions = {"mojolearn": "0.8.42", "mojolearn-nvidia": "0.8.42", "mojolearn-amd": "0.8.42"}
        record = {"relative_path": "mojolearn/cuda_native/sm_89/identical/_mojolearn_rf.so", "exists": True}
        for rc in (0, 1):
            with self.subTest(rc=rc), \
                    patch.object(diagnostics, "_gpu_facts", return_value={}), \
                    patch.object(diagnostics, "_native_files", return_value=[record]), \
                    patch.object(diagnostics, "_distribution_versions", return_value=versions), \
                    patch.object(diagnostics, "_run", return_value={"available": False}), \
                    patch.object(diagnostics, "_import_probe", return_value={
                        "returncode": rc, "result": {"ok": rc == 0, "vendor": "cuda"}
                    }) as probe, contextlib.redirect_stdout(io.StringIO()) as output:
                result = diagnostics.main(["doctor"])
                self.assertEqual(result, rc)
                probe.assert_called_once()
                self.assertIn("mojolearn-nvidia: 0.8.42", output.getvalue())
                self.assertIn("GPU architecture sets installed: cuda_native/sm_89", output.getvalue())
                self.assertNotIn("no Linux wheel", output.getvalue())


if __name__ == "__main__":
    unittest.main()
