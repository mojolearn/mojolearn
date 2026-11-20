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
LAYOUT_SPEC = importlib.util.spec_from_file_location(
    "merged_layout", Path(__file__).resolve().parents[1] / "packaging/linux/merged_layout.py"
)
merged_layout = importlib.util.module_from_spec(LAYOUT_SPEC)
LAYOUT_SPEC.loader.exec_module(merged_layout)


def _installed_layout(root, sets):
    """Write a wheel install under `root` from the REAL merged-group tables
    (bindings/merged_groups.json through packaging/linux/merged_layout.plan):
    `sets` is {distribution: [(directory, plan_key, platform, byte_lm)]}.
    Returns the distribution() stand-in reading its RECORD-style file list."""
    members = {}
    for dist, rows in sets.items():
        files = members.setdefault(dist, [])
        for directory, plan_key, plat, byte_lm in rows:
            plan = merged_layout.plan(plan_key, byte_lm, plat)
            assert plan, (plan_key, plat)
            base = root / directory
            base.mkdir(parents=True, exist_ok=True)
            for name in plan:
                (base / name).write_bytes(b"fixture native payload")
                files.append(f"{directory}/{name}")
            (base / merged_layout.INDEX).write_text(
                json.dumps({"format": merged_layout.INDEX_FORMAT, "files": plan}))
            files.append(f"{directory}/{merged_layout.INDEX}")

    def distribution(name):
        if name not in members:
            raise diagnostics.importlib.metadata.PackageNotFoundError(name)
        return SimpleNamespace(files=members[name], locate_file=lambda p: root / p)
    return distribution


LINUX_NVIDIA = {
    "mojolearn-core": [("mojolearn/host", "host", "linux", False)],
    "mojolearn-nvidia": [("mojolearn/cuda_native/sm_89/identical", "identical", "linux", False),
                         ("mojolearn/cuda_ptx/sm_80/identical", "identical", "linux", True)],
}
LINUX_AMD = {
    "mojolearn-core": [("mojolearn/host", "host", "linux", False)],
    "mojolearn-amd": [("mojolearn/hip_native/gfx942/identical", "identical", "linux", False)],
}
LINUX_CPU = {"mojolearn-core": [("mojolearn/host", "host", "linux", False)]}
MACOS_FLAT = {"mojolearn": [("mojolearn", "fast", "darwin", True),
                            ("mojolearn/identical", "identical", "darwin", True),
                            ("mojolearn/host", "host", "darwin", False)]}


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
                # NATIVE_DISTRIBUTIONS also names mojolearn-core (c34d80a5a); this
                # fixture installs only the three below, so the rest are absent
                if name not in members:
                    raise diagnostics.importlib.metadata.PackageNotFoundError(name)
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
        versions = {"mojolearn": "1.2.3", "mojolearn-nvidia": "1.2.3", "mojolearn-amd": "1.2.3"}
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
                self.assertIn("mojolearn-nvidia: 1.2.3", output.getvalue())
                self.assertIn("GPU architecture sets installed: cuda_native/sm_89", output.getvalue())
                self.assertNotIn("no Linux wheel", output.getvalue())

    def _tiers(self, sets, system, remove=(), drop_indexes=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            distribution = _installed_layout(root, sets)
            for rel in remove:
                (root / rel).unlink()
            with patch.object(diagnostics.importlib.metadata, "distribution", side_effect=distribution), \
                    patch.object(diagnostics.platform, "system", return_value=system):
                records = diagnostics._native_files()
                indexes = [] if drop_indexes else diagnostics._binding_indexes()
                return diagnostics._installed_tiers(records, indexes)

    def test_complete_merged_layouts_raise_no_warning(self):
        # The 2026-11 report: a healthy NVIDIA install read "numeric tiers
        # INCOMPLETE: identical" because 4 merged identical files < 15.
        for label, sets, system, present, answered in [
            ("linux nvidia", LINUX_NVIDIA, "Linux", ["identical"], {"fast": "identical"}),
            ("linux amd", LINUX_AMD, "Linux", ["identical"], {"fast": "identical"}),
            ("linux cpu only", LINUX_CPU, "Linux", ["identical"], {"fast": "identical"}),
            # A measurement wheel's fast-rows set is built into the identical
            # tier directory with the identical plan: the same files.
            ("fast-rows set", {"mojolearn-nvidia": [
                ("mojolearn/cuda_native/sm_89/identical", "identical", "linux", False)]},
             "Linux", ["identical"], {"fast": "identical"}),
            ("macos flat", MACOS_FLAT, "Darwin", ["fast", "identical"], {}),
        ]:
            with self.subTest(label):
                tiers = self._tiers(sets, system)
                self.assertEqual(tiers["incomplete"], [])
                self.assertEqual(tiers["missing_files"], [])
                self.assertEqual(tiers["cannot_tell"], [])
                self.assertEqual(tiers["present"], present)
                self.assertEqual(tiers["absent"], [])
                self.assertEqual(tiers["answered_by"], answered)
                self.assertTrue(tiers["sets"])
                self.assertTrue(all(s["status"] == "complete" for s in tiers["sets"]))
        # mojolearn/host/ is the CPU set, never counted as the fast tier.
        tiers = self._tiers(LINUX_NVIDIA, "Linux")
        self.assertEqual(tiers["extensions_by_tier"]["fast"], 0)
        self.assertEqual({s["directory"]: s["set"] for s in tiers["sets"]}["mojolearn/host"], "host")

    def test_one_missing_file_is_incomplete_and_named(self):
        gone_dir = "mojolearn/cuda_native/sm_89/identical"
        gone = f"{gone_dir}/{sorted(merged_layout.plan('identical', False, 'linux'))[0]}"
        tiers = self._tiers(LINUX_NVIDIA, "Linux", remove=[gone])
        self.assertEqual(tiers["incomplete"], ["identical"])
        self.assertEqual(tiers["missing_files"], [gone])
        statuses = {s["directory"]: s["status"] for s in tiers["sets"]}
        self.assertEqual(statuses[gone_dir], "incomplete")
        self.assertEqual(statuses["mojolearn/cuda_ptx/sm_80/identical"], "complete")
        # The text names the file.
        summary = self._summary(tiers)
        self.assertIn("numeric tiers INCOMPLETE: identical", summary)
        self.assertIn(f"missing: {gone}", summary)

    def test_missing_index_names_it(self):
        index = f"mojolearn/host/{merged_layout.INDEX}"
        tiers = self._tiers(LINUX_CPU, "Linux", remove=[index])
        self.assertEqual(tiers["incomplete"], ["identical"])
        self.assertEqual(len(tiers["missing_files"]), 1)
        self.assertTrue(tiers["missing_files"][0].startswith(index))

    def test_no_manifest_cannot_tell_never_incomplete(self):
        tiers = self._tiers(LINUX_NVIDIA, "Linux", drop_indexes=True)
        self.assertEqual(tiers["incomplete"], [])
        self.assertEqual(sorted(tiers["cannot_tell"]), sorted(
            d for rows in LINUX_NVIDIA.values() for d, *_ in rows))
        summary = self._summary(tiers)
        self.assertNotIn("INCOMPLETE", summary)
        self.assertIn("cannot tell", summary)

    def _summary(self, tiers):
        report = {
            "distributions": {}, "support": {"level": "x", "reason": "y"},
            "import_probe": {"returncode": 1, "result": {"ok": False}},
            "recorded_at_utc": "t", "environment": {}, "gpu_sets": [],
            "platform": {"system": "Linux", "release": "r", "machine": "x86_64",
                         "python_version": "3.12"},
            "numeric_tiers": tiers,
        }
        text = diagnostics._summary(report)
        self.assertTrue(text.startswith("Mojolearn doctor\n"))
        return text


if __name__ == "__main__":
    unittest.main()
