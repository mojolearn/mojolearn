"""Real frozen-cache reuse with bounded source fixtures; no product builds/runs."""
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import release_requirement_dependencies as deps
from test_release_requirement_dependencies import binding, manifest, snapshot, spec
from test_macos_cpu_compile_contract import witness


class SelectedImageAnalysisCache(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "repo"
        self.root.mkdir()
        self.git("init")
        self.git("config", "user.email", "test@example.test")
        self.git("config", "user.name", "Analysis Cache Test")
        (self.root / "kernel.mojo").write_text("original source\n")
        self.git("add", ".")
        self.git("commit", "-m", "fixture")
        self.revision = self.git("rev-parse", "HEAD")
        self.logs = self.base / "logs"
        self.key = "osx-arm64/host/-/host/one"
        self.contract = witness(self.base)
        self.config = dict(build_flags={}, release_modes=["identical"],
            dependency_context=dict(cpu_target="osx-arm64", compiler_sha256={"osx-arm64": "a" * 64},
                selected_binding_build_flags={self.key: {"MOJOLEARN_MOJO_BUILD_FLAGS": "-O0"}},
                selected_binding_compile_jobs={self.key: "10"},
                selected_macos_cpu_toolchains={self.key: self.contract}))
        self.requests = []
        self.snapshots = []

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.root), *args],
            stderr=subprocess.DEVNULL, text=True).strip()

    def worker(self, tree, temporary, request, log):
        # The real frozen_snapshot cache/checkouts run; only the expensive
        # whole-library analyzer is replaced by a small native source graph.
        self.requests.append(copy.deepcopy(request))
        result = snapshot()
        result["revision"] = subprocess.check_output(
            ["git", "-C", str(tree), "rev-parse", "HEAD"], text=True).strip()
        host = binding("one", deps.hashlib.sha256((tree / "kernel.mojo").read_bytes()).hexdigest(),
                       vendor="host", arch="-", target="osx-arm64")
        host.update(key=self.key, tier="host")
        host["identity"].update(tier="host", flags=dict(target_column="cpu", compile_jobs="2"),
                              toolchain=dict(pixi_lock_sha256="b" * 64), host_toolchain=None)
        result.update(bindings=[host], lanes={"one": result["lanes"]["one"]},
                      toolchains={"osx-arm64": {"packages": ["mojo-pin"]}})
        self.snapshots.append(copy.deepcopy(result))
        return result

    def expand(self, config=None, revision=None):
        rows = [dict(spec(kind), id=kind + "-apple") for kind in ("compile", "identity")]
        value = deps.expand_catalog(self.root, revision or self.revision,
            dict(requirements=rows), config or self.config, log_dir=self.logs)
        return manifest(value, revision or self.revision)

    def test_selected_maps_share_real_cache_but_each_final_fingerprint_stays_exact(self):
        original = copy.deepcopy(self.config)
        with patch.object(deps, "_run_snapshot_worker", side_effect=self.worker):
            before = self.expand()
            reports = list(self.logs.glob("*.json"))
            self.assertEqual(len(reports), 1)
            retained = reports[0].read_bytes()
            for field, replacement in (
                ("selected_binding_build_flags", {self.key: {"MOJOLEARN_MOJO_BUILD_FLAGS": "-O3"}}),
                ("selected_binding_compile_jobs", {self.key: "32"}),
                ("selected_macos_cpu_toolchains", {self.key: copy.deepcopy(self.contract)}),
            ):
                with self.subTest(field=field):
                    config = copy.deepcopy(self.config)
                    if field == "selected_macos_cpu_toolchains":
                        replacement[self.key]["image"]["rpaths"] = ["/different/original/toolchain"]
                    config["dependency_context"][field] = replacement
                    after = self.expand(config)
                    changed = {name for name, row in before["requirements"].items()
                        if row["fingerprint"] != after["requirements"][name]["fingerprint"]}
                    self.assertEqual(changed, {"compile-apple/" + self.key, "identity-apple/one"})
                    self.assertEqual(reports[0].read_bytes(), retained)
            self.assertEqual(before, self.expand())
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(self.config, original)
        self.assertNotIn("selected_build_flags", self.snapshots[0]["bindings"][0]["identity"])
        self.assertEqual(self.snapshots[0]["bindings"][0]["identity"]["flags"]["compile_jobs"], "2")
        self.assertIsNone(self.snapshots[0]["bindings"][0]["identity"]["host_toolchain"])

    def test_source_target_compiler_environment_and_manifest_changes_reanalyze(self):
        with patch.object(deps, "_run_snapshot_worker", side_effect=self.worker):
            self.expand()
            for field, value in (
                ("cpu_target", "linux-64"),
                ("compiler_sha256", {"osx-arm64": "c" * 64}),
                ("apple_toolchain", {"sdk": "another original SDK"}),
                ("optional_native_files", {"osx-arm64": {"optional.so": False}}),
                ("environment", {"MOJOLEARN_CUSTOM_SWITCH": "1"}),
            ):
                with self.subTest(field=field):
                    config = copy.deepcopy(self.config)
                    config["dependency_context"][field] = value
                    count = len(self.requests)
                    self.expand(config)
                    self.assertEqual(len(self.requests), count + 1)
                    self.assertEqual(self.requests[-1]["configuration"]["dependency_context"][field], value)
            (self.root / "kernel.mojo").write_text("changed source\n")
            self.git("add", ".")
            self.git("commit", "-m", "source change")
            count = len(self.requests)
            self.expand(revision=self.git("rev-parse", "HEAD"))
            self.assertEqual(len(self.requests), count + 1)

    def test_cached_analysis_does_not_skip_selected_map_validation(self):
        with patch.object(deps, "_run_snapshot_worker", side_effect=self.worker):
            self.expand()
            for field, value in (
                ("selected_binding_build_flags", {"unknown": {}}),
                ("selected_binding_compile_jobs", {self.key: "0"}),
                ("selected_macos_cpu_toolchains", {self.key: {}}),
            ):
                with self.subTest(field=field), self.assertRaises(ValueError):
                    config = copy.deepcopy(self.config)
                    config["dependency_context"][field] = value
                    self.expand(config)
            self.assertEqual(len(self.requests), 1)


if __name__ == "__main__":
    unittest.main()
