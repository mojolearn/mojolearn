#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE LINUX GPU PLUGINS, AS THE LOADER SEES THEM, ON A MACHINE WITH NO GPU.

    .pixi/envs/test/bin/python -m pytest -q packaging/linux/test_plugin_loader.py

`_backend._layout()` is pure Python over the filesystem, the installed
distributions' metadata and a box probe, so every case below fabricates an
installed site-packages directory (the core's .dist-info with or without its
gpu_plugins.json marker, plugin .dist-infos, set directories) and a probe,
and asks the REAL function. Nothing loads an extension. The package is
loaded under a private name with its __init__ skipped (no binary is built in
a worktree), so the tests run anywhere.

Each case that asserts a choice has a twin whose fixture differs in exactly
the one thing the choice depends on (the selector-test rule: a check must be
seen to move).
"""
import importlib
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG_SRC = ROOT / "python" / "mojolearn"
ALIAS = "mojolearn_plugin_loader_under_test"


def fresh_backend():
    """A fresh `_backend` module under a private package whose __path__ is
    the real package directory, so its relative imports resolve to the real
    gpu_plugins.py, host_surface.py and _version.py and nothing else runs."""
    for name in [n for n in sys.modules if n == ALIAS or n.startswith(ALIAS + ".")]:
        del sys.modules[name]
    pkg = types.ModuleType(ALIAS)
    pkg.__path__ = [str(PKG_SRC)]
    sys.modules[ALIAS] = pkg
    return importlib.import_module(ALIAS + "._backend")


def probe(B, cuda=False, hip=False):
    def one(found, spec):
        return {"paths": {p: found for p in spec["paths"]},
                "libs": {lib: False for lib in spec["libs"]}, "found": found}
    return {"cuda": one(cuda, B._PROBE["cuda"]), "hip": one(hip, B._PROBE["hip"])}


class PluginLoader(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.site = Path(self.tmp.name) / "site-packages"
        self.pkg = self.site / "mojolearn"
        self.pkg.mkdir(parents=True)
        self.B = fresh_backend()
        self.B._pkg_dir = lambda: str(self.pkg)
        self.version = self.B._CORE_VERSION
        self.env = {k: os.environ.pop(k) for k in ("MOJOLEARN_VENDOR", "MOJOLEARN_GPU_ARCH",
                                                    "MOJOLEARN_VENDOR_FORCE", "MOJOLEARN_CUDA_PATH",
                                                    "MOJOLEARN_EXPERIMENTAL_PTX") if k in os.environ}
        self.addCleanup(os.environ.update, self.env)
        for k in ("MOJOLEARN_VENDOR", "MOJOLEARN_GPU_ARCH", "MOJOLEARN_CUDA_PATH", "MOJOLEARN_EXPERIMENTAL_PTX"):
            self.addCleanup(os.environ.pop, k, None)

    # ---- fixture builders -------------------------------------------------
    def dist_info(self, name, version, marker=None):
        d = self.site / f"{name.replace('-', '_')}-{version}.dist-info"
        d.mkdir(parents=True, exist_ok=True)
        (d / "METADATA").write_text(f"Metadata-Version: 2.4\nName: {name}\nVersion: {version}\n")
        if marker is not None:
            (d / marker[0]).write_text(marker[1])
        return d

    def split_core(self):
        self.B._SPLIT = None
        import json
        gp = self.B.gpu_plugins
        self.dist_info("mojolearn", self.version,
                       (gp.CORE_MARKER, json.dumps(gp.core_marker(self.version))))

    def combined_core(self):
        self.dist_info("mojolearn", self.version)

    def sets(self, vendor, arches):
        for arch in arches:
            for tier in ("", "identical"):
                directory = self.B.gpu_plugins.native_directory(vendor) if self.B._split_core() else vendor
                d = self.pkg / directory / arch / tier
                d.mkdir(parents=True, exist_ok=True)
                (d / "_mojolearn_gbdt.so").write_bytes(b"inert")

    def ptx_set(self, source="a" * 40):
        """The PTX slot (Andrew 2026-10-10: PTX is a normal target; no flag):
        cuda_ptx/sm_80 with a fast and an identical binding, its v2 manifest
        and the core's source witness. Returns (root, identical binary)."""
        import hashlib
        gp = self.B.gpu_plugins
        root = self.pkg / gp.PTX_DIRECTORY / gp.PTX_ARCH
        (root / "identical").mkdir(parents=True, exist_ok=True)
        rows = []
        for rel, mode in (("_mojolearn_gbdt.so", "fast"), ("identical/_mojolearn_gbdt.so", "identical")):
            (root / rel).write_bytes(b"fake embedded PTX, not executable " + mode.encode())
            rows.append(dict(file=rel, numeric_mode=mode, sha256=hashlib.sha256((root / rel).read_bytes()).hexdigest(),
                             ptx_modules=[dict(target="sm_80", sha256="b" * 64)]))
        doc = dict(schema=gp.PTX_MANIFEST_SCHEMA, code_format=gp.PTX_CODE_FORMAT, vendor="cuda",
                   target="sm_80", min_compute_capability=[8, 0], source_commit=source,
                   source_dirty=False, mojo_version="fixture", errors=[], files=rows)
        (root / gp.BASELINE_MANIFEST).write_text(json.dumps(doc))
        (self.pkg / "identity_columns").mkdir(exist_ok=True)
        (self.pkg / "identity_columns" / "COMMIT").write_text(source)
        return root, root / "identical" / "_mojolearn_gbdt.so"

    def plugin(self, vendor, arches, version=None):
        import hashlib
        gp = self.B.gpu_plugins
        version = version or self.version
        if vendor == "cuda" and "sm_89" not in arches:
            arches = ["sm_89"] + list(arches)
        bundle = None
        if vendor == "cuda":
            root, _ = self.ptx_set()
            bundle = dict(manifest_sha256=hashlib.sha256((root / gp.BASELINE_MANIFEST).read_bytes()).hexdigest())
        self.dist_info(gp.plugin(vendor)["distribution"], version,
                       (gp.PLUGIN_MARKER, json.dumps(gp.plugin_marker(vendor, version, arches, bundled_ptx=bundle))))
        self.sets(vendor, arches)

    def host_binding(self):
        (self.pkg / "host").mkdir(exist_ok=True)
        (self.pkg / "host" / "_mojolearn_forest_host.so").write_bytes(b"inert")

    def box(self, cuda=False, hip=False, device=("sm_90", "fixture")):
        """Point the selector at the fabricated install and box, uncached."""
        B = self.B
        B._LAYOUT = B._VENDOR_SELECTED = B._VENDOR_HOW = None
        B._ARCH_SELECTED = B._ARCH_HOW = None
        B._SPLIT = None
        B._pkg_dir = lambda: str(self.pkg)
        B._probe_box = lambda: probe(B, cuda, hip)
        B._device_arch = lambda vendor: device if vendor == "cuda" else ("gfx942", "fixture")

    def layout(self, **box):
        self.box(**box)
        return self.B._layout()

    def refusal(self, **box):
        with self.assertRaises(ImportError) as ctx:
            self.layout(**box)
        return ctx.exception

    # ---- the split core ---------------------------------------------------
    def test_gpu_without_its_plugin_refuses_by_name(self):
        self.split_core()
        exc = self.refusal(cuda=True)
        self.assertIsInstance(exc, self.B.GpuPluginError)
        # the core requires both plugins, so a missing one is a broken
        # install: the message says so and names the core reinstall, never a
        # separate `pip install mojolearn-nvidia`
        self.assertIn("mojolearn-nvidia", str(exc))
        self.assertIn("incomplete", str(exc))
        self.assertIn(f'pip install --force-reinstall "mojolearn=={self.version}"', str(exc))
        self.assertNotIn('pip install "mojolearn-nvidia', str(exc))
        self.assertIn("MOJOLEARN_VENDOR=cpu", str(exc))
        # twin: the same box with the plugin installed loads it
        self.plugin("cuda", ["sm_90a"])
        kind, base = self.layout(cuda=True)
        self.assertEqual((kind, base), ("vendor", str(self.pkg / "cuda_native" / "sm_90a")))
        self.assertEqual(self.B._PLUGINS_FOUND["cuda"]["distribution"], "mojolearn-nvidia")
        self.assertEqual(self.B._PLUGINS_FOUND["cuda"]["version"], self.version)

    def test_amd_gpu_names_the_amd_plugin(self):
        self.split_core()
        exc = self.refusal(hip=True)
        self.assertIsInstance(exc, self.B.GpuPluginError)
        self.assertIn("mojolearn-amd, the package", str(exc))
        self.assertIn(f'pip install --force-reinstall "mojolearn=={self.version}"', str(exc))
        self.assertNotIn("mojolearn-nvidia", str(exc))
        self.plugin("hip", ["gfx942"])
        self.assertEqual(self.layout(hip=True), ("vendor", str(self.pkg / "hip_native" / "gfx942")))

    def test_no_gpu_no_plugin_is_the_cpu_only_install(self):
        self.split_core()
        exc = self.refusal()
        # a PLAIN ImportError: select() turns it into the CPU-only set
        self.assertNotIsInstance(exc, self.B.GpuPluginError)
        self.assertIn("no supported GPU found", str(exc))

    def test_the_other_vendors_plugin_does_not_answer_for_this_gpu(self):
        self.split_core()
        self.plugin("hip", ["gfx942"])
        exc = self.refusal(cuda=True)
        self.assertIsInstance(exc, self.B.GpuPluginError)
        self.assertIn("mojolearn-nvidia, the package", str(exc))
        self.assertIn('pip install --force-reinstall "mojolearn==', str(exc))
        # twin: the AMD box takes the AMD plugin
        self.assertEqual(self.layout(hip=True)[0], "vendor")

    def test_both_plugins_pick_by_the_probe(self):
        self.split_core()
        self.plugin("cuda", ["sm_89", "sm_90a"])
        self.plugin("hip", ["gfx942"])
        self.assertEqual(self.layout(cuda=True), ("vendor", str(self.pkg / "cuda_native" / "sm_90a")))
        self.assertEqual(self.layout(cuda=True, device=("sm_89", "fixture")),
                         ("vendor", str(self.pkg / "cuda_native" / "sm_89")))
        self.assertEqual(self.layout(hip=True), ("vendor", str(self.pkg / "hip_native" / "gfx942")))
        exc = self.refusal(cuda=True, hip=True)   # today's refusal, unchanged
        self.assertNotIsInstance(exc, self.B.GpuPluginError)
        self.assertIn("MORE THAN ONE", str(exc))

    def test_a_plugin_of_another_version_refuses(self):
        self.split_core()
        self.plugin("cuda", ["sm_90a"], version="0.0.1")
        exc = self.refusal(cuda=True)
        self.assertIsInstance(exc, self.B.GpuPluginError)
        self.assertIn("mojolearn-nvidia 0.0.1", str(exc))
        self.assertIn(f'pip install --force-reinstall "mojolearn=={self.version}"', str(exc))
        # even on a box with no GPU: a mismatched plugin is never loaded or ignored
        self.assertIsInstance(self.refusal(), self.B.GpuPluginError)

    def test_sets_no_plugin_owns_refuse(self):
        self.split_core()
        self.sets("cuda", ["sm_90a"])
        exc = self.refusal(cuda=True)
        self.assertIsInstance(exc, self.B.GpuPluginError)
        self.assertIn("no mojolearn-nvidia is installed", str(exc))

    def test_a_plugin_without_its_files_refuses(self):
        self.split_core()
        self.dist_info("mojolearn-nvidia", self.version)
        exc = self.refusal(cuda=True)
        self.assertIsInstance(exc, self.B.GpuPluginError)
        self.assertIn(f'pip install --force-reinstall "mojolearn=={self.version}"', str(exc))

    def test_cpu_on_request_is_not_a_plugin_refusal(self):
        self.split_core()
        os.environ["MOJOLEARN_VENDOR"] = "cpu"
        exc = self.refusal(cuda=True)
        self.assertNotIsInstance(exc, self.B.GpuPluginError)
        self.assertIn("MOJOLEARN_VENDOR=cpu", str(exc))

    def test_a_forced_vendor_without_its_plugin_refuses(self):
        self.split_core()
        self.plugin("hip", ["gfx942"])
        os.environ["MOJOLEARN_VENDOR"] = "cuda"
        exc = self.refusal(cuda=True, hip=True)
        self.assertIsInstance(exc, self.B.GpuPluginError)
        self.assertIn("mojolearn-nvidia, the package", str(exc))
        self.assertIn('pip install --force-reinstall "mojolearn==', str(exc))
        os.environ["MOJOLEARN_VENDOR"] = "hip"
        self.assertEqual(self.layout(cuda=True, hip=True)[0], "vendor")

    def test_a_plugin_in_another_environment_is_named(self):
        self.split_core()
        other = Path(self.tmp.name) / "elsewhere"
        d = other / f"mojolearn_nvidia-{self.version}.dist-info"
        d.mkdir(parents=True)
        (d / "METADATA").write_text(f"Metadata-Version: 2.4\nName: mojolearn-nvidia\nVersion: {self.version}\n")
        sys.path.insert(0, str(other))
        self.addCleanup(sys.path.remove, str(other))
        exc = self.refusal(cuda=True)
        self.assertIsInstance(exc, self.B.GpuPluginError)
        self.assertIn("same environment", str(exc))

    # ---- select(): the plugin refusal is never the CPU-only set -------------
    def test_select_keeps_the_plugin_refusal_and_the_cpu_only_set_apart(self):
        self.split_core()
        self.host_binding()
        B = self.B
        self.box(cuda=True)
        with self.assertRaises(B.GpuPluginError):
            B.select()
        self.assertIsNone(B._CPU_ONLY)
        # twin: no GPU on the box -> the CPU-only set, as today
        self.box()
        B.select()
        self.assertIsNotNone(B._CPU_ONLY)
        self.assertEqual(B.vendor(), "cpu")
        self.assertIsNone(B.gpu_plugin())

    # ---- installs that are not the split core: unchanged ------------------
    def test_the_combined_wheel_is_unchanged(self):
        self.combined_core()                 # no gpu_plugins.json marker
        self.sets("cuda", ["sm_90a"])        # sets no plugin owns: fine here
        self.sets("hip", ["gfx942"])
        self.assertEqual(self.layout(cuda=True), ("vendor", str(self.pkg / "cuda" / "sm_90a")))
        self.assertEqual(self.B._PLUGINS_FOUND, {})

    def test_a_checkout_with_no_sets_stays_flat(self):
        # no .dist-info at all: a source checkout; a GPU box is not refused
        self.assertEqual(self.layout(cuda=True), ("flat", str(self.pkg)))
        self.combined_core()
        self.assertEqual(self.layout(cuda=True), ("flat", str(self.pkg)))
        # twin: the split core on the same box refuses
        self.split_core()
        self.assertIsInstance(self.refusal(cuda=True), self.B.GpuPluginError)

    def test_cpu_on_request_on_the_flat_layout(self):
        # The macOS wheel and a source checkout are flat. MOJOLEARN_VENDOR=cpu
        # was read only on the vendor layout until 2026-09-26, so on a Mac it
        # was accepted and ignored and the Metal set answered.
        self.host_binding()
        B = self.B
        os.environ["MOJOLEARN_VENDOR"] = "cpu"
        exc = self.refusal()
        self.assertNotIsInstance(exc, B.GpuPluginError)
        self.assertIn("MOJOLEARN_VENDOR=cpu", str(exc))
        self.box()
        B.select()
        self.assertIsNotNone(B._CPU_ONLY)
        self.assertEqual(B.vendor(), "cpu")
        # twin: the same flat install without the variable stays flat
        os.environ.pop("MOJOLEARN_VENDOR")
        self.assertEqual(self.layout(), ("flat", str(self.pkg)))

    def test_a_malformed_marker_refuses(self):
        self.dist_info("mojolearn", self.version, (self.B.gpu_plugins.CORE_MARKER, "{}"))
        self.assertIsInstance(self.refusal(), self.B.GpuPluginError)


class ArchitecturePayloadLoader(PluginLoader):
    """Inheritance re-runs the existing native compatibility cases alongside new refusals."""

    def test_missing_native_architecture_refuses_vendor_bundle(self):
        self.split_core()
        self.plugin("cuda", ["sm_89", "sm_90a"])
        import shutil
        shutil.rmtree(self.pkg / "cuda_native/sm_89")
        self.assertIn("missing or unexpected", str(self.refusal(cuda=True)))

    def test_payload_marker_cannot_claim_another_architecture(self):
        self.split_core()
        self.plugin("cuda", ["sm_89", "sm_90a"])
        path = self.site / f"mojolearn_nvidia-{self.version}.dist-info" / self.B.gpu_plugins.PLUGIN_MARKER
        doc = json.loads(path.read_text())
        doc["arches"] = ["sm_90a"]
        path.write_text(json.dumps(doc))
        self.assertIn("invalid gpu_plugin.json", str(self.refusal(cuda=True)))

    def test_legacy_roots_cannot_override_new_payloads(self):
        self.split_core()
        self.plugin("cuda", ["sm_89", "sm_90a"])
        legacy = self.pkg / "cuda" / "sm_90a"
        legacy.mkdir(parents=True)
        (legacy / "_mojolearn_gbdt.so").write_bytes(b"old wheel")
        self.assertIn("legacy cuda", str(self.refusal(cuda=True)))

    # ---- THE PTX SLOT (Andrew 2026-10-10: PTX is a normal target; no flag) ----
    def ptx_root(self):
        return self.pkg / "cuda_ptx" / "sm_80"

    def test_native_set_first_then_ptx_for_every_other_nvidia_gpu(self):
        self.split_core()
        self.plugin("cuda", ["sm_89"])
        native = ("vendor", str(self.pkg / "cuda_native" / "sm_89"))
        self.assertEqual(self.layout(cuda=True, device=("sm_89", "fixture")), native)
        self.assertIsNone(self.B.baseline_selection_receipt())
        for device in ("sm_80", "sm_86", "sm_90", "sm_100", "sm_120", "sm_121"):
            with self.subTest(device=device):
                self.B._BASELINE_SELECTION = None
                self.assertEqual(self.layout(cuda=True, device=(device, "fixture")), ("vendor", str(self.ptx_root())))
                self.assertEqual(self.B.gpu_arch(), "sm_80")
                receipt = self.B.baseline_selection_receipt()
                self.assertEqual((receipt["selected"], receipt["code_format"]), ("ptx", "ptx"))
                self.assertEqual(self.B.gpu_plugin()["code_format"], "ptx")

    def test_ptx_serves_identical_with_no_flag_and_no_admission(self):
        self.split_core()
        self.plugin("cuda", ["sm_89"])
        self.B.requested_mode = lambda: "identical"
        self.assertEqual(self.layout(cuda=True, device=("sm_90", "fixture")), ("vendor", str(self.ptx_root())))
        receipt = self.B.baseline_selection_receipt()
        self.assertEqual(receipt["numeric_mode"], "identical")
        for removed in ("identical_qualified", "admission", "qualifying"):
            self.assertNotIn(removed, receipt)
        for name in ("PTX_QUALIFY_COMMAND", "_ptx_qualifying", "_ptx_identical_refusal", "_baseline_layout",
                     "_admitted_baseline_base"):
            self.assertFalse(hasattr(self.B, name), name)

    def test_gpu_arch_names_the_ptx_set_like_any_directory(self):
        self.split_core()
        self.plugin("cuda", ["sm_89"])
        os.environ["MOJOLEARN_GPU_ARCH"] = "sm_80"
        self.assertEqual(self.layout(cuda=True, device=("sm_89", "fixture")), ("vendor", str(self.ptx_root())))
        self.assertIn("MOJOLEARN_GPU_ARCH", self.B.gpu_arch_how())
        os.environ["MOJOLEARN_GPU_ARCH"] = "sm_89"
        self.B._BASELINE_SELECTION = None
        self.assertEqual(self.layout(cuda=True, device=("sm_89", "fixture")),
                         ("vendor", str(self.pkg / "cuda_native" / "sm_89")))

    def test_the_old_experimental_switches_do_nothing(self):
        self.split_core()
        self.plugin("cuda", ["sm_89"])
        os.environ["MOJOLEARN_CUDA_PATH"] = "ptx-baseline"
        os.environ["MOJOLEARN_EXPERIMENTAL_PTX"] = "1"
        self.assertEqual(self.layout(cuda=True, device=("sm_89", "fixture")),
                         ("vendor", str(self.pkg / "cuda_native" / "sm_89")))
        self.assertIsNone(self.B.baseline_selection_receipt())

    def test_ptx_load_receipt_records_actual_loads(self):
        self.split_core()
        self.plugin("cuda", ["sm_89"])
        self.layout(cuda=True, device=("sm_100", "fixture"))
        binary = self.ptx_root() / "identical" / "_mojolearn_gbdt.so"
        self.assertEqual(self.B.baseline_selection_receipt()["loaded_files"], [])
        class Failing:
            path = str(binary)
            def exec_module(self, module):
                raise RuntimeError("failed load")
        with self.assertRaises(RuntimeError):
            self.B._exec_binding(Failing(), object())
        self.assertEqual(self.B.baseline_selection_receipt()["loaded_files"], [])
        class Loader:
            path = str(binary)
            def exec_module(self, module):
                pass
        self.B._exec_binding(Loader(), object())
        self.assertEqual(self.B.baseline_selection_receipt()["loaded_files"][0]["file"],
                         "identical/_mojolearn_gbdt.so")

    def test_ptx_tamper_source_marker_and_capability_refuse(self):
        self.split_core()
        self.plugin("cuda", ["sm_89"])
        binary = self.ptx_root() / "identical" / "_mojolearn_gbdt.so"
        original = binary.read_bytes()
        binary.write_bytes(b"tampered")
        self.assertIn("complete GPU binding bytes", str(self.refusal(cuda=True)))
        binary.write_bytes(original)
        (self.pkg / "identity_columns" / "COMMIT").write_text("c" * 40)
        self.assertIn("source differs", str(self.refusal(cuda=True)))
        (self.pkg / "identity_columns" / "COMMIT").write_text("a" * 40)
        exc = self.refusal(cuda=True, device=("sm_75", "fixture"))
        self.assertIsInstance(exc, self.B.GpuPluginError)
        self.assertIn("older than compute capability 8.0", str(exc))
        manifest = self.ptx_root() / self.B.gpu_plugins.BASELINE_MANIFEST
        manifest.write_text(manifest.read_text() + " ")
        self.assertIn("vendor marker", str(self.refusal(cuda=True)))

    def test_the_nvidia_wheel_must_carry_and_bind_its_ptx_set(self):
        import shutil
        self.split_core()
        self.plugin("cuda", ["sm_89"])
        shutil.rmtree(self.ptx_root())
        exc = self.refusal(cuda=True, device=("sm_89", "fixture"))
        self.assertIsInstance(exc, self.B.GpuPluginError)
        self.assertIn("missing its PTX set", str(exc))
        # twin: the marker no longer binds a PTX manifest
        self.ptx_set()
        path = self.site / f"mojolearn_nvidia-{self.version}.dist-info" / self.B.gpu_plugins.PLUGIN_MARKER
        doc = json.loads(path.read_text())
        del doc["bundled_ptx"]
        path.write_text(json.dumps(doc))
        self.assertIn("PTX slot metadata", str(self.refusal(cuda=True, device=("sm_89", "fixture"))))

    def test_malformed_ptx_records_never_select_cpu(self):
        self.split_core()
        self.plugin("cuda", ["sm_89"])
        self.host_binding()
        path = self.ptx_root() / self.B.gpu_plugins.BASELINE_MANIFEST
        original = json.loads(path.read_text())
        for files in ([None], ["wrong"], [dict(original["files"][0], file={})],
                      [dict(original["files"][0], ptx_modules="wrong")]):
            with self.subTest(files=files):
                path.write_text(json.dumps(dict(original, files=files)))
                self.box(cuda=True)
                with self.assertRaises(self.B.GpuPluginError):
                    self.B.select()
                self.assertIsNone(self.B._CPU_ONLY)

    def test_missing_ptx_tier_never_selects_cpu(self):
        import hashlib
        self.split_core()
        self.plugin("cuda", ["sm_89"])
        self.host_binding()
        root = self.ptx_root()
        (root / "identical" / "_mojolearn_gbdt.so").unlink()
        gp = self.B.gpu_plugins
        path = root / gp.BASELINE_MANIFEST
        doc = json.loads(path.read_text())
        doc["files"] = [row for row in doc["files"] if row["numeric_mode"] == "fast"]
        path.write_text(json.dumps(doc))
        marker = self.site / f"mojolearn_nvidia-{self.version}.dist-info" / gp.PLUGIN_MARKER
        mdoc = json.loads(marker.read_text())
        mdoc["bundled_ptx"] = dict(manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        marker.write_text(json.dumps(mdoc))
        self.box(cuda=True)
        self.B.requested_mode = lambda: "identical"
        self.B._MODULES = ("_mojolearn_gbdt",)
        with self.assertRaisesRegex(self.B.GpuPluginError, "never falls back to the CPU"):
            self.B.select()
        self.assertIsNone(self.B._CPU_ONLY)

    def test_blackwell_native_sets_are_registered(self):
        gp = self.B.gpu_plugins
        for arch in ("sm_100", "sm_100a", "sm_103", "sm_103a", "sm_120", "sm_120a", "sm_121", "sm_121a"):
            self.assertIn(arch, gp.PLUGINS["cuda"]["arches"])
        self.split_core()
        self.plugin("cuda", ["sm_89", "sm_100a"])
        self.assertEqual(self.layout(cuda=True, device=("sm_100", "fixture")),
                         ("vendor", str(self.pkg / "cuda_native" / "sm_100a")))
        self.assertIsNone(self.B.baseline_selection_receipt())


if __name__ == "__main__":
    unittest.main()
