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
        self.version = self.B._CORE_VERSION
        self.env = {k: os.environ.pop(k) for k in ("MOJOLEARN_VENDOR", "MOJOLEARN_GPU_ARCH",
                                                    "MOJOLEARN_VENDOR_FORCE") if k in os.environ}
        self.addCleanup(os.environ.update, self.env)
        for k in ("MOJOLEARN_VENDOR",):
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
        import json
        gp = self.B.gpu_plugins
        self.dist_info("mojolearn", self.version,
                       (gp.CORE_MARKER, json.dumps(gp.core_marker(self.version))))

    def combined_core(self):
        self.dist_info("mojolearn", self.version)

    def sets(self, vendor, arches):
        for arch in arches:
            for tier in ("", "identical"):
                d = self.pkg / vendor / arch / tier
                d.mkdir(parents=True, exist_ok=True)
                (d / "_mojolearn_gbdt.so").write_bytes(b"inert")

    def plugin(self, vendor, arches, version=None):
        row = self.B.gpu_plugins.plugin(vendor)
        self.dist_info(row["distribution"], version or self.version)
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
        self.assertIn(f'pip install "mojolearn[cuda]=={self.version}"', str(exc))
        self.assertIn("MOJOLEARN_VENDOR=cpu", str(exc))
        # twin: the same box with the plugin installed loads it
        self.plugin("cuda", ["sm_90a"])
        kind, base = self.layout(cuda=True)
        self.assertEqual((kind, base), ("vendor", str(self.pkg / "cuda" / "sm_90a")))
        self.assertEqual(self.B._PLUGINS_FOUND["cuda"]["distribution"], "mojolearn-cuda")
        self.assertEqual(self.B._PLUGINS_FOUND["cuda"]["version"], self.version)

    def test_amd_gpu_names_the_rocm_extra(self):
        self.split_core()
        exc = self.refusal(hip=True)
        self.assertIsInstance(exc, self.B.GpuPluginError)
        self.assertIn(f'pip install "mojolearn[rocm]=={self.version}"', str(exc))
        self.assertNotIn("mojolearn[cuda]", str(exc))
        self.plugin("hip", ["gfx942"])
        self.assertEqual(self.layout(hip=True), ("vendor", str(self.pkg / "hip" / "gfx942")))

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
        self.assertIn("mojolearn[cuda]", str(exc))
        # twin: the AMD box takes the AMD plugin
        self.assertEqual(self.layout(hip=True)[0], "vendor")

    def test_both_plugins_pick_by_the_probe(self):
        self.split_core()
        self.plugin("cuda", ["sm_89", "sm_90a"])
        self.plugin("hip", ["gfx942"])
        self.assertEqual(self.layout(cuda=True), ("vendor", str(self.pkg / "cuda" / "sm_90a")))
        self.assertEqual(self.layout(cuda=True, device=("sm_89", "fixture")),
                         ("vendor", str(self.pkg / "cuda" / "sm_89")))
        self.assertEqual(self.layout(hip=True), ("vendor", str(self.pkg / "hip" / "gfx942")))
        exc = self.refusal(cuda=True, hip=True)   # today's refusal, unchanged
        self.assertNotIsInstance(exc, self.B.GpuPluginError)
        self.assertIn("MORE THAN ONE", str(exc))

    def test_a_plugin_of_another_version_refuses(self):
        self.split_core()
        self.plugin("cuda", ["sm_90a"], version="0.0.1")
        exc = self.refusal(cuda=True)
        self.assertIsInstance(exc, self.B.GpuPluginError)
        self.assertIn("mojolearn-cuda 0.0.1", str(exc))
        self.assertIn(f"mojolearn[cuda]=={self.version}", str(exc))
        # even on a box with no GPU: a mismatched plugin is never loaded or ignored
        self.assertIsInstance(self.refusal(), self.B.GpuPluginError)

    def test_sets_no_plugin_owns_refuse(self):
        self.split_core()
        self.sets("cuda", ["sm_90a"])
        exc = self.refusal(cuda=True)
        self.assertIsInstance(exc, self.B.GpuPluginError)
        self.assertIn("no mojolearn-cuda is installed", str(exc))

    def test_a_plugin_without_its_files_refuses(self):
        self.split_core()
        self.dist_info("mojolearn-cuda", self.version)
        exc = self.refusal(cuda=True)
        self.assertIsInstance(exc, self.B.GpuPluginError)
        self.assertIn("--force-reinstall", str(exc))

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
        self.assertIn("mojolearn[cuda]", str(exc))
        os.environ["MOJOLEARN_VENDOR"] = "hip"
        self.assertEqual(self.layout(cuda=True, hip=True)[0], "vendor")

    def test_a_plugin_in_another_environment_is_named(self):
        self.split_core()
        other = Path(self.tmp.name) / "elsewhere"
        d = other / f"mojolearn_cuda-{self.version}.dist-info"
        d.mkdir(parents=True)
        (d / "METADATA").write_text(f"Metadata-Version: 2.4\nName: mojolearn-cuda\nVersion: {self.version}\n")
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

    def test_a_malformed_marker_refuses(self):
        self.dist_info("mojolearn", self.version, (self.B.gpu_plugins.CORE_MARKER, "{}"))
        self.assertIsInstance(self.refusal(), self.B.GpuPluginError)


if __name__ == "__main__":
    unittest.main()
