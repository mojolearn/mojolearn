# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""The NVIDIA PTX slot as the loader decides it, on a machine with no GPU.

    python python/mojolearn/tests/test_ptx_fallback.py

The package is loaded under a private name with its __init__ skipped, so this
file needs no built binding. In a checkout without bindings, pytest must be
pointed at the file from outside the package (a symlink in another directory),
because collecting it as `mojolearn.tests.*` imports the real package.

Every case fabricates an installed split core and a `mojolearn-nvidia` plugin
carrying the native sm_89 set and the PTX slot (cuda_ptx/sm_80), then asks the
real `_backend` about a synthetic device. Nothing loads an extension and
nothing touches a device. The rules under test (docs/NVIDIA_PTX_IDENTITY.md;
Andrew 2026-10-10: PTX is a normal target; no flag):

  * native first: a device with a native set loads it;
  * every other NVIDIA device of compute capability 8.0 or newer loads the PTX
    set, in FAST, DETERMINISTIC and IDENTICAL alike, with no admission;
  * MOJOLEARN_GPU_ARCH=sm_80 selects the PTX set by name;
  * an older device, or a PTX set that fails its provenance, refuses and never
    selects the CPU set;
  * MOJOLEARN_EXPERIMENTAL_PTX and MOJOLEARN_CUDA_PATH no longer mean anything.
"""
import hashlib
import importlib
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

PKG_SRC = Path(__file__).resolve().parents[1]
ALIAS = "mojolearn_ptx_fallback_under_test"
SOURCE = "a" * 40
MODES = ("fast", "deterministic", "identical")


def fresh(module="_backend"):
    for name in [n for n in sys.modules if n == ALIAS or n.startswith(ALIAS + ".")]:
        del sys.modules[name]
    pkg = types.ModuleType(ALIAS)
    pkg.__path__ = [str(PKG_SRC)]
    sys.modules[ALIAS] = pkg
    return importlib.import_module(ALIAS + "." + module)


def probe(B, cuda=True):
    def one(found, spec):
        return {"paths": {p: found for p in spec["paths"]},
                "libs": {lib: False for lib in spec["libs"]}, "found": found}
    return {"cuda": one(cuda, B._PROBE["cuda"]), "hip": one(False, B._PROBE["hip"])}


class Install(unittest.TestCase):
    """A fabricated install: split core, NVIDIA plugin with sm_89 and the PTX slot."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.site = Path(self.tmp.name) / "site-packages"
        self.pkg = self.site / "mojolearn"
        self.pkg.mkdir(parents=True)
        self.B = fresh()
        self.version = self.B._CORE_VERSION
        env = patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        for key in ("MOJOLEARN_VENDOR", "MOJOLEARN_GPU_ARCH", "MOJOLEARN_VENDOR_FORCE", "MOJOLEARN_CUDA_PATH",
                    "MOJOLEARN_EXPERIMENTAL_PTX", "MOJOLEARN_NUMERIC_MODE"):
            os.environ.pop(key, None)
        self.device = "sm_80"
        self.build()

    def dist_info(self, name, marker=None):
        d = self.site / f"{name.replace('-', '_')}-{self.version}.dist-info"
        d.mkdir(parents=True, exist_ok=True)
        (d / "METADATA").write_text(f"Metadata-Version: 2.4\nName: {name}\nVersion: {self.version}\n")
        if marker is not None:
            (d / marker[0]).write_text(marker[1])

    def marker_path(self):
        gp = self.B.gpu_plugins
        return self.site / f"mojolearn_nvidia-{self.version}.dist-info" / gp.PLUGIN_MARKER

    def build(self, dirty=False):
        B, gp = self.B, self.B.gpu_plugins
        self.dist_info("mojolearn", (gp.CORE_MARKER, json.dumps(gp.core_marker(self.version))))
        self.arches = ["sm_89"]
        for arch in self.arches:
            for tier in ("", "deterministic", "identical"):
                d = self.pkg / gp.native_directory("cuda") / arch / tier
                d.mkdir(parents=True, exist_ok=True)
                (d / "_mojolearn_gbdt.so").write_bytes(b"inert native")
        self.root = self.pkg / gp.PTX_DIRECTORY / gp.PTX_ARCH
        files = []
        for tier in MODES:
            d = self.root if tier == "fast" else self.root / tier
            d.mkdir(parents=True, exist_ok=True)
            binary = d / "_mojolearn_gbdt.so"
            binary.write_bytes(b"fake embedded PTX " + tier.encode())
            files.append(dict(file=binary.relative_to(self.root).as_posix(), numeric_mode=tier,
                              sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                              ptx_modules=[dict(target="sm_80", sha256="b" * 64)]))
        manifest = dict(schema=gp.PTX_MANIFEST_SCHEMA, code_format=gp.PTX_CODE_FORMAT, vendor="cuda",
                        target=gp.PTX_ARCH, min_compute_capability=[8, 0], source_commit=SOURCE,
                        source_dirty=dirty, mojo_version="test", errors=[], files=files)
        (self.root / gp.BASELINE_MANIFEST).write_text(json.dumps(manifest))
        self.manifest_hash = hashlib.sha256((self.root / gp.BASELINE_MANIFEST).read_bytes()).hexdigest()
        self.write_marker(dict(manifest_sha256=self.manifest_hash))
        (self.pkg / "identity_columns").mkdir(exist_ok=True)
        (self.pkg / "identity_columns" / "COMMIT").write_text(SOURCE)
        B._pkg_dir = lambda: str(self.pkg)
        B._probe_box = lambda: probe(B)
        B._device_arch = lambda vendor: (self.device, "synthetic driver")
        self.reset()

    def write_marker(self, bundle):
        gp = self.B.gpu_plugins
        doc = gp.plugin_marker("cuda", self.version, self.arches, bundled_ptx=bundle)
        self.dist_info(gp.plugin("cuda")["distribution"], (gp.PLUGIN_MARKER, json.dumps(doc)))

    def reset(self):
        B = self.B
        B._LAYOUT = B._VENDOR_SELECTED = B._VENDOR_HOW = B._ARCH_SELECTED = B._ARCH_HOW = None
        B._SPLIT = B._BASELINE_SELECTION = B._BASELINE_ROOT = None
        B._BASELINE_FILES = {}
        B._SELECTED = B._CPU_ONLY = B._DEFAULT_MODE = None
        B._SETS.clear()

    def layout(self, mode, **env):
        self.reset()
        with patch.dict(os.environ, {"MOJOLEARN_NUMERIC_MODE": mode, **env}):
            return self.B._layout()

    def refusal(self, mode, **env):
        """The layout's refusal, which must be a GPU refusal `select()` never turns into the CPU set."""
        self.reset()
        with patch.dict(os.environ, {"MOJOLEARN_NUMERIC_MODE": mode, **env}):
            with patch.object(self.B, "host_binding_built", return_value=True):
                with patch.object(self.B, "_select_cpu_only") as cpu:
                    with self.assertRaises(self.B.GpuPluginError) as ctx:
                        self.B.select()
                    cpu.assert_not_called()
        self.assertIsNone(self.B._CPU_ONLY)
        self.assertIsNone(self.B.baseline_selection_receipt())
        return str(ctx.exception)


class Selection(Install):
    def test_a_device_with_a_native_set_loads_it(self):
        self.device = "sm_89"
        for mode in MODES:
            with self.subTest(mode=mode):
                self.assertTrue(self.layout(mode)[1].endswith("cuda_native/sm_89"))
                self.assertIsNone(self.B.baseline_selection_receipt())
                self.assertEqual(self.B.gpu_arch(), "sm_89")

    def test_every_other_nvidia_device_takes_ptx_in_every_mode(self):
        # sm_86 included: the PTX slot is not a native family member of 8.x.
        for device in ("sm_80", "sm_86", "sm_90", "sm_100", "sm_120"):
            self.device = device
            for mode in MODES:
                with self.subTest(device=device, mode=mode):
                    self.assertEqual(self.layout(mode), ("vendor", str(self.root)))
                    receipt = self.B.baseline_selection_receipt()
                    self.assertEqual((receipt["schema"], receipt["selected"], receipt["fallback"],
                                      receipt["code_format"], receipt["arch"], receipt["numeric_mode"]),
                                     ("mojolearn.ptx-selection.v2", "ptx", "ptx", "ptx", "sm_80", mode))
                    self.assertEqual((receipt["manifest_sha256"], receipt["source_commit"]),
                                     (self.manifest_hash, SOURCE))
                    self.assertEqual(receipt["loaded_files"], [])
                    for gone in ("identical_qualified", "admission", "qualifying", "requested"):
                        self.assertNotIn(gone, receipt)
                    self.assertEqual(self.B.gpu_arch(), "sm_80")
                    self.assertIn(device, self.B.gpu_arch_how())
                    self.assertEqual(self.B.tier_dir(mode),
                                     str(self.root if mode == "fast" else self.root / mode))
                    self.assertEqual(self.B.gpu_plugin()["code_format"], "ptx")

    def test_a_device_older_than_8_0_refuses_and_never_selects_cpu(self):
        self.device = "sm_75"
        for mode in MODES:
            with self.subTest(mode=mode):
                self.assertIn("older than compute capability 8.0", self.refusal(mode))

    def test_gpu_arch_sm_80_selects_ptx_on_a_native_device(self):
        self.device = "sm_89"
        for mode in MODES:
            with self.subTest(mode=mode):
                self.assertEqual(self.layout(mode, MOJOLEARN_GPU_ARCH="sm_80"), ("vendor", str(self.root)))
                receipt = self.B.baseline_selection_receipt()
                self.assertEqual(receipt["selected"], "ptx")
                self.assertIn("MOJOLEARN_GPU_ARCH", receipt["how"])

    def test_the_removed_switches_have_no_effect(self):
        for env in ({"MOJOLEARN_EXPERIMENTAL_PTX": "1"}, {"MOJOLEARN_CUDA_PATH": "ptx-baseline"},
                    {"MOJOLEARN_EXPERIMENTAL_PTX": "1", "MOJOLEARN_CUDA_PATH": "ptx-baseline"}):
            with self.subTest(env=env):
                self.device = "sm_89"
                self.assertTrue(self.layout("identical", **env)[1].endswith("cuda_native/sm_89"))
                self.assertIsNone(self.B.baseline_selection_receipt())
                self.device = "sm_90"
                self.assertEqual(self.layout("identical", **env), ("vendor", str(self.root)))
        for gone in ("_baseline_layout", "_admitted_baseline_base", "_ptx_identical_refusal",
                     "_ptx_qualifying", "PTX_QUALIFY_COMMAND", "_ptx_runtime_configuration"):
            self.assertFalse(hasattr(self.B, gone), gone)

    def test_per_call_identical_is_served_on_a_ptx_process(self):
        self.device = "sm_90"
        self.layout("fast")
        marker = object()
        self.B._SETS["identical"] = marker
        with patch.dict(os.environ, {"MOJOLEARN_NUMERIC_MODE": "fast"}):
            self.assertIs(self.B.load_set("identical"), marker)

    def test_a_ptx_process_with_no_binary_never_selects_cpu(self):
        with patch.object(self.B, "_exec_binding", side_effect=ImportError("inert fixture binary")):
            self.reset()
            with patch.dict(os.environ, {"MOJOLEARN_NUMERIC_MODE": "fast"}):
                with patch.object(self.B, "host_binding_built", return_value=True):
                    with patch.object(self.B, "_select_cpu_only") as cpu:
                        with self.assertRaises(ImportError):
                            self.B.select()
                        cpu.assert_not_called()
        self.assertIsNone(self.B._CPU_ONLY)


class Provenance(Install):
    def test_loaded_ptx_bindings_are_recorded_and_checked(self):
        self.layout("identical")
        binary = self.root / "identical" / "_mojolearn_gbdt.so"
        self.B._record_baseline_load(str(binary))
        self.B._record_baseline_load(str(binary))
        digest = hashlib.sha256(binary.read_bytes()).hexdigest()
        self.assertEqual(self.B.baseline_selection_receipt()["loaded_files"],
                         [dict(file="identical/_mojolearn_gbdt.so", sha256=digest)])
        self.assertEqual(self.B.ptx_payload_files()["identical/_mojolearn_gbdt.so"], digest)
        binary.write_bytes(b"altered after the manifest check")
        with self.assertRaisesRegex(self.B.GpuPluginError, "changed after manifest verification"):
            self.B._record_baseline_load(str(binary))
        native = self.pkg / "cuda_native" / "sm_89" / "_mojolearn_gbdt.so"
        with self.assertRaisesRegex(self.B.GpuPluginError, "native GPU binding loaded"):
            self.B._record_baseline_load(str(native))

    def test_manifest_differing_from_the_vendor_marker_refuses(self):
        self.write_marker(dict(manifest_sha256="f" * 64))
        for mode in MODES:
            with self.subTest(mode=mode):
                self.assertIn("differs from the mojolearn-nvidia vendor marker", self.refusal(mode))

    def test_payload_and_source_failures_refuse_in_every_tier(self):
        gp = self.B.gpu_plugins
        manifest = self.root / gp.BASELINE_MANIFEST
        binary = self.root / "identical" / "_mojolearn_gbdt.so"
        commit = self.pkg / "identity_columns" / "COMMIT"
        for target, change in ((manifest, lambda p: p.unlink()),
                               (binary, lambda p: p.write_bytes(b"changed")),
                               (binary, lambda p: p.unlink()),
                               (commit, lambda p: p.write_text("c" * 40))):
            old = target.read_bytes()
            change(target)
            for mode in MODES:
                with self.subTest(target=target.name, mode=mode):
                    self.assertIn("PTX set refused", self.refusal(mode))
            target.write_bytes(old)
        self.assertEqual(self.layout("identical"), ("vendor", str(self.root)))

    def test_a_dirty_source_manifest_refuses(self):
        self.build(dirty=True)
        self.assertIn("source differs from the clean installed core", self.refusal("identical"))

    def test_a_marker_without_the_ptx_slot_refuses(self):
        gp = self.B.gpu_plugins
        doc = gp.plugin_marker("cuda", self.version, self.arches, bundled_ptx=dict(manifest_sha256=self.manifest_hash))
        del doc["bundled_ptx"]
        self.marker_path().write_text(json.dumps(doc))
        self.device = "sm_89"   # even a native device: the install itself is incomplete
        self.assertIn("PTX slot metadata", self.refusal("fast"))

    def test_an_empty_ptx_directory_refuses(self):
        for so in self.root.rglob("*.so"):
            so.unlink()
        self.device = "sm_89"
        self.assertIn("missing its PTX set", self.refusal("fast"))

    def test_other_native_errors_never_reach_ptx(self):
        with patch.object(self.B, "_pick_arch", side_effect=ImportError("native probe broken")):
            with patch.object(self.B, "_ptx_base") as ptx:
                self.reset()
                with patch.dict(os.environ, {"MOJOLEARN_NUMERIC_MODE": "fast"}):
                    with self.assertRaisesRegex(ImportError, "native probe broken"):
                        self.B._layout()
                ptx.assert_not_called()


if __name__ == "__main__":
    unittest.main()
