# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""The NVIDIA PTX fallback as the loader decides it, on a machine with no GPU.

    python python/mojolearn/tests/test_ptx_fallback.py

The package is loaded under a private name with its __init__ skipped, so this
file needs no built binding. In a checkout without bindings, pytest must be
pointed at the file from outside the package (a symlink in another directory),
because collecting it as `mojolearn.tests.*` imports the real package.

Every case fabricates an installed split core, a `mojolearn-nvidia` plugin with
native sets and a bundled PTX payload, and a device with no compatible native
set, then asks the real `_backend`. Nothing loads an extension and nothing
touches a device. The rules under test (docs/NVIDIA_PTX_IDENTITY.md):

  * native first; only `_NoCompatibleNative` on a detected device reaches PTX;
  * FAST and DETERMINISTIC take the PTX payload with no admission;
  * IDENTICAL takes it only under a bundled or local admission for exactly this
    configuration, and otherwise refuses naming the qualification command;
  * a per-call identical request on an unqualified PTX process refuses;
  * no refusal ever selects the CPU set.
"""
import copy
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
CONFIG = dict(device_name="Synthetic A100", compute_capability=[8, 0],
              driver_version="580.1.2", cuda_driver_version=13000)
OTHER = dict(device_name="Synthetic A10", compute_capability=[8, 6],
             driver_version="570.9.9", cuda_driver_version=12080)
SOURCE = "a" * 40
FIXTURES = ["base", "odd"]


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
    """A fabricated install: split core, NVIDIA plugin, bundled PTX, sm_80 device."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.site = Path(self.tmp.name) / "site-packages"
        self.pkg = self.site / "mojolearn"
        self.pkg.mkdir(parents=True)
        self.admissions = Path(self.tmp.name) / "state"
        self.B = fresh()
        self.A = importlib.import_module(ALIAS + ".ptx_admission")
        self.version = self.B._CORE_VERSION
        cleared = ("MOJOLEARN_VENDOR", "MOJOLEARN_GPU_ARCH", "MOJOLEARN_VENDOR_FORCE", "MOJOLEARN_CUDA_PATH",
                   "MOJOLEARN_EXPERIMENTAL_PTX", "MOJOLEARN_PTX_QUALIFYING", "MOJOLEARN_NUMERIC_MODE")
        env = patch.dict(os.environ, {"MOJOLEARN_PTX_ADMISSION_DIR": str(self.admissions)})
        env.start()
        self.addCleanup(env.stop)
        for key in cleared:
            os.environ.pop(key, None)
        argv = patch.object(sys, "argv", ["pytest"])
        argv.start()
        self.addCleanup(argv.stop)
        self.build()

    def dist_info(self, name, marker=None):
        d = self.site / f"{name.replace('-', '_')}-{self.version}.dist-info"
        d.mkdir(parents=True, exist_ok=True)
        (d / "METADATA").write_text(f"Metadata-Version: 2.4\nName: {name}\nVersion: {self.version}\n")
        if marker is not None:
            (d / marker[0]).write_text(marker[1])

    def release_admission(self, manifest_hash, configurations):
        A = self.A
        return dict(schema=A.SCHEMA, qualified=True, numeric_mode="identical", source_commit=SOURCE,
                    manifest_sha256=manifest_hash, coverage_contract=A.COVERAGE_CONTRACT,
                    coverage=dict(inventory_sha256="c" * 64, harness_sha256="d" * 64,
                        shared=dict(lanes=["synthetic-lane"], fixtures=list(A.SHARED_FIXTURES),
                                    parts=list(A.SHARED_PARTS), vendors=["cuda", "hip", "metal"],
                                    comparison_sha256="e" * 64),
                        nvidia=dict(lanes=["synthetic-lane"], fixtures=list(A.NVIDIA_FIXTURES),
                                    parts=list(A.NVIDIA_PARTS), comparison_sha256="f" * 64)),
                    configurations=copy.deepcopy(configurations))

    def build(self, admitted=(OTHER,)):
        """`admitted`: the configurations the bundled release admission names."""
        B, gp = self.B, self.B.gpu_plugins
        self.dist_info("mojolearn", (gp.CORE_MARKER, json.dumps(gp.core_marker(self.version))))
        arches = ["sm_89", "sm_90a"]
        for arch in arches:
            for tier in ("", "identical"):
                d = self.pkg / gp.native_directory("cuda") / arch / tier
                d.mkdir(parents=True, exist_ok=True)
                (d / "_mojolearn_gbdt.so").write_bytes(b"inert native")
        self.root = self.pkg / "cuda_ptx" / "sm_80"
        files = []
        for tier in ("fast", "deterministic", "identical"):
            d = self.root if tier == "fast" else self.root / tier
            d.mkdir(parents=True, exist_ok=True)
            binary = d / "_mojolearn_gbdt.so"
            binary.write_bytes(b"fake embedded PTX " + tier.encode())
            files.append(dict(file=binary.relative_to(self.root).as_posix(), numeric_mode=tier,
                              sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                              ptx_modules=[dict(target="sm_80", sha256="b" * 64)]))
        self.identical_sha = files[2]["sha256"]
        manifest = dict(schema="mojolearn.ptx-baseline.v1", code_format="ptx-baseline", vendor="cuda",
                        target="sm_80", min_compute_capability=[8, 0], source_commit=SOURCE,
                        source_dirty=False, experimental=True, identical_qualified=False,
                        qualification_required=True, errors=[], files=files)
        (self.root / gp.BASELINE_MANIFEST).write_text(json.dumps(manifest))
        self.manifest_hash = hashlib.sha256((self.root / gp.BASELINE_MANIFEST).read_bytes()).hexdigest()
        raw = json.dumps(self.release_admission(self.manifest_hash, list(admitted))).encode()
        (self.root / self.A.ADMISSION_FILE).write_bytes(raw)
        bundle = dict(manifest_sha256=self.manifest_hash, admission_sha256=hashlib.sha256(raw).hexdigest())
        self.dist_info(gp.plugin("cuda")["distribution"],
                       (gp.PLUGIN_MARKER, json.dumps(gp.plugin_marker("cuda", self.version, arches, bundled_ptx=bundle))))
        (self.pkg / "identity_columns").mkdir(exist_ok=True)
        (self.pkg / "identity_columns" / "COMMIT").write_text(SOURCE)
        (self.pkg / "verify_reference").mkdir(exist_ok=True)
        (self.pkg / "verify_reference" / "table.json").write_text(
            json.dumps(dict(fixtures={f: {} for f in FIXTURES}, cells={})))
        (self.pkg / "_identity_break.py").write_text("# synthetic harness\n")
        self.reference = self.A.reference_hashes(str(self.pkg))
        B._pkg_dir = lambda: str(self.pkg)
        B._probe_box = lambda: probe(B)
        B._device_arch = lambda vendor: ("sm_80", "synthetic driver")
        B._ptx_runtime_configuration = lambda: copy.deepcopy(CONFIG)
        self.reset()

    def reset(self):
        B = self.B
        B._LAYOUT = B._VENDOR_SELECTED = B._VENDOR_HOW = B._ARCH_SELECTED = B._ARCH_HOW = None
        B._SPLIT = B._BASELINE_SELECTION = B._BASELINE_ROOT = None
        B._BASELINE_FILES = {}
        B._SELECTED = B._CPU_ONLY = B._DEFAULT_MODE = None
        B._SETS.clear()

    def layout(self, mode):
        self.reset()
        with patch.dict(os.environ, {"MOJOLEARN_NUMERIC_MODE": mode}):
            return self.B._layout()

    def refusal(self, mode):
        """The layout's refusal, which must be a GPU refusal `select()` never turns into the CPU set."""
        self.reset()
        with patch.dict(os.environ, {"MOJOLEARN_NUMERIC_MODE": mode}):
            with patch.object(self.B, "host_binding_built", return_value=True):
                with patch.object(self.B, "_select_cpu_only") as cpu:
                    with self.assertRaises(self.B.GpuPluginError) as ctx:
                        self.B.select()
                    cpu.assert_not_called()
        self.assertIsNone(self.B._CPU_ONLY)
        return str(ctx.exception)

    def summaries(self):
        return [dict(profile=profile, verdict="VERIFIED",
                     counts=dict(IDENTICAL=12, DIVERGENT=0, REFUSED=0, OWED=0, **{"N/A": 3}),
                     lanes_in_scope=3, lanes_verified=2, lanes_not_applicable=["par-synthetic"],
                     fixtures=list(FIXTURES), parts_identical=["infer", "train"], cells=15,
                     cross_check=dict(ran=True, passed=True, compared=4, agree=4), report_sha256="9" * 64)
                for profile in self.A.QUALIFY_PROFILES]

    def write_local(self, configuration=CONFIG, mutate=None):
        A = self.A
        doc = A.build_local_admission(source_commit=SOURCE, manifest_sha256=self.manifest_hash,
            configuration=configuration, reference=self.reference, summaries=self.summaries(),
            core_version=self.version, created_utc="2026-10-04T00:00:00Z")
        if mutate:
            mutate(doc)
        path = Path(A.local_admission_path(str(self.admissions), doc["key"]))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc))
        return path


class FastFallback(Install):
    def test_fast_takes_ptx_without_an_admission(self):
        self.assertEqual(self.layout("fast"), ("vendor", str(self.root)))
        receipt = self.B.baseline_selection_receipt()
        self.assertEqual((receipt["requested"], receipt["selected"], receipt["fallback"], receipt["numeric_mode"]),
                         ("native-first", "ptx-baseline", "ptx", "fast"))
        self.assertIs(receipt["identical_qualified"], False)
        self.assertIsNone(receipt["admission"])
        self.assertIs(receipt["qualifying"], False)
        self.assertEqual(receipt["configuration"], CONFIG)
        self.assertEqual(self.B.gpu_arch(), "sm_80")
        self.assertIn("not IDENTICAL-qualified", self.B.gpu_arch_how())
        self.assertEqual(self.B.tier_dir("fast"), str(self.root))
        self.assertEqual(self.B.gpu_plugin()["code_format"], "ptx-baseline")
        self.assertIs(self.B.gpu_plugin()["identical_qualified"], False)

    def test_twin_a_device_with_native_code_never_reaches_ptx(self):
        self.B._device_arch = lambda vendor: ("sm_89", "synthetic driver")
        self.assertTrue(self.layout("fast")[1].endswith("cuda_native/sm_89"))
        self.assertIsNone(self.B.baseline_selection_receipt())

    def test_deterministic_takes_ptx_without_an_admission(self):
        self.assertEqual(self.layout("deterministic"), ("vendor", str(self.root)))
        receipt = self.B.baseline_selection_receipt()
        self.assertEqual(receipt["numeric_mode"], "deterministic")
        self.assertIs(receipt["identical_qualified"], False)
        self.assertEqual(self.B.tier_dir("deterministic"), str(self.root / "deterministic"))

    def test_fast_does_not_need_a_readable_device_record(self):
        def unreadable():
            raise ValueError("PTX admission currently requires one visible CUDA device")
        self.B._ptx_runtime_configuration = unreadable
        self.assertEqual(self.layout("fast"), ("vendor", str(self.root)))
        receipt = self.B.baseline_selection_receipt()
        self.assertIsNone(receipt["configuration"])
        self.assertIs(receipt["identical_qualified"], False)
        self.assertIn("one visible CUDA device", self.refusal("identical"))

    def test_manifest_only_descriptor_serves_fast_and_refuses_identical(self):
        # A vendor marker that binds the payload and names no release admission.
        self.reset()
        found = {"cuda": {"distribution": "mojolearn-nvidia", "bundled_ptx": {"manifest_sha256": self.manifest_hash}}}
        (self.root / self.A.ADMISSION_FILE).unlink()
        with patch.dict(self.B._PLUGINS_FOUND, found, clear=True):
            with patch.dict(os.environ, {"MOJOLEARN_NUMERIC_MODE": "fast"}):
                self.assertEqual(self.B._admitted_baseline_base(str(self.pkg), "native refusal"), str(self.root))
            self.assertIs(self.B.baseline_selection_receipt()["identical_qualified"], False)
            self.reset()
            with patch.dict(os.environ, {"MOJOLEARN_NUMERIC_MODE": "identical"}):
                with self.assertRaisesRegex(self.B.GpuPluginError, "bundles no PTX identity admission"):
                    self.B._admitted_baseline_base(str(self.pkg), "native refusal")

    def test_payload_and_provenance_failures_refuse_in_every_tier(self):
        gp = self.B.gpu_plugins
        manifest = self.root / gp.BASELINE_MANIFEST
        admission = self.root / self.A.ADMISSION_FILE
        binary = self.root / "_mojolearn_gbdt.so"
        commit = self.pkg / "identity_columns" / "COMMIT"
        for target, change in ((manifest, lambda p: p.write_bytes(p.read_bytes() + b" ")),
                               (manifest, lambda p: p.unlink()),
                               (admission, lambda p: p.write_bytes(p.read_bytes() + b" ")),
                               (admission, lambda p: p.unlink()),
                               (binary, lambda p: p.write_bytes(b"changed")),
                               (binary, lambda p: p.unlink()),
                               (commit, lambda p: p.write_text("c" * 40))):
            old = target.read_bytes()
            change(target)
            for mode in ("fast", "deterministic", "identical"):
                with self.subTest(target=target.name, mode=mode):
                    self.assertIn("PTX fallback refused", self.refusal(mode))
                    self.assertIsNone(self.B.baseline_selection_receipt())
            target.write_bytes(old)
        self.assertEqual(self.layout("fast"), ("vendor", str(self.root)))

    def test_no_bundle_refuses_and_other_native_errors_never_reach_ptx(self):
        gp = self.B.gpu_plugins
        marker = self.site / f"mojolearn_nvidia-{self.version}.dist-info" / gp.PLUGIN_MARKER
        marker.write_text(json.dumps(gp.plugin_marker("cuda", self.version, ["sm_89", "sm_90a"])))
        self.assertIn("no PTX fallback", self.refusal("fast"))
        self.build()
        with patch.object(self.B, "_pick_arch", side_effect=ImportError("native probe broken")):
            with patch.object(self.B, "_admitted_baseline_base") as fallback:
                self.reset()
                with patch.dict(os.environ, {"MOJOLEARN_NUMERIC_MODE": "fast"}):
                    with self.assertRaisesRegex(ImportError, "native probe broken"):
                        self.B._layout()
                fallback.assert_not_called()

    def test_a_fast_ptx_process_with_no_binary_never_selects_cpu(self):
        # Every FAST binding path is absent from this fixture except gbdt, which is inert bytes.
        with patch.object(self.B, "_exec_binding", side_effect=ImportError("inert fixture binary")):
            self.reset()
            with patch.dict(os.environ, {"MOJOLEARN_NUMERIC_MODE": "fast"}):
                with patch.object(self.B, "host_binding_built", return_value=True):
                    with patch.object(self.B, "_select_cpu_only") as cpu:
                        with self.assertRaises(ImportError):
                            self.B.select()
                        cpu.assert_not_called()
        self.assertIsNone(self.B._CPU_ONLY)


class IdenticalAdmission(Install):
    def test_identical_refuses_and_names_the_command(self):
        text = self.refusal("identical")
        self.assertIn("IDENTICAL PTX fallback refused", text)
        self.assertIn(self.B.PTX_QUALIFY_COMMAND, text)
        self.assertEqual(self.B.PTX_QUALIFY_COMMAND, "python -m mojolearn verify --qualify-gpu")
        self.assertIn("this device/driver configuration is not admitted", text)
        self.assertIn("no local qualification exists", text)
        self.assertIn("No CPU substitution", text)
        self.assertIsNone(self.B.baseline_selection_receipt())

    def test_default_mode_is_identical_and_refuses(self):
        self.reset()
        with self.assertRaisesRegex(self.B.GpuPluginError, "verify --qualify-gpu"):
            self.B._layout()

    def test_bundled_release_admission_is_accepted_for_its_configuration(self):
        self.build(admitted=(CONFIG,))
        self.assertEqual(self.layout("identical"), ("vendor", str(self.root)))
        receipt = self.B.baseline_selection_receipt()
        self.assertIs(receipt["identical_qualified"], True)
        self.assertEqual(receipt["admission"], "bundled")
        self.assertIn("bundled admission", self.B.gpu_arch_how())

    def test_local_admission_is_accepted_and_reported(self):
        path = self.write_local()
        self.assertEqual(self.layout("identical"), ("vendor", str(self.root)))
        receipt = self.B.baseline_selection_receipt()
        self.assertIs(receipt["identical_qualified"], True)
        self.assertEqual(receipt["admission"], "local")
        self.assertEqual(receipt["admission_detail"]["admission_path"], str(path))
        self.assertEqual(receipt["admission_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(self.B.gpu_plugin()["admission"], "local")
        # twin: without the file the same install refuses
        path.unlink()
        self.assertIn("verify --qualify-gpu", self.refusal("identical"))

    def test_stale_local_admission_refuses_again(self):
        self.write_local()
        self.assertEqual(self.layout("identical")[1], str(self.root))
        with self.subTest("driver update"):
            self.B._ptx_runtime_configuration = lambda: {**CONFIG, "driver_version": "581.0.0"}
            self.assertIn("verify --qualify-gpu", self.refusal("identical"))
        with self.subTest("CUDA API change"):
            self.B._ptx_runtime_configuration = lambda: {**CONFIG, "cuda_driver_version": 13010}
            self.assertIn("verify --qualify-gpu", self.refusal("identical"))
        with self.subTest("another device"):
            self.B._ptx_runtime_configuration = lambda: {**CONFIG, "device_name": "Synthetic A40"}
            self.assertIn("verify --qualify-gpu", self.refusal("identical"))
        self.B._ptx_runtime_configuration = lambda: copy.deepcopy(CONFIG)
        with self.subTest("different wheel: reference table"):
            table = self.pkg / "verify_reference" / "table.json"
            old = table.read_bytes()
            table.write_bytes(old + b" ")
            self.assertIn("verify --qualify-gpu", self.refusal("identical"))
            table.write_bytes(old)
        with self.subTest("different wheel: PTX payload"):
            self.build()                      # same bytes, so the admission still fits
            self.assertEqual(self.layout("identical")[1], str(self.root))
            (self.root / "_mojolearn_gbdt.so").write_bytes(b"a rebuilt payload")
            self.assertIn("PTX fallback refused", self.refusal("identical"))
        self.build()
        self.assertEqual(self.layout("identical")[1], str(self.root))

    def test_an_edited_local_admission_refuses(self):
        for label, mutate, expect in (
                ("another driver inside", lambda d: d["configuration"].update(driver_version="1.2"), "another device or driver"),
                ("unqualified", lambda d: d.update(qualified=False), "not a qualified local admission"),
                ("release schema", lambda d: d.update(schema=self.A.SCHEMA), "not a qualified local admission"),
                ("divergent summary", lambda d: d["comparison"]["profiles"][0]["counts"].update(DIVERGENT=1), "not complete and identical"),
                ("owed summary", lambda d: d["comparison"]["profiles"][1]["counts"].update(OWED=2), "not complete and identical"),
                ("one profile", lambda d: d["comparison"]["profiles"].pop(), "every verification profile"),
                ("lanes unaccounted", lambda d: d["comparison"]["profiles"][0].update(lanes_in_scope=9), "every lane in scope"),
                ("another source", lambda d: d.update(source_commit="c" * 40), "another source commit"),
                ("another payload", lambda d: d.update(manifest_sha256="c" * 64), "another PTX payload")):
            with self.subTest(label):
                path = self.write_local(mutate=mutate)
                text = self.refusal("identical")
                self.assertIn(expect, text)
                self.assertIn("verify --qualify-gpu", text)
                path.unlink()
        path = self.write_local()
        path.write_text("{not json")
        self.assertIn("verify --qualify-gpu", self.refusal("identical"))

    def test_a_release_record_is_not_a_local_admission(self):
        doc = self.release_admission(self.manifest_hash, [CONFIG])
        with self.assertRaisesRegex(ValueError, "not a qualified local admission"):
            self.A.validate_local_admission(doc, source_commit=SOURCE, manifest_sha256=self.manifest_hash,
                                            configuration=CONFIG, reference=self.reference)
        local = json.loads(self.write_local().read_text())
        with self.assertRaises(ValueError):
            self.A.validate_admission(local, source_commit=SOURCE, manifest_sha256=self.manifest_hash,
                                      configuration=CONFIG)


class PerCallIdentical(Install):
    def test_per_call_identical_on_a_fast_ptx_process_refuses(self):
        self.layout("fast")
        with patch.dict(os.environ, {"MOJOLEARN_NUMERIC_MODE": "fast"}):
            for call in (lambda: self.B.load_set("identical"),
                         lambda: self.B.binding("_mojolearn_gbdt", "identical"),
                         lambda: self.B.set_default_mode("identical")):
                with self.assertRaises(self.B.GpuPluginError) as ctx:
                    call()
                self.assertIn("numeric_mode='identical' refused", str(ctx.exception))
                self.assertIn(self.B.PTX_QUALIFY_COMMAND, str(ctx.exception))
                self.assertIn("No CPU substitution", str(ctx.exception))
            self.assertIsNone(self.B._CPU_ONLY)
            self.assertEqual(self.B.default_mode(), "fast")

    def test_a_cached_identical_set_does_not_answer_a_refused_request(self):
        self.layout("fast")
        marker = object()
        with patch.dict(os.environ, {"MOJOLEARN_NUMERIC_MODE": "fast"}):
            with self.B.host_helper_scope():
                self.B._SETS["identical"] = marker         # what `_buffer._native` leaves behind
                self.assertIs(self.B.load_set("identical"), marker)
            with self.assertRaises(self.B.GpuPluginError):
                self.B.load_set("identical")
            self.B._SETS["fast"] = marker
            self.assertIs(self.B.load_set("fast"), marker)   # the process's own tier is untouched

    def test_first_identical_request_refuses_before_any_binary_loads(self):
        # The layout is decided inside the first load_set call; the refusal must still fire.
        self.reset()
        with patch.dict(os.environ, {"MOJOLEARN_NUMERIC_MODE": "fast"}):
            with patch.object(self.B, "_load_member") as load:
                with self.assertRaises(self.B.GpuPluginError):
                    self.B.load_set("identical")
                load.assert_not_called()

    def test_per_call_identical_is_served_once_this_configuration_is_admitted(self):
        self.write_local()
        self.layout("fast")
        receipt = self.B.baseline_selection_receipt()
        self.assertEqual((receipt["numeric_mode"], receipt["admission"]), ("fast", "local"))
        marker = object()
        self.B._SETS["identical"] = marker
        with patch.dict(os.environ, {"MOJOLEARN_NUMERIC_MODE": "fast"}):
            self.assertIs(self.B.load_set("identical"), marker)

    def test_native_and_forced_experimental_processes_are_not_judged_here(self):
        self.B._device_arch = lambda vendor: ("sm_89", "synthetic driver")
        self.layout("fast")
        self.assertIsNone(self.B._ptx_identical_refusal())
        self.B._BASELINE_SELECTION = dict(requested="ptx-baseline", identical_qualified=False)
        self.assertIsNone(self.B._ptx_identical_refusal())


class Qualifying(Install):
    def test_the_qualification_run_loads_identical_ptx_without_claiming_it(self):
        with patch.dict(os.environ, {"MOJOLEARN_PTX_QUALIFYING": "1"}):
            self.assertEqual(self.layout("identical"), ("vendor", str(self.root)))
        receipt = self.B.baseline_selection_receipt()
        self.assertIs(receipt["identical_qualified"], False)
        self.assertIs(receipt["qualifying"], True)
        self.assertIsNone(receipt["admission"])
        self.assertIn("qualification run in progress", self.B.gpu_arch_how())
        self.assertIsNone(self.B._ptx_identical_refusal())

    def test_the_command_line_is_recognized_and_nothing_else_is(self):
        for argv, expected in ((["-m", "verify", "--qualify-gpu"], True),
                               (["/site/mojolearn/__main__.py", "verify", "--json", "--qualify-gpu"], True),
                               (["-m", "verify", "--all"], False),
                               (["train.py", "verify", "--qualify-gpu"], False),
                               (["-m", "doctor", "--qualify-gpu"], False)):
            with self.subTest(argv=argv):
                with patch.object(sys, "argv", argv):
                    self.assertIs(self.B._ptx_qualifying(), expected)
        with patch.dict(os.environ, {"MOJOLEARN_PTX_QUALIFYING": "0"}):
            self.assertIs(self.B._ptx_qualifying(), False)

    def test_qualifying_never_relaxes_fast_or_an_existing_admission(self):
        with patch.dict(os.environ, {"MOJOLEARN_PTX_QUALIFYING": "1"}):
            self.layout("fast")
            self.assertIs(self.B.baseline_selection_receipt()["qualifying"], False)
            self.write_local()
            self.layout("identical")
            receipt = self.B.baseline_selection_receipt()
            self.assertEqual((receipt["admission"], receipt["qualifying"]), ("local", False))


def clean_report(case, profile, **changes):
    """A finished verifier report that qualifies; each test moves one thing."""
    lanes = {"ols": dict(state="VERIFIED", reason=None), "pca": dict(state="VERIFIED", reason=None),
             "par-synthetic": dict(state="NOT APPLICABLE", reason="two-device claim")}
    cells = [dict(lane=lane, fixture=fixture, part=part, state="IDENTICAL", detail="")
             for lane in ("ols", "pca") for fixture in FIXTURES for part in ("train", "infer")]
    cells.append(dict(lane="ols", fixture="base", part="stepfull", state="N/A", detail="n/a:no-sequence"))
    report = dict(
        format="mojolearn.verify-all-report.v1", verdict="VERIFIED", exit=0, depth="full",
        fixtures=list(FIXTURES), selection=dict(profile=dict(name=profile)),
        device=dict(vendor="cuda", numeric_mode="identical"),
        table=dict(sha256=case.reference["table_sha256"]),
        harness=dict(sha256=case.reference["harness_sha256"], matches_table=True),
        execution=dict(interrupted=None, completed_cells=4, total_cells=4),
        self_test=dict(passed=True), cells=cells,
        counts={"IDENTICAL": 8, "DIVERGENT": 0, "OWED": 0, "REFUSED": 0, "N/A": 1},
        lane_accounting=dict(total=3, verdict_scope=sorted(lanes), lanes=lanes),
        bindings=[dict(module="_mojolearn_gbdt", sha256=case.identical_sha, size=1),
                  dict(module="mojolearn._host._mojolearn_forest_host", sha256="0" * 64, size=1, kind="host")],
        cross_check=(dict(ran=True, passed=True, compared=4, agree=4, differ=0) if profile == "routine"
                     else dict(ran=False, passed=None)))
    report.update(changes)
    return report


class JudgeReport(Install):
    def judge(self, report, profile="routine"):
        return self.A.judge_report(report, profile=profile, reference=self.reference, table_fixtures=FIXTURES,
                                   ptx_identical_sha256={self.identical_sha}, report_sha256="9" * 64)

    def test_a_complete_identical_report_qualifies(self):
        for profile in self.A.QUALIFY_PROFILES:
            result = self.judge(clean_report(self, profile), profile)
            self.assertEqual((result["differing"], result["missing"], result["incomplete"]), ([], [], []))
            self.assertEqual(result["summary"]["lanes_verified"], 2)
            self.assertEqual(result["summary"]["lanes_not_applicable"], ["par-synthetic"])
            self.assertEqual(result["summary"]["parts_identical"], ["infer", "train"])

    def moved(self, expect, bucket, profile="routine", **changes):
        result = self.judge(clean_report(self, profile, **changes), profile)
        self.assertTrue(any(expect in row for row in result[bucket]), (expect, result))
        return result

    def test_each_defect_alone_denies_qualification(self):
        base = clean_report(self, "routine")
        def cells(state, detail):
            return [dict(base["cells"][0], state=state, detail=detail)] + base["cells"][1:]
        def lanes(state):
            rows = dict(base["lane_accounting"]["lanes"], pca=dict(state=state, reason="synthetic reason"))
            return dict(base["lane_accounting"], lanes=rows)
        self.moved("ols/base/train: this box 11, reference 22", "differing", cells=cells("DIVERGENT", "this box 11, reference 22"))
        self.moved("ols/base/train: no committed record", "missing", cells=cells("OWED", "no committed record carries this cell part yet"))
        self.moved("ols/base/train: worker exited", "incomplete", cells=cells("REFUSED", "worker exited 1"))
        for state in ("OWED", "HELD", "NOT RUN", "UNDECLARED", "SMOKE"):
            self.moved(f"pca: lane {state}", "missing", lane_accounting=lanes(state))
        self.moved("pca: lane DIVERGENT", "differing", lane_accounting=lanes("DIVERGENT"))
        self.moved("pca: lane REFUSED", "incomplete", lane_accounting=lanes("REFUSED"))
        self.moved("full-depth", "incomplete", fixtures=["base"])
        self.moved("full-depth", "incomplete", depth="base")
        self.moved("not cuda in identical mode", "incomplete", device=dict(vendor="cuda", numeric_mode="fast"))
        self.moved("reference table and harness this install ships", "incomplete", table=dict(sha256="1" * 64))
        self.moved("not generated by the shipped harness", "missing",
                   harness=dict(sha256=self.reference["harness_sha256"], matches_table=False))
        self.moved("stopped after", "incomplete", execution=dict(interrupted="timeout", completed_cells=1, total_cells=4))
        self.moved("self-test", "incomplete", self_test=dict(passed=False))
        self.moved("not loaded from the IDENTICAL PTX payload", "incomplete",
                   bindings=[dict(module="_mojolearn_gbdt", sha256="7" * 64, size=1)])
        self.moved("not loaded from the IDENTICAL PTX payload", "incomplete", bindings=[])
        self.moved("GPU and CPU inference differ", "differing", cross_check=dict(ran=True, passed=False, differ=2))
        self.moved("did not run to a pass", "incomplete", cross_check=dict(ran=False, passed=None))
        self.moved("report is for profile", "incomplete", selection=dict(profile=dict(name="neural-training")))
        self.moved("verifier verdict", "incomplete", verdict="MISMATCH", exit=1)
        self.moved("no lane was verified", "incomplete",
                   lane_accounting=dict(total=1, verdict_scope=["par-synthetic"], lanes=base["lane_accounting"]["lanes"]))

    def test_no_report_is_incomplete(self):
        for report in (None, dict(format="mojolearn.verify-all-report.v1", verdict="CANNOT RUN", exit=4, detail="import raised")):
            result = self.judge(report)
            self.assertIsNone(result["summary"])
            self.assertTrue(result["incomplete"])


class QualifyCommand(Install):
    def command(self, reports, argv=("-m", "verify", "--qualify-gpu"), mode="identical"):
        Q = importlib.import_module(ALIAS + "._ptx_qualify")
        self.reset()
        seen = []

        def run(profile, report_path, log_path, args):
            seen.append(profile)
            report = reports(profile)
            if report is not None:
                Path(report_path).write_text(json.dumps(report))
            Path(log_path).write_text("synthetic verifier log\n")
            return 0 if report is None else report.get("exit", 0)

        args = types.SimpleNamespace(json=True, cell_timeout=120.0, cpu_threads=1)
        with patch.dict(os.environ, {"MOJOLEARN_NUMERIC_MODE": mode}), patch.object(sys, "argv", list(argv)):
            self.B._layout()
            out = tempfile.TemporaryFile("w+")
            with patch.object(sys, "stdout", out), patch.object(sys, "stderr", tempfile.TemporaryFile("w+")):
                code = Q.cmd_qualify_gpu(args, backend=self.B, run=run)
            out.seek(0)
            return code, json.loads(out.read()), seen

    def admissions_written(self):
        return sorted(p.name for p in self.admissions.glob("*.json")) if self.admissions.exists() else []

    def test_a_clean_run_writes_an_admission_the_loader_then_accepts(self):
        self.assertIn("verify --qualify-gpu", self.refusal("identical"))
        code, result, seen = self.command(lambda profile: clean_report(self, profile))
        self.assertEqual((code, result["verdict"]), (0, "QUALIFIED"))
        self.assertEqual(seen, list(self.A.QUALIFY_PROFILES))
        self.assertEqual(len(self.admissions_written()), 1)
        doc = json.loads(Path(result["admission_path"]).read_text())
        self.assertEqual(doc["schema"], self.A.LOCAL_SCHEMA)
        self.assertNotEqual(doc["schema"], self.A.SCHEMA)
        self.assertEqual((doc["source_commit"], doc["manifest_sha256"], doc["configuration"], doc["reference"]),
                         (SOURCE, self.manifest_hash, CONFIG, self.reference))
        self.assertEqual([row["profile"] for row in doc["comparison"]["profiles"]], list(self.A.QUALIFY_PROFILES))
        for row in doc["comparison"]["profiles"]:
            raw = (Path(result["evidence"]) / f"{row['profile']}.json").read_bytes()
            self.assertEqual(row["report_sha256"], hashlib.sha256(raw).hexdigest())
        # an ordinary process, with no qualification signal, now loads IDENTICAL
        self.assertEqual(self.layout("identical"), ("vendor", str(self.root)))
        self.assertEqual(self.B.baseline_selection_receipt()["admission"], "local")
        code, result, seen = self.command(lambda profile: clean_report(self, profile))
        self.assertEqual((code, result["verdict"], seen), (0, "ALREADY QUALIFIED", []))

    def test_a_differing_lane_writes_nothing_and_is_reported(self):
        def reports(profile):
            report = clean_report(self, profile)
            if profile == "neural-training":
                report["cells"][0].update(state="DIVERGENT", detail="this box 11, reference 22")
                report.update(verdict="MISMATCH", exit=1)
            return report
        code, result, seen = self.command(reports)
        self.assertEqual((code, result["verdict"]), (1, "NOT QUALIFIED"))
        self.assertEqual(seen, list(self.A.QUALIFY_PROFILES))
        self.assertEqual(result["differing"], ["ols/base/train: this box 11, reference 22"])
        self.assertEqual(self.admissions_written(), [])
        self.assertIn("verify --qualify-gpu", self.refusal("identical"))

    def test_missing_reference_data_writes_nothing_and_names_what_is_missing(self):
        def reports(profile):
            report = clean_report(self, profile)
            report["lane_accounting"]["lanes"]["pca"] = dict(state="OWED", reason="no committed record carries a hash")
            report["cells"][1].update(state="OWED", detail="no committed record carries this cell part yet")
            return report                                  # `verify` itself still says VERIFIED, exit 0
        code, result, _ = self.command(reports)
        self.assertEqual((code, result["verdict"]), (5, "NOT QUALIFIED"))
        self.assertEqual(result["differing"], [])
        self.assertIn("pca: lane OWED: no committed record carries a hash", result["missing_reference"])
        self.assertIn("ols/base/infer: no committed record carries this cell part yet", result["missing_reference"])
        self.assertEqual(self.admissions_written(), [])

    def test_a_run_that_did_not_finish_writes_nothing(self):
        code, result, _ = self.command(lambda profile: clean_report(self, profile) if profile == "routine" else None)
        self.assertEqual((code, result["verdict"]), (4, "INCOMPLETE"))
        self.assertEqual(self.admissions_written(), [])

    def test_native_devices_and_unreadable_devices_write_nothing(self):
        self.B._device_arch = lambda vendor: ("sm_89", "synthetic driver")
        with patch.object(self.B, "vendor", return_value="cuda"):
            code, result, seen = self.command(lambda profile: clean_report(self, profile))
        self.assertEqual((code, result["verdict"], seen), (0, "NOT NEEDED", []))
        self.B._device_arch = lambda vendor: ("sm_80", "synthetic driver")
        def unreadable():
            raise ValueError("PTX admission currently requires one visible CUDA device")
        self.B._ptx_runtime_configuration = unreadable
        code, result, seen = self.command(lambda profile: clean_report(self, profile))
        self.assertEqual((code, result["verdict"], seen), (4, "CANNOT RUN", []))
        self.assertIn("one visible CUDA device", result["detail"])
        self.assertEqual(self.admissions_written(), [])

    def test_the_verifier_children_run_identical_under_the_qualifying_signal(self):
        Q = importlib.import_module(ALIAS + "._ptx_qualify")
        args = types.SimpleNamespace(cell_timeout=90.0, cpu_threads=2)
        log = Path(self.tmp.name) / "child.log"
        for profile, neural in (("routine", False), ("neural-training", True)):
            with patch.object(Q.subprocess, "run", return_value=types.SimpleNamespace(returncode=0)) as run:
                self.assertEqual(Q._run_verifier(profile, "report.json", str(log), args), 0)
            command, env = run.call_args.args[0], run.call_args.kwargs["env"]
            self.assertEqual(command[1:5], ["-m", "mojolearn", "verify", "--all"])
            self.assertEqual("--neural-training" in command, neural)
            self.assertNotIn("--quick", command)
            self.assertNotIn("--lanes", command)
            self.assertEqual((env["MOJOLEARN_NUMERIC_MODE"], env["MOJOLEARN_PTX_QUALIFYING"]), ("identical", "1"))

    def test_the_cli_dispatches_the_flag(self):
        main = importlib.import_module(ALIAS + ".__main__")
        Q = importlib.import_module(ALIAS + "._ptx_qualify")
        args = main.build_parser().parse_args(["verify", "--qualify-gpu"])
        self.assertIs(args.qualify_gpu, True)
        with patch.object(Q, "cmd_qualify_gpu", return_value=17) as command:
            with patch.object(main._verify_all, "cmd_verify_all") as suite, patch.object(main._verify, "cmd_verify") as card:
                self.assertEqual(args.func(args), 17)
                suite.assert_not_called()
                card.assert_not_called()
        command.assert_called_once_with(args)
        # twin: without the flag the ordinary verifier is dispatched
        plain = main.build_parser().parse_args(["verify", "--quick"])
        self.assertIs(plain.qualify_gpu, False)
        with patch.object(Q, "cmd_qualify_gpu") as command:
            with patch.object(main._verify_all, "cmd_verify_all", return_value=0):
                plain.func(plain)
            command.assert_not_called()


if __name__ == "__main__":
    unittest.main()
