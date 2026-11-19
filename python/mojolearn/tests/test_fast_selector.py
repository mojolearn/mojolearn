# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""The process default named `fast` as the selector decides it.

    python python/mojolearn/tests/test_fast_selector.py

The package is loaded under a private name with its __init__ skipped, so this
file needs no built binding. In a checkout without bindings, pytest must be
pointed at the file from outside the package (a symlink in another directory),
because collecting it as `mojolearn.tests.*` imports the real package.

Every case fabricates a flat install (inert files under a temporary package
directory) and asks the real `_backend`. Nothing loads an extension. The rules
under test (`_backend._FAST_ANSWER`):

  * off Apple, on an install with no fast binaries, MOJOLEARN_NUMERIC_MODE=fast
    and `set_numeric_mode("fast")` select the identical set: the same files the
    name `identical` selects, no tier stub, and `numeric_mode()` reads back
    'fast';
  * off Apple, a CPU-only install answers the name `fast` with its CPU bindings;
  * off Apple, an install that carries fast binaries still loads them;
  * on Apple nothing changes: `fast` names the fast set, built or not;
  * a per-estimator `numeric_mode='fast'` (`binding`, `load_set`) still names
    the fast set;
  * a retired mode value is refused with its wording.

The result hash of a `fast` process against an `identical` one on a built
install is `test_fast_default_runs_the_identical_set` below; it needs the
compiled package and skips without it.
"""
import importlib
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

PKG_SRC = Path(__file__).resolve().parents[1]
ALIAS = "mojolearn_fast_selector_under_test"
BUILT = ("_mojolearn_gbdt", "_mojolearn_rf")


def fresh(module="_backend"):
    for name in [n for n in sys.modules if n == ALIAS or n.startswith(ALIAS + ".")]:
        del sys.modules[name]
    pkg = types.ModuleType(ALIAS)
    pkg.__path__ = [str(PKG_SRC)]
    sys.modules[ALIAS] = pkg
    return importlib.import_module(ALIAS + "." + module)


class Selector(unittest.TestCase):
    """A fabricated flat install: `<pkg>/identical/*.so`, and `<pkg>/*.so` when
    a fast set is asked for."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.pkg = Path(self.tmp.name) / "mojolearn"
        (self.pkg / "identical").mkdir(parents=True)
        for name in BUILT:
            (self.pkg / "identical" / (name + ".so")).write_bytes(b"inert identical")
        self.B = B = fresh()
        self.addCleanup(self.forget)
        env = patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        for key in ("MOJOLEARN_VENDOR", "MOJOLEARN_NUMERIC_MODE"):
            os.environ.pop(key, None)
        B._pkg_dir = lambda: str(self.pkg)
        B._layout = lambda: ("flat", str(self.pkg))
        B._bound_amd_memory_manager = lambda: None

    def forget(self):
        B = self.B
        if B._CANONICAL_FINDER in sys.meta_path:
            sys.meta_path.remove(B._CANONICAL_FINDER)
        for name in [n for n in sys.modules if n == ALIAS or n.startswith(ALIAS + ".")]:
            del sys.modules[name]

    def fast_set(self):
        for name in BUILT:
            (self.pkg / (name + ".so")).write_bytes(b"inert fast")

    def select(self, mode, platform):
        with patch.object(sys, "platform", platform):
            with patch.dict(os.environ, {"MOJOLEARN_NUMERIC_MODE": mode}):
                return self.B.select()

    def stubs(self, kind):
        return sorted(n.rsplit(".", 1)[1] for n, m in sys.modules.items()
                      if n.startswith(ALIAS + "._mojolearn") and isinstance(m, kind))

    # ---- off Apple, no fast set: the identical set answers the name ----

    def test_fast_selects_the_identical_set_off_apple(self):
        B = self.B
        self.assertEqual(self.select("fast", "linux"), "identical")
        self.assertEqual((B._SELECTED, B._DEFAULT_NAME, B._FAST_ANSWER), ("identical", "fast", "identical"))
        self.assertEqual(B.default_mode(), "identical")
        self.assertEqual(B.default_name(), "fast")
        chosen = {name: path for name, (path, _) in B._CANONICAL_PATHS.items()}
        self.assertEqual(chosen, {name: str(self.pkg / "identical" / (name + ".so")) for name in BUILT})
        # no tier refusal stub: the name is answered, not refused
        self.assertEqual(self.stubs(B._IdenticalOnlyTier), [])
        # the unbuilt bindings name the identical build, as under `identical`
        missing = sys.modules[ALIAS + "._mojolearn_svm"]
        with self.assertRaisesRegex(ImportError, "MOJOLEARN_NUMERIC_MODE=identical"):
            missing.anything
        with patch.object(sys, "platform", "linux"):
            self.assertEqual(B.numeric_mode(), "fast")

    def test_the_same_files_as_the_name_identical(self):
        self.select("fast", "linux")
        fast_paths = dict(self.B._CANONICAL_PATHS)
        fast_missing = sorted(self.B._MISSING)
        self.forget()
        self.B = fresh()
        self.B._pkg_dir = lambda: str(self.pkg)
        self.B._layout = lambda: ("flat", str(self.pkg))
        self.B._bound_amd_memory_manager = lambda: None
        self.assertEqual(self.select("identical", "linux"), "identical")
        self.assertIsNone(self.B._DEFAULT_NAME)
        self.assertEqual(self.B.default_name(), "identical")
        self.assertEqual(dict(self.B._CANONICAL_PATHS), fast_paths)
        self.assertEqual(sorted(self.B._MISSING), fast_missing)

    def test_set_numeric_mode_fast_off_apple(self):
        B = self.B
        self.assertEqual(self.select("identical", "linux"), "identical")
        loaded = []
        with patch.object(sys, "platform", "linux"), patch.object(B, "load_set", side_effect=loaded.append):
            self.assertEqual(B.set_default_mode("fast"), "identical")
            self.assertEqual((B._DEFAULT_MODE, B._DEFAULT_NAME), ("identical", "fast"))
            self.assertEqual(B.default_name(), "fast")
            # None keeps the current name; the previous name is returned
            self.assertEqual(B.set_default_mode(None), "fast")
            self.assertEqual((B._DEFAULT_MODE, B._DEFAULT_NAME), ("identical", "fast"))
            self.assertEqual(B.set_default_mode(" Identical "), "fast")
            self.assertEqual((B._DEFAULT_MODE, B._DEFAULT_NAME), ("identical", None))
        self.assertEqual(loaded, ["identical", "identical", "identical"])

    def test_cpu_only_install_answers_fast_off_apple(self):
        B = self.B
        with patch.object(B, "tier_dir", side_effect=ImportError("no GPU set")), \
                patch.object(B, "host_binding_built", return_value=True), \
                patch.object(B, "_refuse_cpu_on_gpu_box"), \
                patch.object(B, "_select_cpu_only", return_value="identical") as cpu:
            self.assertEqual(self.select("fast", "linux"), "identical")
        self.assertEqual(cpu.call_args.args[1], "identical")
        self.assertEqual((B._DEFAULT_NAME, B._FAST_ANSWER), ("fast", "identical"))

    # ---- off Apple, a fast set is installed: unchanged ----

    def test_an_installed_fast_set_is_loaded_off_apple(self):
        B = self.B
        self.fast_set()
        self.assertEqual(self.select("fast", "linux"), "fast")
        self.assertEqual((B._SELECTED, B._DEFAULT_NAME, B._FAST_ANSWER), ("fast", None, None))
        self.assertEqual(B.default_name(), "fast")
        self.assertEqual(B._CANONICAL_PATHS, {})  # the flat fast tier is imported by plain import

    def test_set_numeric_mode_fast_with_a_fast_set(self):
        B = self.B
        self.fast_set()
        self.assertEqual(self.select("identical", "linux"), "identical")
        loaded = []
        with patch.object(sys, "platform", "linux"), patch.object(B, "load_set", side_effect=loaded.append):
            self.assertEqual(B.set_default_mode("fast"), "identical")
        self.assertEqual((B._DEFAULT_MODE, B._DEFAULT_NAME, loaded), ("fast", None, ["fast"]))

    # ---- Apple: unchanged, built or not ----

    def test_apple_fast_is_unchanged_without_a_fast_set(self):
        B = self.B
        self.assertEqual(self.select("fast", "darwin"), "fast")
        self.assertEqual((B._SELECTED, B._DEFAULT_NAME, B._FAST_ANSWER), ("fast", None, None))
        # every binding of the unbuilt fast tier refuses by name, as before
        missing = sys.modules[ALIAS + "._mojolearn_gbdt"]
        with self.assertRaisesRegex(ImportError, "MOJOLEARN_NUMERIC_MODE=fast"):
            missing.anything
        self.assertEqual(self.stubs(B._IdenticalOnlyTier), sorted(B._IDENTICAL_ONLY))

    def test_apple_fast_is_unchanged_with_a_fast_set(self):
        B = self.B
        self.fast_set()
        self.assertEqual(self.select("fast", "darwin"), "fast")
        self.assertEqual((B._SELECTED, B._DEFAULT_NAME, B._FAST_ANSWER), ("fast", None, None))
        self.assertNotIn(ALIAS + "._mojolearn_gbdt", sys.modules)

    def test_apple_set_numeric_mode_fast_is_unchanged(self):
        B = self.B
        self.assertEqual(self.select("identical", "darwin"), "identical")
        refusal = ImportError("no binary for that tier")
        with patch.object(sys, "platform", "darwin"), patch.object(B, "load_set", side_effect=refusal):
            with self.assertRaises(ImportError):
                B.set_default_mode("fast")
        self.assertEqual((B._DEFAULT_MODE, B._DEFAULT_NAME), ("identical", None))

    def test_apple_cpu_only_keeps_the_name(self):
        B = self.B
        with patch.object(B, "tier_dir", side_effect=ImportError("no GPU set")), \
                patch.object(B, "host_binding_built", return_value=True), \
                patch.object(B, "_refuse_cpu_on_gpu_box"), \
                patch.object(B, "_select_cpu_only", return_value="fast") as cpu:
            self.assertEqual(self.select("fast", "darwin"), "fast")
        self.assertEqual(cpu.call_args.args[1], "fast")
        self.assertIsNone(B._DEFAULT_NAME)

    # ---- what the selection does not touch ----

    def test_a_per_estimator_fast_still_names_the_fast_set(self):
        B = self.B
        self.assertEqual(self.select("fast", "linux"), "identical")
        with patch.object(sys, "platform", "linux"):
            self.assertEqual(B.tier_dir("fast"), str(self.pkg))
            with self.assertRaisesRegex(ImportError, "numeric_mode='fast' but no binary for that tier"):
                B.load_set("fast")
            with self.assertRaisesRegex(ImportError, "numeric_mode='fast' but no binary for that tier"):
                B.binding("_mojolearn_gbdt", "fast")

    def test_retired_and_unknown_values_are_refused(self):
        B = self.B
        for value in ("deterministic", "exact", "ordered", "0", "fastest"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ImportError, "it must be 'fast', or 'identical'"):
                    self.select(value, "linux")
                with self.assertRaisesRegex(ValueError, "it must be 'fast', or 'identical'"):
                    B.set_default_mode(value)
                with self.assertRaisesRegex(ValueError, "numeric_mode must be fast, identical or None"):
                    B.binding("_mojolearn_gbdt", value)
        self.assertIsNone(B._SELECTED)
        self.assertIsNone(B._DEFAULT_NAME)

    def test_the_decision_is_made_once(self):
        B = self.B
        with patch.object(sys, "platform", "linux"):
            self.assertEqual(B.fast_answer(), "identical")
        # a later change of the disk or the platform does not move it
        self.fast_set()
        with patch.object(sys, "platform", "darwin"), patch.object(B, "_fast_set_present") as probe:
            self.assertEqual(B.fast_answer(), "identical")
            probe.assert_not_called()


_CHILD = r"""
import hashlib, json, sys
import mojolearn
import numpy as np
rows = np.arange(96, dtype=np.float32).reshape(24, 4)
rows = (rows * np.float32(0.37)) % np.float32(5.0)
model = mojolearn.StandardScaler().fit(rows)
out = np.ascontiguousarray(model.transform(rows), dtype=np.float32)
report = mojolearn.runtime_report()
print(json.dumps(dict(name=mojolearn.numeric_mode(), loaded=report["numeric_mode"],
                      vendor=mojolearn.vendor(), param=model.numeric_mode,
                      sha256=hashlib.sha256(out.tobytes()).hexdigest())))
"""


class BuiltInstall(unittest.TestCase):
    """On a built install (the CPU column of a GitHub runner is enough): a
    process whose default is named `fast` returns the bytes of an `identical`
    process and reads its name back as 'fast'."""

    def child(self, mode):
        import json
        env = dict(os.environ, MOJOLEARN_NUMERIC_MODE=mode)
        done = subprocess.run([sys.executable, "-c", _CHILD], env=env, capture_output=True, text=True, timeout=900)
        if done.returncode != 0:
            return None, done.stderr[-2000:]
        return json.loads(done.stdout.strip().splitlines()[-1]), ""

    def test_fast_default_runs_the_identical_set(self):
        identical, why = self.child("identical")
        if identical is None:
            self.skipTest("no built mojolearn in this environment: " + why[-300:])
        if sys.platform == "darwin":
            self.skipTest("Apple: `fast` names the fast set (unchanged); the routing under test is off Apple")
        fast, why = self.child("fast")
        self.assertIsNotNone(fast, why)
        if fast["loaded"] == "fast":
            self.skipTest("this install carries a fast set, which answers the name itself")
        self.assertEqual(identical["name"], "identical")
        self.assertEqual(fast["name"], "fast")
        self.assertEqual(fast["loaded"], "identical")
        self.assertIsNone(fast["param"])
        self.assertEqual(fast["vendor"], identical["vendor"])
        self.assertEqual(fast["sha256"], identical["sha256"])


if __name__ == "__main__":
    unittest.main()
