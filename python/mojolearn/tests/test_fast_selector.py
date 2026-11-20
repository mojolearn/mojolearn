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
  * `binding` and `load_set` still name the fast set under 'fast' (a
    per-estimator `numeric_mode='fast'` is answered in the compiled doors,
    per door family, core/mode_answer.mojo MODE_FAMILIES_ANSWERED; a family
    not named there refuses it in every operation: `BuiltInstallFamilies`);
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


_FAMILY_CHILD = r"""
import hashlib, json, os, pickle, sys, tempfile
import mojolearn
import numpy as np
X = (np.arange(192, dtype=np.float32).reshape(48, 4) * np.float32(0.37)) % np.float32(5.0) + np.float32(0.25)
y_reg = np.ascontiguousarray(X[:, 0] * np.float32(2.0) - X[:, 1], dtype=np.float32)
y_cls = np.ascontiguousarray((X[:, 0] > X[:, 1]).astype(np.int32))
FAMILIES = json.loads(sys.argv[1])
if sys.argv[2] != "compiled":
    # The test's switch (never used by the package): a trial mask of door
    # families (bits of core/mode_answer.mojo) that answer `fast`, so one
    # built install lists the families that are whole under that set.
    from mojolearn import _native_backend_public as public
    public._NATIVE("connect_mode_answer", (int(sys.argv[2], 0),), public._CONTEXT)
READ = ("predict", "predict_proba", "predict_log_proba", "decision_function", "transform", "score_samples",
        "apply", "kneighbors")
def digest(value):
    if isinstance(value, tuple):
        return "+".join(digest(part) for part in value)
    if isinstance(value, dict):
        return "+".join(k + "=" + digest(value[k]) for k in sorted(value))
    if hasattr(value, "toarray"):
        value = value.toarray()
    value = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(value.tobytes() + str(value.dtype).encode() + str(value.shape).encode()).hexdigest()
def target(kind):
    return () if kind == "unsup" else ((y_cls,) if kind == "cls" else (y_reg,))
def operations(model, kind):
    # Every public operation the class has, after fit: name -> a call.
    ops = {}
    for name in READ:
        if hasattr(model, name):
            ops[name] = (lambda name=name: getattr(model, name)(X))
    if hasattr(model, "score"):
        ops["score"] = lambda: np.float64(model.score(X, *target(kind)))
    if hasattr(model, "inverse_transform") and hasattr(model, "transform"):
        ops["inverse_transform"] = lambda: model.inverse_transform(model.transform(X))
    if hasattr(model, "partial_fit"):
        def again():
            extra = dict(classes=np.unique(y_cls)) if kind == "cls" else {}
            model.partial_fit(X, *target(kind), **extra)
            return getattr(model, "transform" if hasattr(model, "transform") else "predict")(X)
        ops["partial_fit"] = again
    def after(restored):
        return getattr(restored, "transform" if hasattr(restored, "transform") else "predict")(X)
    ops["pickle"] = lambda: after(pickle.loads(pickle.dumps(model)))
    if hasattr(model, "save") and hasattr(type(model), "load"):
        def round_trip():
            with tempfile.TemporaryDirectory() as folder:
                path = os.path.join(folder, "model.npz")
                model.save(path)
                return after(type(model).load(path))
        ops["save_load"] = round_trip
    return ops
# lane fx-fast-mask, 2026-11-20: the neural, sequence, CNN, TSA and LINALG rows. Their public calls are not the
# estimator fit(X, y) form, so each is a recipe: build(word) -> (model, fit, {operation: call}, read), where `read`
# is the call a pickled or loaded copy answers. Same report and the same comparison as every other row.
S = np.ascontiguousarray(np.float32(10.0) + np.sin(np.arange(48, dtype=np.float32) * np.float32(0.4)) * np.float32(3.0)
                         + np.arange(48, dtype=np.float32) * np.float32(0.05), dtype=np.float32)
SEQ = np.ascontiguousarray(X.reshape(48, 2, 2))
IMG = np.ascontiguousarray(np.tile(X, (1, 16)).reshape(48, 1, 8, 8))
SPD = np.ascontiguousarray(X[:4].T @ X[:4] + np.eye(4, dtype=np.float32) * np.float32(4.0), dtype=np.float32)
IDS = np.ascontiguousarray((np.arange(64, dtype=np.int32) * 7) % 16, dtype=np.int32).reshape(4, 16)
def recipe(name, mode, extra):
    kw = dict(extra) if mode is None else dict(extra, numeric_mode=mode)
    M = mojolearn
    if name in ("ETS", "Theta", "CrostonClassic"):
        m = getattr(M, name)(**kw)
        return m, lambda: m.fit(S), {"predict": lambda: m.predict(4)}, lambda r: r.predict(4)
    if name == "ARIMA":
        m = M.ARIMA((1, 0, 0), **kw)
        return m, lambda: m.fit(S), {"predict": lambda: m.predict(), "forecast": lambda: m.forecast(4)}, lambda r: r.forecast(4)
    if name == "GARCH":
        m = M.GARCH(**kw)
        return m, lambda: m.fit(S), {"forecast": lambda: m.forecast(2)}, lambda r: r.forecast(2)
    if name == "Cholesky":
        m = M.Cholesky(**kw)
        return m, lambda: m.fit(SPD), {"solve": lambda: m.solve(SPD[:, :2].copy()), "logdet_": lambda: np.float64(m.logdet_)}, lambda r: r.solve(SPD[:, :2].copy())
    if name == "bootstrap":
        import importlib
        R = importlib.import_module("mojolearn.resample")
        return None, lambda: None, {"bootstrap": lambda: R.bootstrap(S, n_resamples=64, **kw).confidence_interval}, None
    if name in ("GRUClassifier", "RNNClassifier", "LSTMClassifier"):
        m = getattr(M, name)(**kw)
        return m, lambda: m.fit(SEQ, y_cls), {"predict": lambda: m.predict(SEQ), "predict_proba": lambda: m.predict_proba(SEQ)}, lambda r: r.predict(SEQ)
    if name in ("GRURegressor", "RNNRegressor", "LSTMRegressor"):
        m = getattr(M, name)(**kw)
        return m, lambda: m.fit(SEQ, y_reg), {"predict": lambda: m.predict(SEQ)}, lambda r: r.predict(SEQ)
    if name == "Sequential":
        m = M.Sequential(**kw).linear(4).relu().linear(1).compile(loss="mse", optimizer="sgd", lr=0.01)
        return m, lambda: m.fit(X, y_reg, epochs=2), {"predict": lambda: m.predict(X)}, lambda r: r.predict(X)
    if name == "CNNClassifier":
        m = M.CNNClassifier((1, 8, 8), **kw)
        return m, lambda: m.fit(IMG, y_cls), {"predict": lambda: m.predict(IMG), "predict_proba": lambda: m.predict_proba(IMG)}, lambda r: r.predict(IMG)
    if name == "Conv2d":
        m = M.Conv2d(1, 2, 3, **kw)
        return m, lambda: None, {"forward": lambda: m.forward(IMG[:4].copy())}, lambda r: r.forward(IMG[:4].copy())
    if name == "LayerNorm":
        m = M.LayerNorm(4, **kw)
        return m, lambda: None, {"forward": lambda: m.forward(X)}, lambda r: r.forward(X)
    if name == "Embedding":
        m = M.Embedding(16, 4, **kw)
        return m, lambda: None, {"forward": lambda: m.forward(IDS)}, lambda r: r.forward(IDS)
    if name == "SambaStack":
        from mojolearn.training import Generator
        m = M.SambaStack(M.SambaConfig(vocab=16, d_model=16, layers=("mamba3",)), generator=Generator(7), **kw)
        return m, lambda: None, {"train_step": lambda: m.train_step(IDS[:, :-1].copy(), IDS[:, 1:].copy()),
                                 "forward": lambda: m.forward(IDS)}, None
    raise KeyError(name)
def run_recipe(name, extra, mode):
    try:
        model, fit, ops, read = recipe(name, mode, extra)
    except Exception as error:
        return f"{type(error).__name__}: {error}"[:300], {}, None
    try:
        fit()
    except Exception as error:
        return f"{type(error).__name__}: {error}"[:300], {}, model
    if read is not None:
        ops["pickle"] = lambda: read(pickle.loads(pickle.dumps(model)))
        if hasattr(model, "save") and hasattr(type(model), "load"):
            def round_trip():
                with tempfile.TemporaryDirectory() as folder:
                    path = os.path.join(folder, "model.npz")
                    model.save(path)
                    return read(type(model).load(path))
            ops["save_load"] = round_trip
    done = {}
    for op, call in ops.items():
        try:
            done[op] = digest(call())
        except Exception as error:
            done[op] = "refused: " + f"{type(error).__name__}: {error}"[:300]
    return None, done, model if model is not None else type("Call", (), dict(numeric_mode=mode, get_params=lambda self: dict(numeric_mode=mode)))()
def run(name, kind, extra, mode):
    # (fit error or None, {operation: digest or "refused: ..."}, model)
    if kind == "recipe":
        return run_recipe(name, extra, mode)
    # Construction is inside the report too: a class with no `numeric_mode` parameter (ElasticNet, GaussianNB,
    # MaxAbsScaler, RobustScaler) raises TypeError for the word, which is its refusal; its `identical` arm is
    # built without the keyword. Before this, the TypeError escaped the child and the whole unit was skipped.
    # A model-selection row is named "<function>(<Estimator>)": the function (family MSEL) runs over an estimator
    # built with the word, and its one result is the row's one operation.
    call = None
    if name.endswith(")") and "(" in name:
        call, name = name[:-1].split("(", 1)
    cls = getattr(mojolearn, name)
    try:
        model = cls(numeric_mode=mode, **extra)
    except TypeError as error:
        if mode is not None or "numeric_mode" not in str(error):
            return f"{type(error).__name__}: {error}"[:300], {}, None
        model = cls(**extra)
    except Exception as error:
        return f"{type(error).__name__}: {error}"[:300], {}, None
    if call is not None:
        try:
            return None, {call: digest(getattr(mojolearn, call)(model, X, *target(kind), cv=3))}, model
        except Exception as error:
            return f"{type(error).__name__}: {error}"[:300], {}, model
    try:
        model.fit(X, *target(kind))
    except Exception as error:
        return f"{type(error).__name__}: {error}"[:300], {}, model
    done = {}
    for op, call in operations(model, kind).items():
        try:
            done[op] = digest(call())
        except Exception as error:
            done[op] = "refused: " + f"{type(error).__name__}: {error}"[:300]
    return None, done, model
rows = {}
for name, kind, extra in FAMILIES:
    failed, want, _ = run(name, kind, extra, None)
    if failed is not None:
        rows[name] = dict(status="unavailable", why=failed)
        continue
    failed, got, model = run(name, kind, extra, "fast")
    if failed is not None and model is None and failed.startswith("TypeError") and "numeric_mode" in failed:
        # The class has no `numeric_mode` parameter at all: no door of any family answers for it, so it says
        # nothing about whether its family is whole (the summary leaves it out of the family's word).
        rows[name] = dict(status="no_parameter", why=failed)
        continue
    if failed is not None:
        # The family refuses the word at fit: nothing of it runs under `fast`.
        rows[name] = dict(status="refused", why=failed)
        continue
    ops = {}
    for op, expected in want.items():
        if expected.startswith("refused: "):
            continue  # not runnable under `identical` either on this data
        value = got.get(op, "refused: operation missing")
        ops[op] = "same" if value == expected else ("refused" if value.startswith("refused: ") else "different")
        if ops[op] != "same":
            ops[op] += " " + value[:200]
    whole = all(value == "same" for value in ops.values())
    params = model.get_params() if hasattr(model, "get_params") else dict(numeric_mode=getattr(model, "numeric_mode", None))
    rows[name] = dict(status="answered" if whole else "mixed", ops=ops, attribute=getattr(model, "numeric_mode", None),
                      param=params.get("numeric_mode"))
print(json.dumps(dict(answer=mojolearn._backend._FAST_ANSWER, vendor=mojolearn.vendor(), rows=rows)))
"""

#: One estimator per door family that has a mode-answer step: (class, kind, constructor extras).
FAMILIES = [
    ["RandomForestClassifier", "cls", {"n_estimators": 4, "random_state": 0}],
    ["RandomForestRegressor", "reg", {"n_estimators": 4, "random_state": 0}],
    ["ExtraTreesClassifier", "cls", {"n_estimators": 4, "random_state": 0}],
    ["GradientBoostingRegressor", "reg", {"n_estimators": 4}],
    ["GradientBoostingClassifier", "cls", {"n_estimators": 4}],
    ["IsolationForest", "unsup", {"n_estimators": 4, "random_state": 0}],
    ["StandardScaler", "unsup", {}],
    ["MinMaxScaler", "unsup", {}],
    ["MaxAbsScaler", "unsup", {}],
    ["RobustScaler", "unsup", {}],
    ["PCA", "unsup", {"n_components": 2}],
    ["IncrementalPCA", "unsup", {"n_components": 2}],
    ["TruncatedSVD", "unsup", {"n_components": 2}],
    ["KernelPCA", "unsup", {"n_components": 2}],
    ["NMF", "unsup", {"n_components": 2}],
    ["PLSRegression", "reg", {"n_components": 2}],
    ["LinearRegression", "reg", {}],
    ["Ridge", "reg", {}],
    ["LogisticRegression", "cls", {}],
    ["ElasticNet", "reg", {}],
    ["SGDClassifier", "cls", {"random_state": 0}],
    ["KernelRidge", "reg", {}],
    ["KNeighborsClassifier", "cls", {}],
    ["KNeighborsRegressor", "reg", {}],
    ["KMeans", "unsup", {"n_clusters": 2, "random_state": 0}],
    ["GaussianMixture", "unsup", {"n_components": 2, "random_state": 0}],
    ["SVC", "cls", {}],
    ["GaussianNB", "cls", {}],
]

#: The walk (lane fx-fast-per-estimator, 2026-11-19): the public estimators with a `numeric_mode` parameter that
#: FAMILIES does not cover, run only by the trial (tools/ops/fast_trial.py, or MOJOLEARN_FAST_ESTIMATORS naming one):
#: their rows show which estimators the shared doors (RESOLVER, PROGRAM, METADATA) leave `mixed`. An estimator whose
#: default construction does not fit the trial's small data reports `unavailable`, which is not a failure.
WALK = [
    ["DecisionTreeClassifier", "cls", {}],
    ["AdaBoostClassifier", "cls", {"n_estimators": 4, "random_state": 0}],
    ["BaggingClassifier", "cls", {"n_estimators": 4, "random_state": 0}],
    ["DARTClassifier", "cls", {"n_estimators": 4}],
    ["RidgeClassifier", "cls", {}],
    ["Perceptron", "cls", {"random_state": 0}],
    ["PassiveAggressiveClassifier", "cls", {"random_state": 0}],
    ["LogisticRegressionCV", "cls", {}],
    ["LinearSVC", "cls", {}],
    ["NearestCentroid", "cls", {}],
    ["GaussianProcessClassifier", "cls", {}],
    ["LabelPropagation", "cls", {}],
    ["LabelSpreading", "cls", {}],
    ["Lasso", "reg", {}],
    ["LassoCV", "reg", {}],
    ["ElasticNetCV", "reg", {}],
    ["RidgeCV", "reg", {}],
    ["BayesianRidge", "reg", {}],
    ["ARDRegression", "reg", {}],
    ["HuberRegressor", "reg", {}],
    ["SGDRegressor", "reg", {"random_state": 0}],
    ["PassiveAggressiveRegressor", "reg", {"random_state": 0}],
    ["Lars", "reg", {}],
    ["LassoLars", "reg", {}],
    ["QuantileRegressor", "reg", {}],
    ["TweedieRegressor", "reg", {}],
    ["QNRegressor", "reg", {}],
    ["SVR", "reg", {}],
    ["LinearSVR", "reg", {}],
    ["DecisionTreeRegressor", "reg", {}],
    ["ExtraTreesRegressor", "reg", {"n_estimators": 4, "random_state": 0}],
    ["AdaBoostRegressor", "reg", {"n_estimators": 4, "random_state": 0}],
    ["BaggingRegressor", "reg", {"n_estimators": 4, "random_state": 0}],
    ["DARTRegressor", "reg", {"n_estimators": 4}],
    ["GaussianProcessRegressor", "reg", {}],
    ["PLSCanonical", "reg", {"n_components": 1}],
    ["MiniBatchKMeans", "unsup", {"n_clusters": 2, "random_state": 0}],
    ["BisectingKMeans", "unsup", {"n_clusters": 2, "random_state": 0}],
    ["BayesianGaussianMixture", "unsup", {"n_components": 2, "random_state": 0}],
    ["MeanShift", "unsup", {}],
    ["AffinityPropagation", "unsup", {}],
    ["OPTICS", "unsup", {}],
    ["DBSCAN", "unsup", {}],
    ["HDBSCAN", "unsup", {}],
    ["KernelDensity", "unsup", {}],
    ["FastICA", "unsup", {"n_components": 2, "random_state": 0}],
    ["FactorAnalysis", "unsup", {"n_components": 2}],
    ["SparsePCA", "unsup", {"n_components": 2, "random_state": 0}],
    ["DictionaryLearning", "unsup", {"n_components": 2, "random_state": 0}],
    ["GaussianRandomProjection", "unsup", {"n_components": 2, "random_state": 0}],
    ["SparseRandomProjection", "unsup", {"n_components": 2, "random_state": 0}],
    ["Isomap", "unsup", {"n_components": 2}],
    ["LocallyLinearEmbedding", "unsup", {"n_components": 2}],
    ["MDS", "unsup", {"n_components": 2, "random_state": 0}],
    ["MinCovDet", "unsup", {"random_state": 0}],
    ["EllipticEnvelope", "unsup", {"random_state": 0}],
    ["Nystroem", "unsup", {"n_components": 4, "random_state": 0}],
    ["RBFSampler", "unsup", {"n_components": 4, "random_state": 0}],
    ["NearestNeighbors", "unsup", {}],
    ["RadiusNeighbors", "unsup", {}],
    ["LocalOutlierFactor", "unsup", {}],
    ["OneClassSVM", "unsup", {}],
    ["SGDOneClassSVM", "unsup", {"random_state": 0}],
    ["UMAP", "unsup", {"random_state": 0}],
    ["TSNE", "unsup", {"random_state": 0}],
    ["KNNImputer", "unsup", {}],
    ["RandomTreesEmbedding", "unsup", {"n_estimators": 4, "random_state": 0}],
    # lane fx-fast-rest, 2026-11-20: the families the trial did not cover. MSEL: model selection over an
    # answered estimator; XSEQ: the sequence-model MLPs (off: expected refused, never mixed).
    ["cross_val_score(Ridge)", "reg", {}],
    ["cross_val_score(LogisticRegression)", "cls", {}],
    ["cross_val_score(KNeighborsClassifier)", "cls", {}],
    ["cross_val_score(GaussianProcessRegressor)", "reg", {}],
    ["MLPClassifier", "cls", {"random_state": 0, "max_iter": 20}],
    ["MLPRegressor", "reg", {"random_state": 0, "max_iter": 20}],
    # lane fx-fast-mask, 2026-11-20: the families that were off (TSA, NEURAL, XCNN, XSEQ, LINALG), one recipe per
    # public form (the `recipe` function of the child): forecasters, recurrent nets, Sequential, the CNN classifier
    # and layers, the embedding, the Samba stack, ARIMA, Cholesky and the resampling functions.
    ["ARIMA", "recipe", {}],
    ["Cholesky", "recipe", {}],
    ["bootstrap", "recipe", {}],
    ["ETS", "recipe", {}],
    ["Theta", "recipe", {}],
    ["CrostonClassic", "recipe", {}],
    ["GARCH", "recipe", {}],
    ["GRUClassifier", "recipe", {"random_state": 0, "max_epochs": 2}],
    ["LSTMRegressor", "recipe", {"random_state": 0, "max_epochs": 2}],
    ["RNNClassifier", "recipe", {"random_state": 0, "max_epochs": 2}],
    ["Sequential", "recipe", {}],
    ["CNNClassifier", "recipe", {"max_iter": 2}],
    ["Conv2d", "recipe", {}],
    ["LayerNorm", "recipe", {}],
    ["Embedding", "recipe", {}],
    ["SambaStack", "recipe", {}],
]

#: The door family (core/mode_answer.mojo MODE_FAMILY_<name>) whose answer step each estimator's doors hold;
#: "none" for a family with no answer step at all (always refused: maxabs, robust, kmeans, mixture, svm). The
#: trial report (tools/ops/fast_trial.py) rolls the rows up by this name.
ESTIMATOR_FAMILY = {
    "RandomForestClassifier": "FOREST", "RandomForestRegressor": "FOREST", "ExtraTreesClassifier": "FOREST",
    "GradientBoostingRegressor": "GBDT", "GradientBoostingClassifier": "GBDT", "IsolationForest": "IFOREST",
    "StandardScaler": "SCALER", "MinMaxScaler": "SCALER", "MaxAbsScaler": "SCALER", "RobustScaler": "SCALER",
    "PCA": "PCA", "IncrementalPCA": "IPCA", "TruncatedSVD": "TSVD", "KernelPCA": "KPCA", "NMF": "NMF",
    "PLSRegression": "PLS", "LinearRegression": "LINEAR", "Ridge": "LINEAR", "LogisticRegression": "LINEAR",
    "ElasticNet": "LINEAR", "SGDClassifier": "LINEAR", "GaussianNB": "PREP", "KernelRidge": "KERNEL",
    "KNeighborsClassifier": "NEIGHBORS", "KNeighborsRegressor": "NEIGHBORS", "KMeans": "none",
    "GaussianMixture": "MIXTURE", "SVC": "none",
    "DecisionTreeClassifier": "FOREST", "DecisionTreeRegressor": "FOREST", "AdaBoostClassifier": "FOREST", "AdaBoostRegressor": "FOREST", "BaggingClassifier": "FOREST",
    "BaggingRegressor": "FOREST", "ExtraTreesRegressor": "FOREST", "RandomTreesEmbedding": "FOREST", "DARTClassifier": "GBDT", "DARTRegressor": "GBDT",
    "RidgeClassifier": "LINEAR", "Perceptron": "LINEAR", "PassiveAggressiveClassifier": "LINEAR", "LogisticRegressionCV": "LINEAR", "Lasso": "LINEAR",
    "LassoCV": "LINEAR", "ElasticNetCV": "LINEAR", "RidgeCV": "LINEAR", "BayesianRidge": "LINEAR", "ARDRegression": "LINEAR",
    "HuberRegressor": "LINEAR", "SGDRegressor": "LINEAR", "PassiveAggressiveRegressor": "LINEAR", "Lars": "LINEAR", "LassoLars": "LINEAR",
    "QuantileRegressor": "LINEAR", "TweedieRegressor": "LINEAR", "QNRegressor": "LINEAR", "SGDOneClassSVM": "LINEAR", "NearestNeighbors": "NEIGHBORS",
    "RadiusNeighbors": "NEIGHBORS", "GaussianProcessClassifier": "GP", "GaussianProcessRegressor": "GP", "BayesianGaussianMixture": "MIXTURE", "PLSCanonical": "PLS",
    "FastICA": "DECOMP_KIT", "FactorAnalysis": "DECOMP_KIT", "SparsePCA": "DECOMP_KIT", "DictionaryLearning": "DECOMP_KIT", "GaussianRandomProjection": "DECOMP_KIT",
    "SparseRandomProjection": "DECOMP_KIT", "Isomap": "DECOMP_KIT", "LocallyLinearEmbedding": "DECOMP_KIT", "MDS": "DECOMP_KIT", "MinCovDet": "DECOMP_KIT",
    "EllipticEnvelope": "DECOMP_KIT", "Nystroem": "KERNEL", "RBFSampler": "KERNEL", "UMAP": "SPECTRAL", "TSNE": "SPECTRAL",
    "LinearSVC": "none", "NearestCentroid": "none", "LabelPropagation": "none", "LabelSpreading": "none", "SVR": "none",
    "LinearSVR": "none", "MiniBatchKMeans": "none", "BisectingKMeans": "none", "MeanShift": "none", "AffinityPropagation": "none",
    "OPTICS": "none", "DBSCAN": "none", "HDBSCAN": "CLUSTER", "KernelDensity": "none", "LocalOutlierFactor": "none",
    "OneClassSVM": "none", "KNNImputer": "none",
    "cross_val_score(Ridge)": "MSEL", "cross_val_score(LogisticRegression)": "MSEL",
    "cross_val_score(KNeighborsClassifier)": "MSEL", "cross_val_score(GaussianProcessRegressor)": "MSEL",
    "MLPClassifier": "XSEQ", "MLPRegressor": "XSEQ",
    "ARIMA": "TSA", "Cholesky": "LINALG", "bootstrap": "LINALG", "ETS": "XSEQ", "Theta": "XSEQ",
    "CrostonClassic": "XSEQ", "GARCH": "XSEQ", "GRUClassifier": "XSEQ", "LSTMRegressor": "XSEQ",
    "RNNClassifier": "XSEQ", "Sequential": "NEURAL", "CNNClassifier": "XCNN", "Conv2d": "XCNN",
    "LayerNorm": "NEURAL", "Embedding": "NEURAL", "SambaStack": "NEURAL",
}


def selected_families():
    """FAMILIES, or only the estimators named in MOJOLEARN_FAST_ESTIMATORS (comma separated; a switch of this
    test, never of the package): tools/ops/fast_trial.py runs one estimator per process across the cores."""
    names = [n.strip() for n in os.environ.get("MOJOLEARN_FAST_ESTIMATORS", "").split(",") if n.strip()]
    if not names:
        return FAMILIES
    unknown = sorted(set(names) - {row[0] for row in FAMILIES + WALK})
    if unknown:
        raise ValueError(f"MOJOLEARN_FAST_ESTIMATORS names estimators not in FAMILIES or WALK: {unknown}")
    return [row for row in FAMILIES + WALK if row[0] in names]


class BuiltInstallFamilies(unittest.TestCase):
    """On a built install off Apple with no fast set, `Estimator(numeric_mode='fast')`
    answers the same way in EVERY public operation of its family. Either the
    family is answered: fit, each other operation the class has (predict,
    predict_proba, decision_function, transform, inverse_transform, score,
    score_samples, partial_fit), a pickle and a save/load round trip all run,
    their result bytes equal the `numeric_mode=None` (identical) ones, and the
    attribute and `get_params()` read back 'fast'. Or the family refuses the
    word at fit, as an install with no fast set always has. A family that fits
    under `fast` and then refuses or differs in another call fails here.

    `MOJOLEARN_FAST_ANSWER_TRIAL_FAMILIES=<mask>` (a switch of this test, not
    of the package; `0xfffffff` is every door family, bits in
    core/mode_answer.mojo) answers `fast` for that set of families, so the run
    lists which are whole under it; `MOJOLEARN_FAST_LEDGER=<path>` keeps the
    report. A family is added to MODE_FAMILIES_ANSWERED only when a run with
    exactly the candidate set shows no `mixed` row."""

    def test_per_estimator_fast_answers_every_operation_alike(self):
        import json
        if sys.platform == "darwin":
            self.skipTest("Apple: `fast` names the fast set (unchanged); the answer under test is off Apple")
        every = os.environ.get("MOJOLEARN_FAST_ANSWER_TRIAL_FAMILIES", "compiled")
        chosen = selected_families()
        done = subprocess.run([sys.executable, "-c", _FAMILY_CHILD, json.dumps(chosen), every],
                              env=dict(os.environ, MOJOLEARN_NUMERIC_MODE="identical"),
                              capture_output=True, text=True, timeout=1800)
        ledger = os.environ.get("MOJOLEARN_FAST_LEDGER")
        if done.returncode != 0:
            if not ledger:
                self.skipTest("no built mojolearn in this environment: " + done.stderr[-300:])
            # A trial asked for a report: the child's failure IS the report (a skip hid it and left no ledger).
            why = f"child rc={done.returncode}: " + " | ".join(done.stderr.strip().splitlines()[-6:])[-600:]
            with open(ledger, "w") as out:
                json.dump(dict(answer=None, vendor=None, families=every,
                               rows={name: dict(status="crashed", why=why, family=ESTIMATOR_FAMILY.get(name, "?"))
                                     for name, _, _ in chosen}), out, indent=1, sort_keys=True)
            self.fail(why)
        report = json.loads(done.stdout.strip().splitlines()[-1])
        for name, row in report["rows"].items():
            row["family"] = ESTIMATOR_FAMILY.get(name, "?")
        if ledger:
            with open(ledger, "w") as out:
                json.dump(dict(report, families=every), out, indent=1, sort_keys=True)
        if report["answer"] != "identical":
            self.skipTest("this install carries a fast set, which answers the word itself")
        ran = 0
        for name, row in sorted(report["rows"].items()):
            with self.subTest(estimator=name):
                if row["status"] in ("unavailable", "no_parameter"):
                    continue  # not runnable under `identical` either on this install
                ran += 1
                self.assertIn(row["status"], ("answered", "refused"), row.get("ops", ""))
                if row["status"] == "answered":
                    self.assertEqual((row["attribute"], row["param"]), ("fast", "fast"))
        self.assertGreater(ran, 0, report["rows"])


if __name__ == "__main__":
    unittest.main()
