# SPDX-License-Identifier: Apache-2.0
"""The receipt of a fit and `reproduce`, at the Python boundary.

No binding and no GPU. The real `python/mojolearn/_reproduce.py` and `_check_both.py` are imported as
`mojolearn._reproduce` from a stub `mojolearn` package whose `__path__` is the source tree, and are handed a stub
install (`Install` below): a version, a numeric mode, a vendor, small estimators that fit in NumPy, and a door
with the native state-hash entry (hashlib stands in for the Mojo SHA-256, the same digest). What is held here: the
round trip, each kind of difference detected and named in order, the version message, the `fast` refusal, an
estimator without a fitted-state hash, the sabotage (one bit of X), the command's exit codes and its one JSON
line, and the public wiring (`mojolearn.reproduce`, `estimator.receipt`, the `reproduce` subcommand).

Runs under pytest and under `python3 -m unittest tools.tests.test_reproduce` (one process).
"""
import argparse
import ast
import contextlib
import hashlib
import importlib
import io
import json
import os
import re
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "python/mojolearn"
_MODULES = ("mojolearn", "mojolearn._reproduce", "mojolearn._check_both")


class Door:
    """The host core binding as the tool sees it: the activity record's two calls and the state-hash entry."""

    def __init__(self, with_state=True):
        self.hashed = []
        if with_state:
            self.state_file_sha256 = self._hash

    def activity_begin(self):
        return True

    def activity_end(self):
        return {}

    def _hash(self, path):
        self.hashed.append(os.path.basename(path))
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Ridge:
    """A small estimator with a deterministic fit, a save and plain parameters."""
    fits = 0
    drift = 0.0     # a class-level fault: a fit that is not the same fit twice

    def __init__(self, alpha=1.0, fit_intercept=True, sizes=(4, 2), numeric_mode=None):
        self.alpha, self.fit_intercept, self.sizes, self.numeric_mode = alpha, fit_intercept, sizes, numeric_mode

    def get_params(self, deep=True):
        return dict(alpha=self.alpha, fit_intercept=self.fit_intercept, sizes=self.sizes,
                    numeric_mode=self.numeric_mode)

    def fit(self, X, y):
        type(self).fits += 1
        X64 = np.asarray(X, dtype=np.float64)
        self.coef_ = ((X64.T @ np.asarray(y, dtype=np.float64)) / (self.alpha + len(X64))
                      + type(self).drift).astype(np.float32)
        return self

    def predict(self, X):
        return np.asarray(X, dtype=np.float32) @ self.coef_

    def save(self, path):
        Path(path).write_bytes(b"ridge\0" + self.coef_.tobytes())


class Scaler:
    """An estimator with no save: no fitted-state hash, an answer only."""

    def __init__(self, scale=2.0):
        self.scale = scale

    def get_params(self, deep=True):
        return dict(scale=self.scale)

    def fit(self, X):
        self.mean_ = np.asarray(X, dtype=np.float32).mean(axis=0)
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=np.float32) - self.mean_) * np.float32(self.scale)


class Clusterer:
    """An estimator that answers at fit: no save, no predict, `labels_`."""

    def __init__(self, cut=0.5):
        self.cut = cut

    def get_params(self, deep=True):
        return dict(cut=self.cut)

    def fit(self, X):
        self.labels_ = (np.asarray(X)[:, 0] > self.cut).astype(np.int32)
        return self


class Install(types.ModuleType):
    """A stub `mojolearn`: what `_reproduce` asks of the package."""

    def __init__(self, version="9.9.9", mode="identical", vendor="nvidia", with_state=True):
        super().__init__("mojolearn")
        self.__path__ = [str(PKG)]
        self.__package__ = "mojolearn"
        self.__version__ = version
        self._mode, self._vendor, self.door = mode, vendor, Door(with_state)
        self.Ridge, self.Scaler, self.Clusterer = Ridge, Scaler, Clusterer

    def numeric_mode(self):
        return self._mode

    def vendor(self):
        return self._vendor

    def activity_door(self):
        return self.door


def data(rows=64):
    rng = np.random.default_rng(7)
    X = rng.random((rows, 5), dtype=np.float32)
    y = (X @ np.arange(1, 6, dtype=np.float32)).astype(np.float32)
    return X, y


class Base(unittest.TestCase):
    def setUp(self):
        self._saved = {name: sys.modules.get(name) for name in _MODULES}
        for name in _MODULES:
            sys.modules.pop(name, None)
        self.ml = Install()
        sys.modules["mojolearn"] = self.ml
        self.rp = importlib.import_module("mojolearn._reproduce")
        self.cb = sys.modules["mojolearn._check_both"]
        Ridge.fits, Ridge.drift = 0, 0.0
        self._dir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._dir.name)
        self.X, self.y = data()

    def tearDown(self):
        self._dir.cleanup()
        for name, module in self._saved.items():
            sys.modules.pop(name, None)
            if module is not None:
                sys.modules[name] = module

    def fitted(self, **params):
        return Ridge(**params).fit(self.X, self.y)

    def receipt(self, **params):
        return self.rp.make_receipt(self.ml, self.fitted(**params), self.X, self.y)


class TestReceipt(Base):
    def test_what_a_receipt_holds(self):
        r = self.receipt(alpha=0.5)
        self.assertEqual(r["format"], "mojolearn.receipt.v1")
        self.assertEqual((r["mojolearn_version"], r["numeric_mode"], r["estimator"]), ("9.9.9", "identical", "Ridge"))
        self.assertEqual(list(r["parameters"]), sorted(r["parameters"]), "canonical order")
        self.assertEqual(r["parameters"]["alpha"], 0.5)
        self.assertEqual(r["parameters"]["sizes"], {"__tuple__": [4, 2]})
        self.assertEqual(r["data"]["X"]["shape"], [64, 5])
        self.assertEqual(r["data"]["X"]["dtype"], "float32")
        self.assertEqual(r["data"]["y"]["bytes"], self.y.nbytes)
        self.assertTrue(r["fitted_state"]["present"] and r["fitted_state"]["form"] == "save")
        self.assertTrue(r["predictions"]["present"] and r["predictions"]["method"] == "predict")
        self.assertEqual(r["recorded_on"]["vendor"], "nvidia")

    def test_the_hashes_are_the_forms_the_check_uses(self):
        est = self.fitted()
        r = self.rp.make_receipt(self.ml, est, self.X, self.y)
        self.assertEqual(r["data"]["X"]["sha256"], self.cb.describe_output(self.X)["sha256"])
        self.assertEqual(r["data"]["X"]["sha256"], hashlib.sha256(self.X.tobytes()).hexdigest())
        self.assertEqual(r["predictions"]["sha256"], self.cb.outputs_digest(self.cb.as_arrays(est.predict(self.X))))

    def test_the_state_hash_is_the_native_entry_over_the_saved_file(self):
        est = self.fitted()
        r = self.rp.make_receipt(self.ml, est, self.X, self.y)
        self.assertEqual(self.ml.door.hashed, ["model.npz"], "one native call, on the file the estimator's save wrote")
        self.assertEqual(r["fitted_state"]["sha256"], hashlib.sha256(b"ridge\0" + est.coef_.tobytes()).hexdigest())
        source = (PKG / "_reproduce.py").read_text()
        self.assertNotIn("hashlib", source, "the tool hashes nothing itself: the check's forms and the native entry")

    def test_the_same_fit_writes_the_same_receipt_bytes(self):
        a, b = self.tmp / "a.json", self.tmp / "b.json"
        self.rp.make_receipt(self.ml, self.fitted(), self.X, self.y, path=a)
        self.rp.make_receipt(self.ml, self.fitted(), self.X, self.y, path=b)
        self.assertEqual(a.read_bytes(), b.read_bytes())
        self.assertEqual(self.rp.load_receipt(a), json.loads(a.read_text()))

    def test_a_parameter_that_is_not_a_plain_value_is_refused_by_name(self):
        est = self.fitted()
        est.alpha = object()
        with self.assertRaises(self.rp.NotReproducible) as raised:
            self.rp.make_receipt(self.ml, est, self.X, self.y)
        self.assertIn("Ridge(alpha=...)", str(raised.exception))

    def test_fast_is_refused_in_words(self):
        est = self.fitted()
        self.ml._mode = "fast"
        with self.assertRaises(self.rp.NotReproducible) as raised:
            self.rp.make_receipt(self.ml, est, self.X, self.y)
        self.assertIn("only identical mode promises bits", str(raised.exception))
        self.ml._mode = "identical"
        with self.assertRaises(self.rp.NotReproducible) as raised:
            self.rp.make_receipt(self.ml, self.fitted(numeric_mode="fast"), self.X, self.y)
        self.assertIn("numeric_mode='fast'", str(raised.exception))

    def test_an_estimator_without_a_save_carries_the_prediction_hash_and_says_so(self):
        est = Scaler().fit(self.X)
        r = self.rp.make_receipt(self.ml, est, self.X)
        self.assertEqual(r["fitted_state"], dict(present=False, reason="Scaler has no save"))
        self.assertTrue(r["predictions"]["present"] and r["predictions"]["method"] == "transform")
        self.assertIsNone(r["data"]["y"])
        self.assertEqual(self.ml.door.hashed, [])

    def test_binaries_without_the_entry_give_no_state_hash_and_no_python_hash_in_its_place(self):
        self.ml.door = Door(with_state=False)
        r = self.rp.make_receipt(self.ml, self.fitted(), self.X, self.y)
        self.assertFalse(r["fitted_state"]["present"])
        self.assertIn("no fitted-state hash entry", r["fitted_state"]["reason"])


class TestReproduce(Base):
    def run_again(self, receipt, X=None, y=None, **kw):
        return self.rp.reproduce_with(self.ml, receipt, self.X if X is None else X, self.y if y is None else y, **kw)

    def test_round_trip_is_the_same_fit(self):
        path = self.tmp / "r.json"
        self.rp.make_receipt(self.ml, self.fitted(alpha=0.5), self.X, self.y, path=path)
        fits = Ridge.fits
        record = self.run_again(path)
        self.assertTrue(record["ran"] and record["same"])
        self.assertEqual((record["different"], record["first"]), ([], None))
        self.assertEqual(Ridge.fits, fits + 1, "one fit")
        self.assertTrue(all(record["parts"][p]["same"] for p in self.rp.PARTS))
        text = self.rp.format_reproduce(record)
        self.assertTrue(text.endswith("SAME: this fit equals the recorded fit, bit for bit."), text)
        self.assertEqual(self.rp.exit_code(record), 0)

    def test_sabotage_one_bit_of_X_is_caught_and_named(self):
        receipt = self.receipt()
        X = self.X.copy()
        X.view(np.uint32)[3, 2] ^= np.uint32(1 << 22)       # one bit of one value
        self.assertEqual(int(np.unpackbits((X.view(np.uint8) ^ self.X.view(np.uint8))).sum()), 1)
        record = self.run_again(receipt, X=X)
        self.assertTrue(record["ran"])
        self.assertFalse(record["same"])
        self.assertEqual(record["first"], "data")
        self.assertEqual(record["different"], ["data", "fitted_state", "predictions"])
        self.assertEqual(record["parts"]["data"]["X"]["differing"], ["bytes"])
        self.assertTrue(record["parts"]["data"]["y"]["same"])
        self.assertFalse(record["reproducibility_failure"])
        self.assertEqual(self.rp.exit_code(record), 1)
        text = self.rp.format_reproduce(record)
        self.assertRegex(text, r"data X\s+DIFFERENT\s+bytes: receipt sha256 \w{16}, now sha256 \w{16}")
        self.assertIn("DIFFERENT: data fingerprint, fitted state, predictions.", text)

    def test_shape_dtype_and_y_differences_are_named(self):
        receipt = self.receipt()
        record = self.run_again(receipt, X=self.X[:-1], y=self.y[:-1])
        self.assertEqual(record["parts"]["data"]["X"]["differing"], ["shape", "bytes"])
        self.assertEqual(record["parts"]["data"]["y"]["differing"], ["shape", "bytes"])
        record = self.run_again(receipt, X=self.X.astype(np.float64))
        self.assertEqual(record["parts"]["data"]["X"]["differing"], ["dtype", "bytes"])
        self.assertIn("dtype: receipt float32, now float64", self.rp.format_reproduce(record))
        y = self.y.copy()
        y[0] += 1
        record = self.run_again(receipt, y=y)
        self.assertTrue(record["parts"]["data"]["X"]["same"])
        self.assertEqual(record["parts"]["data"]["y"]["differing"], ["bytes"])
        self.assertEqual(record["first"], "data")

    def test_a_missing_y_is_a_presence_difference(self):
        receipt = self.rp.make_receipt(self.ml, Scaler().fit(self.X), self.X)
        record = self.rp.reproduce_with(self.ml, receipt, self.X, None)
        self.assertTrue(record["same"])
        receipt["data"]["y"] = self.rp.fingerprint(self.y, "y")
        record = self.rp.reproduce_with(self.ml, receipt, self.X, None)
        self.assertEqual(record["parts"]["data"]["y"]["differing"], ["presence"])
        self.assertEqual(record["first"], "data")

    def test_parameters_that_differ_are_named_first(self):
        receipt = self.receipt(alpha=1.0)
        record = self.run_again(receipt, params={"alpha": 2.0})
        self.assertEqual(record["first"], "parameters")
        self.assertEqual(record["parts"]["parameters"]["differing"], [dict(name="alpha", receipt="1.0", now="2.0")])
        self.assertTrue(record["parts"]["data"]["same"])
        self.assertFalse(record["reproducibility_failure"], "a parameter changed: not a reproducibility failure")
        self.assertIn("alpha: receipt 1.0, now 2.0", self.rp.format_reproduce(record))
        # the same tuple given as a JSON list is the same parameter
        self.assertTrue(self.run_again(receipt, params={"sizes": [4, 2]})["same"])
        # an estimator handed in is compared, class and parameters
        record = self.run_again(receipt, estimator=Ridge(alpha=1.0, fit_intercept=False))
        self.assertEqual([d["name"] for d in record["parts"]["parameters"]["differing"]], ["fit_intercept"])

    def test_another_version_stops_before_anything_is_fitted(self):
        receipt = self.receipt()
        fits = Ridge.fits
        self.ml.__version__ = "9.9.10"
        record = self.run_again(receipt)
        self.assertTrue(record["ran"])
        self.assertFalse(record["same"])
        self.assertEqual((record["different"], record["first"], record["stopped"]), (["version"], "version", "version"))
        self.assertEqual(list(record["parts"]), ["version"], "nothing else compared")
        self.assertEqual(Ridge.fits, fits, "no fit")
        text = self.rp.format_reproduce(record)
        for words in ("receipt 9.9.9", "this installation 9.9.10", "A receipt holds within one version",
                      "bits may change between versions", "mojolearn==9.9.9"):
            self.assertIn(words, text)
        self.assertEqual(self.rp.exit_code(record), 1)

    def test_fast_refuses_and_is_never_a_verdict(self):
        receipt = self.receipt()
        fits = Ridge.fits
        self.ml._mode = "fast"
        record = self.run_again(receipt)
        self.assertFalse(record["ran"])
        self.assertIsNone(record["same"])
        self.assertEqual(record["refused"], "fast")
        self.assertEqual(Ridge.fits, fits)
        self.assertEqual(self.rp.exit_code(record), 3)
        self.assertIn("REFUSED", self.rp.format_reproduce(record))
        self.ml._mode = "identical"
        record = self.run_again(receipt, params={"numeric_mode": "fast"})
        self.assertEqual((record["ran"], record.get("refused")), (False, "fast"))
        self.assertEqual(Ridge.fits, fits)
        record = self.run_again(dict(receipt, numeric_mode="fast"))
        self.assertEqual((record["ran"], record.get("refused")), (False, "fast"))

    def test_an_estimator_without_a_state_hash_is_compared_on_its_answer(self):
        receipt = self.rp.make_receipt(self.ml, Scaler().fit(self.X), self.X)
        record = self.rp.reproduce_with(self.ml, receipt, self.X)
        self.assertTrue(record["ran"] and record["same"])
        self.assertEqual(record["parts"]["fitted_state"]["compared"], False)
        self.assertTrue(record["parts"]["predictions"]["compared"] and record["parts"]["predictions"]["same"])
        self.assertRegex(self.rp.format_reproduce(record), r"fitted state\s+not compared\s+the receipt has no")
        X = self.X.copy()
        X[0, 0] += 1
        record = self.rp.reproduce_with(self.ml, receipt, X)
        self.assertEqual(record["different"], ["data", "predictions"])

    def test_an_estimator_that_answers_at_fit_is_compared_on_its_labels(self):
        receipt = self.rp.make_receipt(self.ml, Clusterer().fit(self.X), self.X)
        self.assertEqual(receipt["predictions"]["attribute"], "labels_")
        self.assertNotIn("method", receipt["predictions"])
        self.assertTrue(self.rp.reproduce_with(self.ml, receipt, self.X)["same"])
        record = self.rp.reproduce_with(self.ml, receipt, self.X, params={"cut": 0.25})
        self.assertEqual(record["different"], ["parameters", "predictions"])
        est = Clusterer().fit(self.X)
        del est.labels_
        with self.assertRaises(self.rp.NotReproducible) as raised:
            self.rp.make_receipt(self.ml, est, self.X)
        self.assertIn("nothing to record", str(raised.exception))

    def test_same_data_and_a_different_model_is_the_reproducibility_failure(self):
        receipt = self.receipt()
        Ridge.drift = 1e-3
        record = self.run_again(receipt)
        self.assertTrue(record["parts"]["parameters"]["same"] and record["parts"]["data"]["same"])
        self.assertEqual(record["different"], ["fitted_state", "predictions"])
        self.assertTrue(record["reproducibility_failure"])
        self.assertEqual(record["parts"]["predictions"]["first_differing_output"], 0)
        self.assertIn("the data fingerprint and the parameters are the same and the model differs",
                      self.rp.format_reproduce(record))
        self.assertEqual(self.rp.exit_code(record), 1)

    def test_a_recorded_state_hash_that_cannot_be_computed_here_is_not_called_the_same(self):
        receipt = self.receipt()
        self.ml.door = Door(with_state=False)
        self.cb._DOORS.clear()      # the door is found once per package: this is another installation
        record = self.run_again(receipt)
        self.assertFalse(record["ran"])
        self.assertIsNone(record["same"])
        self.assertIn("cannot be computed here", record["reason"])
        self.assertEqual(self.rp.exit_code(record), 4)

    def test_what_is_not_a_receipt_does_not_run(self):
        for bad in ({"format": "other"}, self.tmp / "absent.json"):
            record = self.run_again(bad)
            self.assertFalse(record["ran"])
            self.assertEqual(self.rp.exit_code(record), 4)
        record = self.run_again(dict(self.receipt(), estimator="Absent"))
        self.assertIn("not a class of this Mojolearn", record["reason"])


class TestCommand(Base):
    def files(self, X=None, y=None):
        receipt = self.tmp / "receipt.json"
        self.rp.make_receipt(self.ml, self.fitted(), self.X, self.y, path=receipt)
        np.save(self.tmp / "X.npy", self.X if X is None else X)
        np.save(self.tmp / "y.npy", self.y if y is None else y)
        return receipt

    def command(self, receipt, as_json=False, params=None, X="X.npy", y="y.npy"):
        args = argparse.Namespace(receipt=str(receipt), X=str(self.tmp / X), y=str(self.tmp / y) if y else None,
                                  params=params, json=as_json)
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = self.rp.cmd_reproduce(args, ml=self.ml)
        return code, out.getvalue(), err.getvalue()

    def test_exit_zero_for_the_same_fit_and_light_lines(self):
        code, out, _ = self.command(self.files())
        self.assertEqual(code, 0)
        lines = out.splitlines()
        self.assertEqual(lines[0], "reproduce Ridge")
        self.assertEqual([line.split()[0] for line in lines[1:7]],
                         ["version", "parameters", "data", "data", "fitted", "predictions"])
        self.assertTrue(lines[-1].startswith("SAME:"))

    def test_exit_one_for_one_changed_byte(self):
        X = self.X.copy()
        X.view(np.uint8)[5, 1] ^= 1
        code, out, _ = self.command(self.files(X=X))
        self.assertEqual(code, 1)
        self.assertIn("DIFFERENT: data fingerprint", out)

    def test_json_is_one_line_and_nothing_else(self):
        code, out, err = self.command(self.files(), as_json=True)
        self.assertEqual((code, err), (0, ""))
        self.assertEqual(out.count("\n"), 1)
        record = json.loads(out)
        self.assertEqual((record["format"], record["ran"], record["same"]), ("mojolearn.reproduce.v1", True, True))

    def test_an_npz_with_X_and_y_gives_both(self):
        receipt = self.files()
        np.savez(self.tmp / "table.npz", y=self.y, X=self.X)
        self.assertEqual(self.command(receipt, X="table.npz", y=None)[0], 0)

    def test_params_and_usage_and_unreadable_data(self):
        receipt = self.files()
        code, out, _ = self.command(receipt, params='{"alpha": 3.0}')
        self.assertEqual(code, 1)
        self.assertIn("parameters", out)
        code, out, err = self.command(receipt, params="[1]")
        self.assertEqual((code, out), (2, ""))
        self.assertIn("USAGE", err)
        code, out, _ = self.command(receipt, X="absent.npy")
        self.assertEqual(code, 4)
        self.assertIn("NOT RUN", out)
        self.ml._mode = "fast"
        self.assertEqual(self.command(receipt)[0], 3)


class TestWiring(unittest.TestCase):
    """The source statements that make the three names public."""

    def test_reproduce_is_exported_and_routed_lazily(self):
        tree = ast.parse((PKG / "__init__.py").read_text())

        def literal(name):
            return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
        self.assertIn("reproduce", literal("__all__"))
        self.assertEqual(literal("_LAZY_NAMES")["reproduce"], "._reproduce")
        imported = set()
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                imported.update(alias.name for alias in node.names)
        self.assertNotIn("_reproduce", imported, "import mojolearn does not load the tool")

    def test_the_receipt_method_is_a_door_onto_the_one_function(self):
        tree = ast.parse((PKG / "_mode.py").read_text())
        mixin = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CheckMixin")
        methods = [n for n in mixin.body if isinstance(n, ast.FunctionDef)]
        self.assertEqual([m.name for m in methods], ["check", "receipt"])
        method = methods[1]
        self.assertEqual([a.arg for a in method.args.args], ["self", "X", "y"])
        self.assertEqual([a.arg for a in method.args.kwonlyargs], ["path", "method"])
        module_level = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        self.assertFalse([n for n in module_level if "_reproduce" in ast.unparse(n)])
        # the real class statement, run as `mojolearn._mode` runs it, calls the one receipt function
        calls = []
        stub = types.ModuleType("mojolearn._reproduce")
        stub.receipt = lambda est, X, y=None, *, method=None, path=None: calls.append((est, X, y, method, path)) or "R"
        saved = {name: sys.modules.get(name) for name in ("mojolearn", "mojolearn._reproduce")}
        package = types.ModuleType("mojolearn")
        package.__path__ = [str(PKG)]
        sys.modules.update({"mojolearn": package, "mojolearn._reproduce": stub})
        try:
            namespace = {"__name__": "mojolearn._mode", "__package__": "mojolearn"}
            exec(compile(ast.Module(body=[mixin], type_ignores=[]), str(PKG / "_mode.py"), "exec"), namespace)
            est = namespace["CheckMixin"]()
            self.assertEqual(est.receipt("X", "y", path="p"), "R")
            self.assertEqual(calls, [(est, "X", "y", None, "p")])
        finally:
            for name, module in saved.items():
                sys.modules.pop(name, None)
                if module is not None:
                    sys.modules[name] = module

    def test_the_subcommand_is_registered_and_dispatches_lazily(self):
        source = (PKG / "__main__.py").read_text()
        tree = ast.parse(source)
        dispatch = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_reproduce_dispatch")
        self.assertIn("_reproduce.cmd_reproduce(args)", ast.unparse(dispatch))
        module_level = [ast.unparse(n) for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        self.assertFalse([line for line in module_level if "_reproduce" in line])
        # the real parser statements, run alone
        namespace = {"argparse": argparse}
        for name in ("_reproduce_dispatch", "_add_reproduce"):
            node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
            exec(compile(ast.Module(body=[node], type_ignores=[]), "__main__.py", "exec"), namespace)
        parser = argparse.ArgumentParser()
        namespace["_add_reproduce"](parser.add_subparsers(dest="command"))
        args = parser.parse_args(["reproduce", "r.json", "--X", "X.npy", "--y", "y.npy", "--json"])
        self.assertEqual((args.receipt, args.X, args.y, args.json, args.params),
                         ("r.json", "X.npy", "y.npy", True, None))
        self.assertIs(args.func, namespace["_reproduce_dispatch"])
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            parser.parse_args(["reproduce", "r.json"])      # --X is required

    def test_the_docstrings_are_user_documentation(self):
        mode = ast.parse((PKG / "_mode.py").read_text())
        mixin = next(n for n in mode.body if isinstance(n, ast.ClassDef) and n.name == "CheckMixin")
        method = next(n for n in mixin.body if isinstance(n, ast.FunctionDef) and n.name == "receipt")
        tool = ast.parse((PKG / "_reproduce.py").read_text())
        function = next(n for n in tool.body if isinstance(n, ast.FunctionDef) and n.name == "reproduce")
        for node in (method, function):
            doc = " ".join(ast.get_docstring(node).split())
            for section in ("Parameters", "Returns", "Examples"):
                self.assertIn(section, doc)
            self.assertIn("In `identical` mode, Mojolearn returns the same bits on Apple, NVIDIA and AMD GPUs", doc)
            for private in ("_reproduce", "_check_both", "_mode", "_verify"):
                self.assertIsNone(re.search(r"(?<![A-Za-z0-9])" + private, doc), private)
            self.assertNotIn("fast", doc, "the public docs speak of identical only")
            self.assertIsNone(re.search(r"20\d\d-\d\d-\d\d|Andrew|lane", doc))

    def test_the_native_entry_is_registered_under_the_name_the_tool_calls(self):
        binding = (ROOT / "bindings/_mojolearn_core_host.mojo").read_text()
        self.assertIn('module.def_function[state_file_sha256_binding]("state_file_sha256")', binding)
        self.assertIn("return PythonObject(sha256_file(String(py=path)))", binding)
        tool = ast.parse((PKG / "_reproduce.py").read_text())
        entry = next(ast.literal_eval(n.value) for n in tool.body if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == "STATE_ENTRY" for t in n.targets))
        self.assertEqual(entry, "state_file_sha256")
        self.assertIn('"state_file_sha256"', (PKG / "host_surface.py").read_text())


if __name__ == "__main__":
    unittest.main()
