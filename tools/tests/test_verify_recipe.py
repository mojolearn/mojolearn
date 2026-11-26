# SPDX-License-Identifier: Apache-2.0
"""`verify --recipe`: a user's algorithm fitted from scratch on every device, at the Python boundary.

No binding and no GPU. A stub `mojolearn` package whose `__path__` is the source tree loads the real
`_verify_recipe.py` and `_check_both.py`; the stub estimator and Sequential stand in for the product and take
`device=` as the product does (`vendor_used` answers 'cpu' for device='cpu'); both devices fit in this process.
What is held here: the recipe format and its refusals in words (fast mode, a device or seed in params, a data
file with another hash), the generators' determinism, `recipe(model, ...)` from a live object, IDENTICAL over
equal devices and DIVERGENT naming the first differing value, the record, `--against` and its refusals (another
version, another recipe, other data), and the self-test's four controls. The real fits are the box run of the
shipped recipes (`verify --recipe example:<name>`).
"""
import copy
import hashlib
import importlib
import json
import re
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "python/mojolearn"


class Toy:
    """A stub estimator: deterministic answers; `nudge` moves one prediction by one ulp."""
    nudge = False

    def __init__(self, alpha=1.0, random_state=0, numeric_mode=None, device=None):
        self.alpha, self.random_state, self.numeric_mode, self.device = alpha, random_state, numeric_mode, device

    def get_params(self, deep=True):
        return dict(alpha=self.alpha, random_state=self.random_state, numeric_mode=self.numeric_mode,
                    device=self.device)

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float32)
        self.coef_ = (X.mean(axis=0) * np.float32(self.alpha) + np.float32(self.random_state)).astype(np.float32)
        return self

    def vendor_used(self):
        return "cpu" if self.device == "cpu" else "cuda"

    def predict(self, X):
        out = np.zeros(len(X), dtype=np.float32)
        for j, c in enumerate(self.coef_):
            out = out + np.asarray(X, dtype=np.float32)[:, j] * c
        if self.nudge:
            out[3] = np.nextafter(out[3], np.inf, dtype=np.float32)
        return out

    def score(self, X, y):
        return float(np.mean(self.predict(X)))

    @property
    def digest_(self):
        return hashlib.sha256(self.coef_.tobytes()).hexdigest()


class Seq:
    """A stub Sequential: records its layer calls in `_layers` and its compile options in `_options`."""

    def __init__(self, numeric_mode=None, *, input_shape=None, device=None):
        self.numeric_mode, self.input_shape, self.device = numeric_mode, input_shape, device
        self._layers, self._options = [], None

    def _add(self, code, *args):
        self._layers.append((code, args[0] if args else 0, 0, 0.0, tuple(list(args[:6]) + [0] * (6 - len(args[:6])))))
        return self

    def linear(self, units, bias=True):
        return self._add(1, units)

    def relu(self):
        return self._add(2)

    def residual_block(self, body, shortcut=None):
        self._layers.extend(body._layers)
        return self._add(8, -1)

    def compile(self, loss="mse", optimizer="adam", lr=1e-3):
        self._options = dict(loss=loss, opt={"sgd": 0, "adam": 1}[optimizer], lr=float(lr))
        return self

    def fit(self, X, y, *, epochs=1, batch_size=None, shuffle=None, seed=0):
        words = repr((self._layers, sorted(self._options.items()), epochs, batch_size, shuffle, seed))
        self.params_ = np.frombuffer(hashlib.sha256(words.encode()).digest(), dtype=np.float32).copy()
        return self

    def predict(self, X):
        return (np.asarray(X, dtype=np.float32)[:, :1] * self.params_[0]).astype(np.float32)

    def vendor_used(self):
        return "cpu" if self.device == "cpu" else "cuda"

    @property
    def digest_(self):
        return hashlib.sha256(self.params_.tobytes()).hexdigest()


def stub(vendor="cuda", mode="identical", version="0.0.0+test", nudge=False):
    ml = types.ModuleType("mojolearn")
    ml.__path__ = [str(PKG)]
    ml.__package__ = "mojolearn"
    ml.__version__ = version
    ml.vendor = lambda: vendor
    ml.numeric_mode = lambda: mode
    ml.runtime_report = lambda: dict(version=version, numeric_mode=mode, vendor=vendor, bindings={})
    ml.Toy = type("Toy", (Toy,), {"nudge": nudge})
    ml.Sequential = Seq
    return ml


@pytest.fixture
def vr(monkeypatch):
    """The real module under a stub package; both devices fit here, through device=."""
    for name in ("mojolearn._verify_recipe", "mojolearn._check_both"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    parent = stub()
    # A real `mojolearn` imported earlier in this worker (pytest -n shares workers across test files) leaves its
    # family loader on the import path; it would connect this stub package's modules as product doors. Suspend it
    # for the stub's imports (lane nn-integrate, 2026-11-23).
    families = sys.modules.get("mojolearn._families")
    if families is not None:
        monkeypatch.setattr(families, "_SUSPENDED", [True])   # read as a module global at each import
    monkeypatch.setitem(sys.modules, "mojolearn", parent)
    module = importlib.import_module("mojolearn._verify_recipe")
    monkeypatch.setattr(module.cb, "side_environment",
                        lambda ml, side, **_: dict(side=side, device=f"stub {side}", mode="identical"))
    devices = {"gpu_nudge": False}
    # the GPU side's estimator moves one prediction by one ulp when asked
    parent.Toy = type("Toy", (Toy,), {"nudge": property(lambda self: self.device == "gpu" and devices["gpu_nudge"])})
    module._devices = devices
    module._parent = parent
    return module


def toy_recipe(**outputs):
    return {"format": "mojolearn.recipe.v1", "model": {"estimator": "Toy", "params": {"alpha": 2.0}}, "seed": 3,
            "fit": {}, "data": {"generator": "regression", "args": {"rows": 64, "features": 4, "test_rows": 16,
                                                                     "seed": 5}},
            "outputs": outputs or {"digest": True, "predict": {"method": "predict", "on": "test"}, "score": True}}


def run_cli(vr, monkeypatch, capsys, recipe_arg, **flags):
    args = types.SimpleNamespace(recipe=recipe_arg, against=None, record_out=None, tamper_test=False, json=False)
    vars(args).update(flags)
    code = vr.cmd_recipe(args)
    return code, capsys.readouterr().out


# ---------------------------------------------------------------- data

def test_generators_are_deterministic_and_split_test_rows(vr):
    for name, args in (("regression", {}), ("classification", {"classes": 3}), ("blobs", {}),
                       ("categorical", {"levels": 3}), ("sequence", {"shape": [4, 2]})):
        a = vr.generate(name, dict(args, rows=40, test_rows=8, seed=1))
        b = vr.generate(name, dict(args, rows=40, test_rows=8, seed=1))
        c = vr.generate(name, dict(args, rows=40, test_rows=8, seed=2))
        assert vr.data_sha256(a) == vr.data_sha256(b) != vr.data_sha256(c), name
        assert len(a["X"]) == 40 and len(a["X_test"]) == 8 and len(a["y"]) == 40, name
    cat = vr.generate("categorical", dict(rows=50, categorical=2, levels=3, seed=0))
    assert set(np.unique(cat["X"][:, :2]).tolist()) <= {0.0, 1.0, 2.0}


def test_unknown_generator_and_arguments_are_refused_in_words(vr):
    with pytest.raises(vr.Refused, match="no built-in generator"):
        vr.generate("nope", {})
    with pytest.raises(vr.Refused, match="does not take"):
        vr.generate("regression", {"colour": 1})


def test_data_file_with_another_hash_is_refused(vr, tmp_path):
    path = tmp_path / "d.npz"
    np.savez(path, X=np.ones((4, 2), np.float32), y=np.zeros(4, np.float32))
    good = vr.file_sha256(path)
    assert vr.load_data({"file": "d.npz", "sha256": good}, str(tmp_path))["X"].shape == (4, 2)
    with pytest.raises(vr.Refused, match="these are other data"):
        vr.load_data({"file": "d.npz", "sha256": "0" * 64}, str(tmp_path))
    with pytest.raises(vr.Refused, match="does not exist"):
        vr.load_data({"file": "missing.npz", "sha256": good}, str(tmp_path))


# ---------------------------------------------------------------- the recipe

@pytest.mark.parametrize("change, words", [
    (lambda r: r["model"]["params"].update(numeric_mode="fast"), "no promise about bits"),
    (lambda r: r["model"]["params"].update(device="cpu"), "leaves device out"),
    (lambda r: r["model"]["params"].update(random_state=1), "the seed is the recipe's own"),
    (lambda r: r.update(outputs={}), "hashes nothing"),
    (lambda r: r.update(format="other"), "not a recipe"),
    (lambda r: r.update(data={}), "generator or a file"),
    (lambda r: r["outputs"]["predict"].update(on="elsewhere"), "'train' or 'test'"),
    (lambda r: r.update(fit={"seed": 1}), "seed twice"),
])
def test_recipe_refusals_are_words_not_verdicts(vr, change, words):
    r = toy_recipe()
    change(r)
    with pytest.raises(vr.Refused, match=words):
        vr.check_recipe(r)


def test_sequential_fast_recipe_is_refused(vr):
    r = copy.deepcopy(vr.EXAMPLES["seq-recurrent"])
    r["model"]["sequential"]["numeric_mode"] = "fast"
    with pytest.raises(vr.Refused, match="no promise about bits"):
        vr.check_recipe(r)


def test_shipped_recipes_are_valid(vr):
    assert {"ridge", "logistic", "kmeans", "gbdt-categorical", "seq-transformer", "seq-mamba2", "seq-recurrent",
            "seq-conv1d-residual"} <= set(vr.EXAMPLES)
    for name, r in vr.EXAMPLES.items():
        vr.check_recipe(r)
        json.dumps(r)
        assert vr.load_recipe("example:" + name)[0] == r


def test_canonical_hash_ignores_where_the_data_file_is(vr):
    a = toy_recipe()
    a["data"] = {"file": "here/d.npz", "sha256": "AB" * 32}
    b = copy.deepcopy(a)
    b["data"]["file"] = "/elsewhere/d.npz"
    assert vr.recipe_sha256(a) == vr.recipe_sha256(b)
    b["data"]["sha256"] = "cd" * 32
    assert vr.recipe_sha256(a) != vr.recipe_sha256(b)


def test_recipe_from_a_live_estimator(vr, tmp_path):
    model = vr._parent.Toy(alpha=0.5, random_state=9, device="cpu")
    X = np.arange(24, dtype=np.float32).reshape(6, 4)
    with pytest.raises(vr.Refused, match="needs path"):
        vr.recipe(model, X, X[:, 0])
    r = vr.recipe(model, X, X[:, 0], score=True, path=str(tmp_path / "toy.json"))
    assert r["model"] == {"estimator": "Toy", "params": {"alpha": 0.5, "numeric_mode": None}}
    assert r["seed"] == 9 and r["outputs"]["predict"] == {"method": "predict", "on": "train"}
    assert r["data"]["sha256"] == vr.file_sha256(tmp_path / "toy.data.npz")
    assert json.loads((tmp_path / "toy.json").read_text()) == r
    g = vr.recipe(model, data={"generator": "regression", "args": {"rows": 32, "test_rows": 8}})
    assert g["outputs"]["predict"]["on"] == "test"


def test_recipe_from_a_live_sequential_rebuilds_the_same_model(vr):
    ml = vr._parent
    body = ml.Sequential().linear(4).relu()
    live = ml.Sequential("identical", input_shape=(2, 2)).linear(4).residual_block(body).linear(1)
    live = live.compile(loss="mse", optimizer="sgd", lr=0.01)
    with pytest.raises(vr.Refused, match="not compiled"):
        vr.recipe(ml.Sequential().linear(1), data={"generator": "sequence", "args": {"shape": [2, 2]}})
    r = vr.recipe(live, data={"generator": "sequence", "args": {"rows": 8, "shape": [2, 2]}},
                  fit_params={"epochs": 2, "seed": 4})
    assert r["seed"] == 4 and r["fit"] == {"epochs": 2}
    json.dumps(r)
    built, fit = vr.build_model(ml, json.loads(json.dumps(r)))
    assert built._layers == live._layers and built._options == live._options and fit == {"epochs": 2, "seed": 4}
    calls = {"format": vr.FORMAT, "model": {"sequential": {"numeric_mode": "identical", "input_shape": [2, 2],
             "layers": [["linear", 4], ["residual_block", {"sequential": [["linear", 4], ["relu"]]}], ["linear", 1]],
             "compile": {"loss": "mse", "optimizer": "sgd", "lr": 0.01}}}, "seed": 4, "fit": {"epochs": 2},
             "data": r["data"], "outputs": r["outputs"]}
    from_calls, _ = vr.build_model(ml, calls)
    assert from_calls._layers == live._layers
    with pytest.raises(vr.Refused, match="no layer method"):
        vr.build_model(ml, dict(calls, model={"sequential": dict(calls["model"]["sequential"], layers=[["fit"]])}))


def test_seed_goes_to_random_state_or_is_refused(vr):
    est, _ = vr.build_model(vr._parent, toy_recipe())
    assert est.random_state == 3 and est.alpha == 2.0

    class NoSeed:
        def __init__(self, a=1):
            self.a = a

        def get_params(self, deep=True):
            return dict(a=self.a)

    vr._parent.NoSeed = NoSeed
    r = toy_recipe()
    r["model"] = {"estimator": "NoSeed", "params": {}}
    with pytest.raises(vr.Refused, match="takes no random_state"):
        vr.build_model(vr._parent, r)


# ---------------------------------------------------------------- devices, verdicts, record

def test_equal_devices_are_identical_and_the_record_is_written(vr, monkeypatch, capsys, tmp_path):
    path = tmp_path / "toy.json"
    path.write_text(json.dumps(toy_recipe()))
    code, out = run_cli(vr, monkeypatch, capsys, str(path))
    assert code == vr.EXIT_IDENTICAL, out
    assert "IDENTICAL: 2 fit(s) from scratch, 3 output(s) each" in out
    record = json.loads((tmp_path / "toy.record.json").read_text())
    assert record["format"] == vr.RECORD_FORMAT and record["version"] == "0.0.0+test"
    assert set(record["results"]) == {"cpu", "gpu"} and record["verdict"]["identical"]
    assert record["recipe_sha256"] == vr.recipe_sha256(toy_recipe())
    assert [o["name"] for o in record["results"]["cpu"]["outputs"]] == ["digest", "predict", "score"]


def test_a_one_ulp_difference_between_devices_is_divergent_with_the_element(vr, monkeypatch, capsys, tmp_path):
    vr._devices["gpu_nudge"] = True
    path = tmp_path / "toy.json"
    path.write_text(json.dumps(toy_recipe(predict={"method": "predict", "on": "test"})))
    code, out = run_cli(vr, monkeypatch, capsys, str(path))
    assert code == vr.EXIT_DIVERGENT, out
    assert "DIVERGENT: cpu and gpu differ in output predict: first differing element at [3]" in out


def test_one_device_install_records_and_says_there_is_nothing_to_compare(vr, monkeypatch, capsys, tmp_path):
    vr._parent.vendor = lambda: "cpu"
    path = tmp_path / "toy.json"
    path.write_text(json.dumps(toy_recipe()))
    code, out = run_cli(vr, monkeypatch, capsys, str(path))
    assert code == vr.EXIT_IDENTICAL and "one device (the CPU)" in out, out
    assert set(json.loads((tmp_path / "toy.record.json").read_text())["results"]) == {"cpu"}


def test_a_fast_process_is_refused_before_anything_runs(vr, monkeypatch, capsys):
    vr._parent.numeric_mode = lambda: "fast"
    code, out = run_cli(vr, monkeypatch, capsys, "example:ridge")
    assert code == vr.EXIT_REFUSED and "no promise about" in out


def test_against_compares_and_refuses_other_versions_recipes_and_data(vr, monkeypatch, capsys, tmp_path):
    path = tmp_path / "toy.json"
    path.write_text(json.dumps(toy_recipe()))
    a = str(tmp_path / "a.record.json")
    assert run_cli(vr, monkeypatch, capsys, str(path), record_out=a)[0] == 0
    code, out = run_cli(vr, monkeypatch, capsys, str(path), against=a, record_out=str(tmp_path / "b.record.json"))
    assert code == vr.EXIT_IDENTICAL and "IDENTICAL: 4 fit(s)" in out, out
    other = json.loads(Path(a).read_text())
    for change, words in ((dict(version="9.9.9"), "bits are promised within one version"),
                          (dict(data_sha256="0" * 64), "other data"),
                          (dict(recipe_sha256="1" * 64), "another recipe")):
        changed = tmp_path / "changed.record.json"
        changed.write_text(json.dumps(dict(other, **change)))
        code, out = run_cli(vr, monkeypatch, capsys, str(path), against=str(changed),
                            record_out=str(tmp_path / "c.record.json"))
        assert code == vr.EXIT_REFUSED and words in out, (change, out)
    # a record of another machine whose GPU answered one ulp apart: DIVERGENT, the element named from the record
    moved = copy.deepcopy(other)
    out_gpu = next(o for o in moved["results"]["gpu"]["outputs"] if o["name"] == "predict")
    values = vr._decode(out_gpu)
    values[5] = np.nextafter(values[5], -np.inf, dtype=values.dtype)
    out_gpu.update(vr.cb.describe_output(values), values=vr._encode(values))
    changed = tmp_path / "moved.record.json"
    changed.write_text(json.dumps(moved))
    code, out = run_cli(vr, monkeypatch, capsys, str(path), against=str(changed),
                        record_out=str(tmp_path / "d.record.json"))
    assert code == vr.EXIT_DIVERGENT and "first differing element at [5]" in out, out


def test_self_test_controls_all_pass_on_one_and_two_devices(vr, monkeypatch, capsys, tmp_path):
    path = tmp_path / "toy.json"
    path.write_text(json.dumps(toy_recipe()))
    for vendor in ("cuda", "cpu"):
        vr._parent.vendor = lambda v=vendor: v
        code, out = run_cli(vr, monkeypatch, capsys, str(path), tamper_test=True,
                            record_out=str(tmp_path / f"{vendor}.record.json"))
        assert code == vr.EXIT_IDENTICAL and "TAMPER-TEST PASS: 4 of 4 controls" in out, out
        assert out.count("  PASS  ") == 4


def test_self_test_fails_when_the_comparison_cannot_see_a_difference(vr, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(vr, "compare_results", lambda entries: dict(verdict="IDENTICAL", identical=True,
                                                                     compared=len(entries), outputs=1))
    path = tmp_path / "toy.json"
    path.write_text(json.dumps(toy_recipe()))
    code, out = run_cli(vr, monkeypatch, capsys, str(path), tamper_test=True,
                        record_out=str(tmp_path / "x.record.json"))
    assert code == vr.EXIT_DIVERGENT and "FAIL  one-ulp change is DIVERGENT" in out and "TAMPER-TEST FAIL" in out


def test_a_device_that_cannot_fit_is_not_run_never_identical(vr, monkeypatch, capsys, tmp_path):
    r = toy_recipe()
    r["model"]["params"]["alpha"] = "not a number"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(r))
    code, out = run_cli(vr, monkeypatch, capsys, str(path))
    assert code == vr.EXIT_CANNOT_RUN and "NOT RUN" in out and "fit raised" in out, out


def test_a_class_that_refuses_the_device_is_not_run_in_its_words(vr, monkeypatch, capsys, tmp_path):
    class NoDevice(Toy):
        def __init__(self, alpha=1.0, random_state=0, numeric_mode=None):
            super().__init__(alpha, random_state, numeric_mode)

    vr._parent.NoDevice = NoDevice
    r = toy_recipe()
    r["model"]["estimator"] = "NoDevice"
    path = tmp_path / "nodevice.json"
    path.write_text(json.dumps(r))
    code, out = run_cli(vr, monkeypatch, capsys, str(path))
    assert code == vr.EXIT_CANNOT_RUN and "NoDevice(device='cpu') refused" in out and "IDENTICAL" not in out, out


def test_a_side_that_did_not_run_on_its_device_is_never_compared(vr, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(Toy, "vendor_used", lambda self: "cuda")
    path = tmp_path / "toy.json"
    path.write_text(json.dumps(toy_recipe()))
    code, out = run_cli(vr, monkeypatch, capsys, str(path))
    assert code == vr.EXIT_CANNOT_RUN and "compare a thing to itself" in out, out


def test_package_routes_recipe_lazily_and_verify_has_the_flags():
    init = (PKG / "__init__.py").read_text()
    assert '"recipe": "._verify_recipe",' in init
    main = (PKG / "__main__.py").read_text()
    for flag in ('"--recipe"', '"--against"', '"--record-out"'):
        assert flag in main
    assert "recipe\t" in (ROOT / "docs-site/api_exclude.txt").read_text()


def test_the_one_ulp_control_also_moves_a_nan_answer(vr):
    nan = np.full(4, np.nan, dtype=np.float32)
    results = {"cpu": {"ran": True, "outputs": [dict(name="predict", kind="array", values=vr._encode(nan),
                                                     **vr.cb.describe_output(nan))]}}
    moved, arrays, what = vr._nudged(results, {"cpu": [nan]})
    assert what and moved["cpu"]["outputs"][0]["sha256"] != results["cpu"]["outputs"][0]["sha256"]
    verdict = vr.compare_results([("a", results["cpu"], [nan]), ("b", moved["cpu"], arrays["cpu"])])
    assert verdict["verdict"] == "DIVERGENT"



# ---------------------------------------------------------------- answers that are not numbers (lane mamba2-nan)

def test_two_devices_answering_nan_are_not_finite_never_identical(vr, monkeypatch, capsys, tmp_path):
    """The shipped seq-mamba2 recipe predicted NaN on the CPU column and was called IDENTICAL: equal bits of NaN
    must be a failure in words with its own exit code, on one device and on two."""
    def poisoned(self, X):
        out = np.zeros(len(X), dtype=np.float32)
        out[2] = np.nan
        out[5] = np.inf
        return out
    monkeypatch.setattr(Toy, "predict", poisoned)
    path = tmp_path / "toy.json"
    path.write_text(json.dumps(toy_recipe(digest=True, predict={"method": "predict", "on": "test"})))
    code, out = run_cli(vr, monkeypatch, capsys, str(path))
    assert code == vr.EXIT_NOT_FINITE, out
    assert "NOT FINITE: cpu answers 2 of 16 values of output predict as NaN or infinity" in out, out
    assert "IDENTICAL" not in out
    record = json.loads((tmp_path / "toy.record.json").read_text())
    assert record["verdict"]["verdict"] == "NOT FINITE"
    assert record["results"]["cpu"]["outputs"][1]["nonfinite"] == 2
    vr._parent.vendor = lambda: "cpu"
    code, out = run_cli(vr, monkeypatch, capsys, str(path), record_out=str(tmp_path / "one.record.json"))
    assert code == vr.EXIT_NOT_FINITE and "this installation: NOT FINITE" in out, out


def test_the_self_test_still_passes_its_controls_on_a_nan_answer(vr, monkeypatch, capsys, tmp_path):
    """The one-ulp control compares bits first, so a NaN answer still shows the comparison can fail; the run's
    own verdict stays NOT FINITE."""
    monkeypatch.setattr(Toy, "predict", lambda self, X: np.full(len(X), np.nan, dtype=np.float32))
    path = tmp_path / "toy.json"
    path.write_text(json.dumps(toy_recipe(predict={"method": "predict", "on": "test"})))
    code, out = run_cli(vr, monkeypatch, capsys, str(path), tamper_test=True)
    assert code == vr.EXIT_NOT_FINITE, out
    assert "TAMPER-TEST PASS: 4 of 4 controls" in out, out


def test_a_record_without_counts_is_counted_from_its_values(vr):
    nan = np.array([1.0, np.nan], dtype=np.float32)
    out = dict(name="predict", kind="array", values=vr._encode(nan), **vr.cb.describe_output(nan))
    result = {"ran": True, "outputs": [out]}
    verdict = vr.compare_results([("this", result, None), ("other", copy.deepcopy(result), None)])
    assert verdict["verdict"] == "NOT FINITE" and verdict["first"]["nonfinite"] == 1
    finite = np.array([1.0, 2.0], dtype=np.float32)
    ok = {"ran": True, "outputs": [dict(name="predict", kind="array", values=vr._encode(finite),
                                        **vr.cb.describe_output(finite))]}
    assert vr.compare_results([("a", ok, None), ("b", copy.deepcopy(ok), None)])["verdict"] == "IDENTICAL"


# ---------------------------------------------------------------- the results the user selects (paper 5.4)

def test_fitted_parameters_are_compared_array_by_array(vr, monkeypatch, capsys, tmp_path):
    path = tmp_path / "toy.json"
    path.write_text(json.dumps(toy_recipe(parameters=True, predict={"method": "predict", "on": "test"})))
    code, out = run_cli(vr, monkeypatch, capsys, str(path))
    assert code == vr.EXIT_IDENTICAL, out
    record = json.loads((tmp_path / "toy.record.json").read_text())
    assert [o["name"] for o in record["results"]["gpu"]["outputs"]] == ["param:coef_", "predict"]


def test_a_parameter_one_ulp_apart_is_divergent_and_named(vr, monkeypatch, capsys, tmp_path):
    def fit(self, X, y=None):
        Toy.fit(self, X, y)
        if self.device == "gpu":
            self.coef_[1] = np.nextafter(self.coef_[1], np.inf, dtype=np.float32)
        return self
    vr._parent.Toy = type("Toy", (Toy,), {"fit": fit})
    path = tmp_path / "toy.json"
    path.write_text(json.dumps(toy_recipe(parameters=["coef_"])))
    code, out = run_cli(vr, monkeypatch, capsys, str(path))
    assert code == vr.EXIT_DIVERGENT, out
    assert "differ in output param:coef_: first differing element at [1]" in out


def test_a_parameter_the_model_does_not_have_is_refused_in_words(vr, monkeypatch, capsys, tmp_path):
    path = tmp_path / "toy.json"
    path.write_text(json.dumps(toy_recipe(parameters=["weights_"])))
    code, out = run_cli(vr, monkeypatch, capsys, str(path))
    assert code == vr.EXIT_REFUSED and "no parameter(s) ['weights_']" in out, out


@pytest.mark.parametrize("outputs, words", [
    (dict(parameters="all"), "outputs.parameters is true"),
    (dict(parameters=[]), "outputs.parameters is true"),
    (dict(digest=True, weights=True), "a recipe compares digest, parameters, predict, score"),
])
def test_bad_output_selections_are_refused(vr, outputs, words):
    with pytest.raises(vr.Refused, match=re.escape(words)):
        vr.check_recipe(toy_recipe(**outputs))


def test_recipe_from_a_live_model_selects_parameters(vr):
    r = vr.recipe(vr._parent.Toy(alpha=2.0), data={"generator": "regression", "args": {"rows": 32, "test_rows": 8}},
                  parameters=["coef_"])
    assert r["outputs"]["parameters"] == ["coef_"]
    assert "parameters" not in vr.recipe(vr._parent.Toy(), data={"generator": "regression",
                                                                 "args": {"rows": 32}})["outputs"]
