# SPDX-License-Identifier: Apache-2.0
"""LinearRegression, Ridge and their host classes on an install without NumPy.

A child process blocks every NumPy import (as on a NumPy-free install), then
runs each public method through the native-call audit: the inputs are Python
lists, `array.array`, memoryviews and mojolearn Arrays, and every output is a
mojolearn Array. This process, with NumPy, runs the same calls on ndarrays and
requires the same bits: coefficients, predictions, scores, saved files, host
predictions and the host coefficient digest.
"""
import hashlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import mojolearn
from mojolearn import LinearRegression, Ridge

#: This module calls the estimators as a user does, so `predict` answers it
#: with the public float64 output, as it answers the child process.
__mojolearn_user_boundary__ = True

_CHILD = r'''
import array, gc, hashlib, io, json, sys, weakref

class BlockNumpy:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "numpy" or fullname.startswith("numpy."):
            raise ModuleNotFoundError("No module named 'numpy'", name="numpy")

sys.meta_path.insert(0, BlockNumpy())
import mojolearn
from mojolearn import Array, LinearRegression, Ridge
from mojolearn.exceptions import NotFittedError
from mojolearn._classical_host import HostLinearRegression, HostRidge
from mojolearn._verify_native_runtime import audit_native_call

payload = json.load(sys.stdin)
X, y, w, tmp = payload["x"], payload["y"], payload["w"], payload["tmp"]
n, d = len(X), len(X[0])


def clean(operation, *args, error=None, **kwargs):
    audit = audit_native_call(operation, *args, **kwargs)
    report = audit.report()
    assert not any(report[k] for k in ("python_frames", "forbidden_native_calls", "wrapper_work",
                                       "unobserved_existing_threads")), report
    if error is not None:
        assert isinstance(audit.exception, error), report
        return audit.exception
    return audit.assert_native()


def hexed(value, dtype):
    assert type(value) is Array and value.dtype == dtype, (type(value), getattr(value, "dtype", None))
    return value.tobytes().hex()


def flat_f8(rows):
    return array.array("d", [v for row in rows for v in row])


out = {}
for name, cls, host in (("LinearRegression", LinearRegression, HostLinearRegression),
                        ("Ridge", Ridge, HostRidge)):
    r = {}
    model = clean(cls, numeric_mode="identical")
    r["params"] = clean(model.get_params)
    assert clean(model.__sklearn_is_fitted__) is False
    clean(model.predict, X, error=NotFittedError)
    assert clean(model.fit, X, y) is model
    assert clean(model.__sklearn_is_fitted__) is True
    coef = clean(getattr, model, "coef_")
    assert coef.shape == (d,)
    r["coef"] = hexed(coef, "<f4")
    r["intercept"] = float.hex(clean(getattr, model, "intercept_"))
    assert clean(getattr, model, "n_features_in_") == d
    assert clean(getattr, model, "input_copied_") is True
    prediction = clean(model.predict, X)
    assert prediction.shape == (n,)
    r["predict"] = hexed(prediction, "<f8")
    r["score"] = float.hex(clean(model.score, X, y))
    r["vendor"] = clean(model.vendor_used)
    r["mode"] = clean(model.numeric_mode_used)

    # The same numbers through every buffer door give the same coefficients.
    rows_f8 = memoryview(flat_f8(X)).cast("B").cast("d", (n, d))
    wide_y = array.array("d", [v for t in y for v in (t, -1.0)])
    variants = {
        "memoryview": (rows_f8, memoryview(wide_y)[::2]),
        "array": (Array.from_list(X, "<f4"), array.array("f", y)),
        "tuple": (tuple(tuple(row) for row in X), tuple(y)),
    }
    for key, (vx, vy) in variants.items():
        other = clean(cls, numeric_mode="identical")
        clean(other.fit, vx, vy)
        r["coef_" + key] = hexed(clean(getattr, other, "coef_"), "<f4")
        r["predict_" + key] = hexed(clean(other.predict, vx), "<f8")
        if key == "array":
            # A float32 C-order Array is read in place.
            assert clean(getattr, other, "input_copied_") is False

    weighted = clean(cls, numeric_mode="identical")
    clean(weighted.fit, X, y, sample_weight=w)
    r["weighted_coef"] = hexed(weighted.coef_, "<f4")
    r["weighted_intercept"] = float.hex(weighted.intercept_)
    clean(weighted.fit, X, y, sample_weight=2.0)
    r["scalar_weight_coef"] = hexed(weighted.coef_, "<f4")
    plain = clean(cls, fit_intercept=False, numeric_mode="identical")
    clean(plain.fit, X, y)
    r["no_intercept_coef"] = hexed(plain.coef_, "<f4")
    assert plain.intercept_ == 0.0

    path = tmp + "/" + name + ".npz"
    clean(model.save, path)
    data = open(path, "rb").read()
    r["file"] = hashlib.sha256(data).hexdigest()
    again = io.BytesIO()
    clean(model.save, again)
    assert again.getvalue() == data
    loaded = clean(cls.load, path)
    assert type(loaded) is cls
    r["loaded_coef"] = hexed(clean(getattr, loaded, "coef_"), "<f4")
    r["loaded_predict"] = hexed(clean(loaded.predict, X), "<f8")
    resaved = io.BytesIO()
    clean(loaded.save, resaved)
    assert resaved.getvalue() == data

    routed = mojolearn.host_model(path)
    assert type(routed) is host and routed.estimator == name
    r["host_predict"] = hexed(clean(routed.predict, X), "<f8")
    r["host_sha"] = clean(routed.model_sha256)
    hosted = clean(host.load, path)
    assert clean(hosted.vendor_used) == "cpu"
    assert hexed(clean(hosted.predict, X), "<f8") == r["host_predict"]

    assert clean(loaded.set_params, fit_intercept=False) is loaded
    assert clean(loaded.__sklearn_is_fitted__) is False
    clean(getattr, loaded, "coef_", error=AttributeError)
    clean(model.fit, X[0], y, error=ValueError)

    # Nothing keeps the caller's inputs, the model or its outputs alive.
    def run():
        local = cls(numeric_mode="identical")
        counts = [sys.getrefcount(v) for v in (X, y, w)]
        local.fit(X, y)
        local.fit(X, y, sample_weight=w)
        result = local.predict(X)
        local.score(X, y)
        local.coef_, local.intercept_
        local.save(tmp + "/leak.npz")
        restored = cls.load(tmp + "/leak.npz")
        assert [sys.getrefcount(v) for v in (X, y, w)] == counts
        # An Array takes no weak reference: an output held by one name must
        # count as a fresh Array held the same way, and a fitted coef_ as one
        # held by a dictionary and a name.
        fresh = Array((n,), "<f8")
        assert sys.getrefcount(result) == sys.getrefcount(fresh)
        coef = object.__getattribute__(local, "coef_")
        holder = {"coef_": Array((d,))}
        held = holder["coef_"]
        assert sys.getrefcount(coef) == sys.getrefcount(held)
        return [weakref.ref(local), weakref.ref(restored)]
    references = run()
    gc.collect()
    assert [ref for ref in references if ref() is not None] == []

    def cycle():
        for _ in range(100):  # repeated whole public operations
            model.fit(X, y).predict(X)
            model.score(X, y)
    cycle()
    cycle()
    # A leak grows every window of 300 operations; a one-time lazy allocation
    # (seen once in the third cycle on the Apple route, then flat) does not.
    # So the gauge reads three windows and holds the smallest growth.
    growth = []
    for _ in range(3):
        gc.collect()
        before = sys.getallocatedblocks()
        cycle()
        gc.collect()
        growth.append(sys.getallocatedblocks() - before)
    assert min(growth) < 100, growth
    out[name] = r

assert not any(m == "numpy" or m.startswith("numpy.") for m in sys.modules)
print("RESULT " + json.dumps(out))
'''


def _data():
    rng = np.random.default_rng(2117)
    x = rng.normal(size=(23, 3)).astype("f4")
    y = (x @ np.array([.75, -.5, 1.25], "f4") + .375 + rng.normal(scale=.125, size=23)).astype("f4")
    w = np.linspace(.5, 1.5, 23).astype("f4")
    return x, y, w


def _numpy_free_run(tmp_path):
    x, y, w = _data()
    payload = json.dumps({"x": x.tolist(), "y": y.tolist(), "w": w.tolist(), "tmp": str(tmp_path)})
    env = dict(os.environ, PYTHONPATH=str(Path(mojolearn.__file__).resolve().parents[1]))
    done = subprocess.run([sys.executable, "-c", _CHILD], input=payload, env=env,
                          capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stderr[-4000:]
    line = [ln for ln in done.stdout.splitlines() if ln.startswith("RESULT ")]
    assert line, done.stdout[-2000:]
    return json.loads(line[-1][len("RESULT "):])


@pytest.fixture(scope="module")
def numpy_free(tmp_path_factory):
    return _numpy_free_run(tmp_path_factory.mktemp("numpy-free"))


@pytest.mark.parametrize("cls", (LinearRegression, Ridge))
def test_numpy_free_methods_return_the_numpy_bits(cls, numpy_free, tmp_path):
    from mojolearn._classical_host import HostLinearRegression, HostRidge
    host = HostLinearRegression if cls is LinearRegression else HostRidge
    got = numpy_free[cls.__name__]
    x, y, w = _data()
    model = cls(numeric_mode="identical")
    assert got["params"] == model.get_params()
    model.fit(x, y)
    coef = np.asarray(model.coef_).tobytes().hex()
    prediction = model.predict(x).tobytes().hex()
    assert got["coef"] == coef
    assert got["intercept"] == float.hex(model.intercept_)
    assert got["predict"] == prediction
    assert got["score"] == float.hex(model.score(x, y))
    assert got["vendor"] == model.vendor_used() and got["mode"] == model.numeric_mode_used()
    for key in ("memoryview", "array", "tuple"):
        assert got["coef_" + key] == coef, key
        assert got["predict_" + key] == prediction, key
    weighted = cls(numeric_mode="identical").fit(x, y, sample_weight=w.astype("f8"))
    assert got["weighted_coef"] == np.asarray(weighted.coef_).tobytes().hex()
    assert got["weighted_intercept"] == float.hex(weighted.intercept_)
    weighted.fit(x, y, sample_weight=2.0)
    assert got["scalar_weight_coef"] == np.asarray(weighted.coef_).tobytes().hex()
    plain = cls(fit_intercept=False, numeric_mode="identical").fit(x, y)
    assert got["no_intercept_coef"] == np.asarray(plain.coef_).tobytes().hex()

    path = tmp_path / "with-numpy.npz"
    model.save(path)
    assert got["file"] == hashlib.sha256(path.read_bytes()).hexdigest()
    loaded = cls.load(path)
    assert got["loaded_coef"] == np.asarray(loaded.coef_).tobytes().hex()
    assert got["loaded_predict"] == loaded.predict(x).tobytes().hex()
    routed = mojolearn.host_model(path)
    assert type(routed) is host
    assert got["host_predict"] == routed.predict(x).tobytes().hex()
    assert got["host_sha"] == routed.model_sha256()
    saved = io.BytesIO()
    loaded.save(saved)
    assert hashlib.sha256(saved.getvalue()).hexdigest() == got["file"]



_LATE_NUMPY_CHILD = r'''
import json, os, sys
import mojolearn
from mojolearn import LinearRegression, Ridge
assert "numpy" not in sys.modules, "importing mojolearn loaded NumPy"
import numpy as np
data = json.loads(sys.stdin.read())
x, y = np.array(data["x"], "f4"), np.array(data["y"], "f4")
out = {}
for cls in (LinearRegression, Ridge):
    model = cls(numeric_mode="identical").fit(x, y)
    prediction = model.predict(x)
    assert type(model.coef_) is np.ndarray and type(prediction) is np.ndarray, cls.__name__
    path = os.path.join(data["tmp"], cls.__name__ + ".npz")
    model.save(path)
    hosted = mojolearn.host_model(path).predict(x)
    assert type(hosted) is np.ndarray, cls.__name__
    out[cls.__name__] = {"coef": model.coef_.tobytes().hex(), "predict": prediction.tobytes().hex(),
                         "host_predict": hosted.tobytes().hex()}
print("RESULT " + json.dumps(out))
'''


def test_numpy_loaded_after_import_returns_ndarrays(tmp_path):
    """Importing mojolearn never loads NumPy; NumPy imported afterwards
    connects the ndarray path, with the bits of a NumPy-first process."""
    x, y, _ = _data()
    env = dict(os.environ, PYTHONPATH=str(Path(mojolearn.__file__).resolve().parents[1]))
    payload = json.dumps({"x": x.tolist(), "y": y.tolist(), "tmp": str(tmp_path)})
    done = subprocess.run([sys.executable, "-c", _LATE_NUMPY_CHILD], env=env, capture_output=True, text=True,
                          input=payload, timeout=600)
    assert done.returncode == 0, done.stderr[-4000:]
    line = [ln for ln in done.stdout.splitlines() if ln.startswith("RESULT ")]
    assert line, done.stdout[-2000:]
    got = json.loads(line[-1][len("RESULT "):])
    for cls in (LinearRegression, Ridge):
        model = cls(numeric_mode="identical").fit(x, y)
        path = tmp_path / ("parent-" + cls.__name__ + ".npz")
        model.save(path)
        assert got[cls.__name__]["coef"] == np.asarray(model.coef_).tobytes().hex(), cls.__name__
        assert got[cls.__name__]["predict"] == np.asarray(model.predict(x)).tobytes().hex(), cls.__name__
        hosted = mojolearn.host_model(path).predict(x)
        assert got[cls.__name__]["host_predict"] == np.asarray(hosted).tobytes().hex(), cls.__name__


_HOST_FIT_CHILD = r'''
import io, json, os, sys
order = sys.argv[1]
if order == "blocked":
    class BlockNumpy:
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "numpy" or fullname.startswith("numpy."):
                raise ModuleNotFoundError("No module named 'numpy'", name="numpy")
    sys.meta_path.insert(0, BlockNumpy())
elif order == "first":
    import numpy
import mojolearn
from mojolearn import LinearRegression, Ridge
if order == "cpu":
    # The CPU column: the public classes on the CPU binaries.
    assert mojolearn.vendor() == "cpu", mojolearn.vendor()
    classes = (("LinearRegression", LinearRegression), ("Ridge", Ridge))
else:
    from mojolearn._classical_host import HostLinearRegression, HostRidge
    classes = (("LinearRegression", HostLinearRegression), ("Ridge", HostRidge))
if order == "after":
    import numpy
data = json.loads(sys.stdin.read())
X, y, w = data["x"], data["y"], data["w"]
out = {}
for name, cls in classes:
    r = {}
    model = cls().fit(X, y)
    r["coef"] = model.coef_.tobytes().hex()
    r["intercept"] = float.hex(model.intercept_)
    r["predict"] = model.predict(X).tobytes().hex()
    r["score"] = float.hex(model.score(X, y))
    r["vendor"] = model.vendor_used()
    r["weighted_coef"] = cls().fit(X, y, sample_weight=w).coef_.tobytes().hex()
    r["no_intercept_coef"] = cls(fit_intercept=False).fit(X, y).coef_.tobytes().hex()
    r["one_column_coef"] = cls().fit([row[:1] for row in X], y).coef_.tobytes().hex()
    if order != "cpu":
        r["type"] = type(model.predict(X)).__name__
        path = os.path.join(data["tmp"], order + "-" + name + ".npz")
        model.save(path)
        loaded = cls.load(path)
        r["loaded_predict"] = loaded.predict(X).tobytes().hex()
        r["sha"] = model.model_sha256()
        r["loaded_sha"] = loaded.model_sha256()
    out[name] = r
if order == "blocked":
    assert not any(m == "numpy" or m.startswith("numpy.") for m in sys.modules)
print("RESULT " + json.dumps(out))
'''


def _host_fit_run(order, tmp_path, **env_extra):
    x, y, w = _data()
    env = dict(os.environ, PYTHONPATH=str(Path(mojolearn.__file__).resolve().parents[1]), **env_extra)
    payload = json.dumps({"x": x.tolist(), "y": y.tolist(), "w": w.tolist(), "tmp": str(tmp_path)})
    done = subprocess.run([sys.executable, "-c", _HOST_FIT_CHILD, order], env=env, capture_output=True,
                          text=True, input=payload, timeout=600)
    assert done.returncode == 0, done.stderr[-4000:]
    line = [ln for ln in done.stdout.splitlines() if ln.startswith("RESULT ")]
    assert line, done.stdout[-2000:]
    return json.loads(line[-1][len("RESULT "):])


def test_host_classes_fit_on_the_cpu_in_every_numpy_order(tmp_path):
    """HostLinearRegression and HostRidge fit, predict, score, save, load
    and digest on the CPU binaries without NumPy, with NumPy imported first
    and with NumPy imported after mojolearn, with the CPU column's bits."""
    reference = _host_fit_run("cpu", tmp_path, MOJOLEARN_VENDOR="cpu")
    runs = {order: _host_fit_run(order, tmp_path) for order in ("blocked", "first", "after")}
    for order, got in runs.items():
        for name in ("LinearRegression", "Ridge"):
            r = got[name]
            assert r["vendor"] == "cpu", (order, name)
            assert r["type"] == ("Array" if order == "blocked" else "ndarray"), (order, name, r["type"])
            for key, value in reference[name].items():
                if key != "vendor":
                    assert r[key] == value, (order, name, key)
            assert r["loaded_predict"] == r["predict"], (order, name)
            assert r["loaded_sha"] == r["sha"] == runs["blocked"][name]["sha"], (order, name)
