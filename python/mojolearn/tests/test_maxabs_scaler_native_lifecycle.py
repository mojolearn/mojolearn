# SPDX-License-Identifier: Apache-2.0
"""Whole-operation MaxAbsScaler ownership, modes, buffers and validation."""
import json
import os
import subprocess
import sys
import warnings

import numpy as np
import pytest

import mojolearn as ml
from mojolearn._array import Array
from mojolearn._arena_io import DeviceRows
from mojolearn._verify_native_runtime import audit_native_call


def _data(rows=31, missing=False):
    x = np.random.default_rng(712).normal(size=(rows, 5)).astype(np.float32)
    x[:, -1] = 0
    if missing:
        x[::3, 0] = np.nan
        x[:, 1] = np.nan
    return x


def _bits(value):
    return np.asarray(value).tobytes()


def test_constructor_params_and_fitted_state_are_native():
    m = audit_native_call(ml.MaxAbsScaler, copy=False).assert_native()
    assert audit_native_call(m.get_params).assert_native() == {"copy": False}
    assert not audit_native_call(m.__sklearn_is_fitted__).assert_native()
    audit_native_call(m.fit, _data()).assert_native()
    assert audit_native_call(m.__sklearn_is_fitted__).assert_native()
    assert audit_native_call(m.set_params, copy=True).assert_native() is m
    assert m.__sklearn_is_fitted__()  # Existing set_params retains statistics.
    with pytest.raises(ValueError, match="invalid parameter"):
        m.set_params(unknown=True)


@pytest.mark.parametrize("blocked", ["0", "1"])
@pytest.mark.parametrize("rows", [13, 2049])
@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("copy", [False, True])
def test_existing_maxabs_paths_and_output_ownership_are_native(blocked, rows, missing, copy, monkeypatch):
    monkeypatch.setenv("MOJOLEARN_XPREP_BLOCKED", blocked)
    x = _data(rows, missing)
    before = x.tobytes()
    m = ml.MaxAbsScaler(copy=copy)
    out = audit_native_call(m.fit_transform, x).assert_native()
    assert isinstance(out, np.ndarray)
    assert out is not x and x.tobytes() == before
    assert _bits(out) == _bits(audit_native_call(m.transform, x).assert_native())
    assert m.n_features_in_ == x.shape[1] and m.n_samples_seen_ == rows
    assert np.array_equal(np.isnan(out), np.isnan(x))
    assert np.asarray(m.scale_)[-1] == 1
    finite = np.isfinite(x)
    assert np.all(np.abs(np.asarray(out)[finite]) <= 1)


@pytest.mark.parametrize("layout", ["f64", "integer", "fortran", "strided", "readonly", "memoryview", "array", "list", "bigendian"])
def test_supported_input_preparation_stays_native(layout):
    x = _data()
    if layout == "integer":
        x = np.round(x).astype(np.int64)
    source = {
        "f64": lambda: x.astype(np.float64),
        "integer": lambda: x,
        "fortran": lambda: np.asfortranarray(x),
        "strided": lambda: np.repeat(x, 2, axis=0)[::2],
        "readonly": lambda: x.copy(),
        "memoryview": lambda: memoryview(x),
        "array": lambda: Array.from_list(x.tolist(), "<f4"),
        "list": lambda: x.tolist(),
        "bigendian": lambda: x.astype(">f4"),
    }[layout]()
    if layout == "readonly":
        source.flags.writeable = False
    m = ml.MaxAbsScaler()
    actual = audit_native_call(m.fit_transform, source).assert_native()
    expected = ml.MaxAbsScaler().fit_transform(x.astype(np.float32))
    assert _bits(actual) == _bits(expected)


def test_native_row_selection_never_calls_its_python_take_hook():
    x = _data()
    indices = np.array([0, 7, 2, 19], np.int64)
    calls = []

    def take(*args):
        calls.append(True)
        raise AssertionError("Python gather hook executed")

    rows = DeviceRows(Array.from_list(x.tolist(), "<f4"), Array.from_list(indices.tolist(), "<i8"), take)
    m = ml.MaxAbsScaler()
    actual = audit_native_call(m.fit_transform, rows).assert_native()
    expected = ml.MaxAbsScaler().fit_transform(x[indices])
    assert _bits(actual) == _bits(expected)
    assert not calls


@pytest.mark.parametrize("target", [None, 3, "ignored", ["a"] * 31,
    np.arange(31), np.full(31, 1e100, np.float64), np.zeros(31, ">f8"),
    np.full(31, np.nan, np.float16)])
def test_ignored_target_preserves_supported_validation(target):
    m = ml.MaxAbsScaler()
    assert audit_native_call(m.fit, _data(), target).assert_native() is m


@pytest.mark.parametrize("target", [np.zeros(2), np.full(31, np.nan, np.float32),
    np.full(31, np.inf, np.float64), np.ones(31, np.complex64)])
def test_invalid_target_is_refused_natively(target):
    audit = audit_native_call(ml.MaxAbsScaler().fit, _data(), target)
    assert isinstance(audit.exception, ValueError)
    assert not audit.frames and not audit.wrapper_work


def test_column_target_warning_is_native_and_filterable():
    target = np.ones((31, 1), np.float64)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        audit_native_call(ml.MaxAbsScaler().fit, _data(), target).assert_native()
    assert len(seen) == 1
    assert "column-vector y" in str(seen[0].message)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        audit = audit_native_call(ml.MaxAbsScaler().fit, _data(), target)
    assert isinstance(audit.exception, Warning)
    assert not audit.frames and not audit.wrapper_work


def test_python_input_protocol_is_refused_before_any_hook():
    called = []

    class Input:
        def __array__(self, *args, **kwargs):
            called.append(True)
            return _data()

    audit = audit_native_call(ml.MaxAbsScaler().fit, Input())
    assert isinstance(audit.exception, TypeError)
    assert not audit.frames and not called


def test_unfitted_error_is_native():
    from mojolearn.exceptions import NotFittedError
    audit = audit_native_call(ml.MaxAbsScaler().transform, _data())
    assert isinstance(audit.exception, NotFittedError)
    assert not audit.frames


def test_pandas_numeric_input_and_target_are_native():
    pd = pytest.importorskip("pandas")
    x = _data(missing=True)
    frame = pd.DataFrame(x)
    m = ml.MaxAbsScaler()
    actual = audit_native_call(m.fit_transform, frame, pd.Series(np.arange(len(x)))).assert_native()
    assert _bits(actual) == _bits(ml.MaxAbsScaler().fit_transform(x))


def test_numpy_free_input_and_target_remain_native():
    script = r'''
import json,sys
from array import array
import mojolearn as ml
from mojolearn._verify_native_runtime import audit_native_call
assert "numpy" not in sys.modules
x = memoryview(array("f", [1,2,3,4,5,6])).cast("B").cast("f", (3,2))
m = audit_native_call(ml.MaxAbsScaler).assert_native()
r = audit_native_call(m.fit_transform, x, [1.,2.,3.]).assert_native()
assert r.shape == (3,2)
assert r.tolist()[-1] == [1.,1.]
assert "numpy" not in sys.modules
print(json.dumps({"passed":True}))
'''
    result = subprocess.run([sys.executable, "-c", script], text=True, capture_output=True, env=os.environ)
    assert result.returncode == 0, result.stderr[-4000:]
    assert json.loads(result.stdout.splitlines()[-1])["passed"]
