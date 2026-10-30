# SPDX-License-Identifier: Apache-2.0
"""MinMaxScaler whole-operation boundaries, options, buffers and checkpoints."""
import io
import json
import os
import subprocess
import sys

import numpy as np
import pytest

import mojolearn as ml
from mojolearn._array import Array
from mojolearn._verify_native_runtime import audit_native_call


def _data(missing=False):
    x = np.random.default_rng(911).normal(size=(43, 5)).astype(np.float32)
    x[:, 3] = 2.5
    x[:, 4] = np.float32(1) + np.arange(len(x), dtype=np.float32) * np.finfo(np.float32).eps / 64
    if missing:
        x[::5, 0] = np.nan
        x[:, 2] = np.nan
    return x


def _bits(value):
    return np.asarray(value).tobytes()


def test_constructor_params_and_fitted_predicate_are_native():
    m = audit_native_call(ml.MinMaxScaler, (-2, 3), copy=False, clip=True).assert_native()
    assert audit_native_call(m.get_params).assert_native() == dict(
        feature_range=(-2, 3), copy=False, clip=True, numeric_mode=None)
    assert not audit_native_call(m.__sklearn_is_fitted__).assert_native()
    audit_native_call(m.fit, _data()).assert_native()
    assert audit_native_call(m.__sklearn_is_fitted__).assert_native()
    assert audit_native_call(m.set_params).assert_native() is m
    assert m.__sklearn_is_fitted__()
    assert audit_native_call(m.set_params, clip=False).assert_native() is m
    assert not m.__sklearn_is_fitted__()
    with pytest.raises(ValueError, match="Invalid"):
        m.set_params(unknown=True)


@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("clip", [False, True])
@pytest.mark.parametrize("bounds", [(0, 1), (-2.5, 4.75), (1e-6, 5e-6)])
def test_fit_transform_inverse_and_checkpoint_are_native(missing, clip, bounds):
    x = _data(missing)
    m = ml.MinMaxScaler(feature_range=bounds, clip=clip)
    fit = audit_native_call(m.fit_transform, x).assert_native()
    assert isinstance(fit, np.ndarray)
    assert _bits(fit) == _bits(audit_native_call(m.transform, x).assert_native())
    restored = audit_native_call(m.inverse_transform, fit).assert_native()
    assert np.array_equal(np.isnan(restored), np.isnan(x))
    finite = np.isfinite(x)
    np.testing.assert_allclose(np.asarray(restored)[finite], x[finite], atol=5e-6, rtol=5e-6)
    archive = io.BytesIO()
    assert audit_native_call(m.save, archive).assert_native() is archive
    archive.seek(0)
    loaded = audit_native_call(ml.MinMaxScaler.load, archive).assert_native()
    for name in ("data_min_", "data_max_", "data_range_", "scale_", "min_"):
        assert _bits(getattr(m, name)) == _bits(getattr(loaded, name)), name
    assert _bits(fit) == _bits(audit_native_call(loaded.transform, x).assert_native())


@pytest.mark.parametrize("layout", ["c", "fortran", "strided", "readonly", "memoryview", "array", "list", "bigendian"])
@pytest.mark.parametrize("inverse", [False, True])
def test_copy_false_preserves_layout_and_ownership_semantics(layout, inverse):
    x = _data(True)
    m = ml.MinMaxScaler(copy=False, clip=True).fit(x)
    reference = ml.MinMaxScaler(copy=True, clip=True).fit(x)
    operation = m.inverse_transform if inverse else m.transform
    wanted = reference.inverse_transform(x) if inverse else reference.transform(x)
    if layout == "fortran":
        source = np.asfortranarray(x)
    elif layout == "strided":
        source = np.repeat(x, 2, axis=0)[::2]
    elif layout == "readonly":
        source = x.copy()
        source.flags.writeable = False
    elif layout == "memoryview":
        source = memoryview(x.copy())
    elif layout == "array":
        source = Array.from_list(x.tolist(), "<f4")
    elif layout == "list":
        source = x.tolist()
    elif layout == "bigendian":
        source = x.astype(">f4")
    else:
        source = x.copy()
    before = _bits(source)
    result = audit_native_call(operation, source).assert_native()
    assert _bits(result) == _bits(wanted)
    if layout in ("c", "memoryview", "array"):
        assert _bits(source) == _bits(wanted)
        if layout != "array":
            assert result is source
    else:
        assert _bits(source) == before


@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("clip", [False, True])
def test_partial_fit_preserves_extrema_and_counts(missing, clip):
    x = _data(missing)
    m = ml.MinMaxScaler(feature_range=(-2.5, 4.75), clip=clip)
    audit_native_call(m.partial_fit, x[:19]).assert_native()
    audit_native_call(m.partial_fit, x[19:]).assert_native()
    reference = ml.MinMaxScaler(feature_range=(-2.5, 4.75), clip=clip).fit(x)
    for name in ("data_min_", "data_max_", "data_range_", "scale_", "min_"):
        assert _bits(getattr(m, name)) == _bits(getattr(reference, name)), name
    assert m.n_samples_seen_ == len(x)
    assert _bits(audit_native_call(m.transform, x).assert_native()) == _bits(reference.transform(x))


def test_partial_fit_preserves_previously_unseen_column_nan():
    x = _data()
    first = x[:19].copy()
    first[:, 1] = np.nan
    m = ml.MinMaxScaler()
    audit_native_call(m.partial_fit, first).assert_native()
    audit_native_call(m.partial_fit, x[19:]).assert_native()
    assert np.isnan(np.asarray(m.data_min_)[1])
    assert np.isnan(np.asarray(m.transform(x))[:, 1]).all()


@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("clip", [False, True])
def test_unseen_column_mask_survives_finite_transform_input(inverse, clip):
    x = _data()
    x[:, 1] = np.nan
    model = ml.MinMaxScaler(clip=clip).fit(x)
    source = np.nan_to_num(x)
    operation = model.inverse_transform if inverse else model.transform
    result = np.asarray(audit_native_call(operation, source).assert_native())
    assert np.isnan(result[:, 1]).all()
    assert np.isfinite(result[:, [0, 2, 3, 4]]).all()


@pytest.mark.parametrize("bounds", [
    (0, 1), [-2, 3], (np.float32(-1), np.int64(2)),
    np.array([-1, 2], np.float32), np.array([-1, 2], np.float64),
    np.array([-1, 2], object), (10 ** 20, 2 * 10 ** 20),
])
def test_supported_range_endpoints_stay_native(bounds):
    m = ml.MinMaxScaler(feature_range=bounds)
    audit_native_call(m.fit, _data()).assert_native()
    assert m.feature_range_ == tuple(float(np.float32(x)) for x in bounds)


@pytest.mark.parametrize("bounds", [
    (True, 2), (0, np.bool_(True)), (0, 0), (2, 1), (0,), (0, 1, 2),
    (0, np.inf), (-np.inf, 1), (0, np.nan), (0, 1e40),
    np.array([[0, 1]], np.float32), np.array([False, True]), range(2),
])
def test_invalid_ranges_are_refused_before_training(bounds):
    audit = audit_native_call(ml.MinMaxScaler(feature_range=bounds).fit, _data())
    assert isinstance(audit.exception, ValueError)
    assert not audit.frames
    assert not audit.wrapper_work


def test_refit_and_partial_failure_state_boundaries():
    x = _data()
    m = ml.MinMaxScaler().fit(x)
    m.feature_range = (1, 0)
    with pytest.raises(ValueError):
        m.fit(x)
    assert m.__sklearn_is_fitted__()  # Invalid configuration precedes invalidation.
    m.feature_range = (0, 1)
    with pytest.raises(ValueError):
        m.partial_fit(x[:, :-1])
    assert m.n_samples_seen_ == len(x)
    failed = x.copy()
    failed[0, 0] = np.inf
    with pytest.raises(ValueError):
        m.fit(failed)
    assert not m.__sklearn_is_fitted__()


def test_input_protocol_callbacks_are_refused_without_invocation():
    called = []

    class Input:
        def __array__(self, *args, **kwargs):
            called.append(True)
            return _data()

    audit = audit_native_call(ml.MinMaxScaler().fit, Input())
    assert isinstance(audit.exception, TypeError)
    assert not audit.frames
    assert called == []


def test_unfitted_error_is_native_and_preserves_exception_type():
    from mojolearn.exceptions import NotFittedError
    m = ml.MinMaxScaler()
    audit = audit_native_call(m.transform, _data())
    assert isinstance(audit.exception, NotFittedError)
    assert not audit.frames


def test_explicit_host_unfitted_error_uses_its_native_exception_context():
    from mojolearn._classical_host import HostMinMaxScaler
    from mojolearn.exceptions import NotFittedError
    m = audit_native_call(HostMinMaxScaler).assert_native()
    audit = audit_native_call(m.transform, _data())
    assert isinstance(audit.exception, NotFittedError)
    assert not audit.frames
    assert not audit.wrapper_work


def test_object_range_endpoint_refuses_python_numeric_hooks():
    called = []

    class Endpoint:
        def __float__(self):
            called.append(True)
            return 1.0

    m = ml.MinMaxScaler(feature_range=np.array([0, Endpoint()], object))
    audit = audit_native_call(m.fit, _data())
    assert isinstance(audit.exception, ValueError)
    assert not audit.frames
    assert not audit.wrapper_work
    assert called == []


def test_numpy_free_first_fit_transform_and_archive_are_native():
    script = r'''
import io, json, sys
# Model an environment without the optional dependency before package import.
import importlib.abc
class BlockNumPy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "numpy" or fullname.startswith("numpy."):
            raise ModuleNotFoundError("NumPy is absent in this fixture", name="numpy")
sys.meta_path.insert(0, BlockNumPy())
from array import array
import mojolearn as ml
from mojolearn._verify_native_runtime import audit_native_call
assert "numpy" not in sys.modules
m = audit_native_call(ml.MinMaxScaler, copy=False).assert_native()
store = array("f", [1,2,3,4,5,6])
x = memoryview(store).cast("B").cast("f",(3,2))
r = audit_native_call(m.fit_transform, x).assert_native()
assert r is x
assert list(store) == [0,0,.5,.5,1,1]
f = io.BytesIO()
audit_native_call(m.save, f).assert_native()
f.seek(0)
audit_native_call(ml.MinMaxScaler.load, f).assert_native()
assert "numpy" not in sys.modules
print(json.dumps({"passed": True}))
'''
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=os.environ)
    assert result.returncode == 0, result.stderr[-4000:]
    assert json.loads(result.stdout.splitlines()[-1])["passed"]
