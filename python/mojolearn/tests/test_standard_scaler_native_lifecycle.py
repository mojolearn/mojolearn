# SPDX-License-Identifier: Apache-2.0
"""Whole public scaler lifecycle boundaries and observable state/copy semantics."""
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
    x = np.random.default_rng(901).normal(size=(37, 4)).astype(np.float32)
    x[:, 3] = 2.5
    if missing:
        x[::5, 0] = np.nan
        x[:, 2] = np.nan
    return x


def _bits(value):
    return np.asarray(value).tobytes()


def test_constructor_params_and_fitted_state_are_native():
    model = audit_native_call(ml.StandardScaler, copy=False).assert_native()
    assert audit_native_call(model.get_params).assert_native() == dict(
        copy=False, with_mean=True, with_std=True, numeric_mode=None)
    assert audit_native_call(model.__sklearn_is_fitted__).assert_native() is False
    audit_native_call(model.fit, _data()).assert_native()
    assert audit_native_call(model.__sklearn_is_fitted__).assert_native() is True
    assert audit_native_call(model.set_params).assert_native() is model
    assert model.__sklearn_is_fitted__()
    assert audit_native_call(model.set_params, with_mean=False).assert_native() is model
    assert not model.__sklearn_is_fitted__()
    with pytest.raises(ValueError, match="Invalid"):
        model.set_params(unknown=1)


@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("with_mean,with_std", [(True, True), (True, False), (False, True), (False, False)])
def test_transform_fit_transform_and_inverse_are_native(missing, with_mean, with_std):
    x = _data(missing)
    model = ml.StandardScaler(with_mean=with_mean, with_std=with_std)
    fitted = audit_native_call(model.fit_transform, x).assert_native()
    transformed = audit_native_call(model.transform, x).assert_native()
    assert _bits(fitted) == _bits(transformed)
    assert isinstance(transformed, np.ndarray)
    restored = audit_native_call(model.inverse_transform, transformed).assert_native()
    assert np.array_equal(np.isnan(restored), np.isnan(x))
    finite = np.isfinite(x)
    np.testing.assert_allclose(np.asarray(restored)[finite], x[finite], atol=5e-7, rtol=2e-6)


@pytest.mark.parametrize("layout", ["c", "fortran", "strided", "readonly", "memoryview", "array", "list", "bigendian"])
@pytest.mark.parametrize("inverse", [False, True])
def test_native_copy_false_keeps_supported_inplace_semantics(layout, inverse):
    x = _data(True)
    model = ml.StandardScaler(copy=False).fit(x)
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
    operation = model.inverse_transform if inverse else model.transform
    wanted = operation(x, copy=True)
    before = _bits(source)
    result = audit_native_call(operation, source).assert_native()
    assert _bits(result) == _bits(wanted)
    inplace = layout in ("c", "memoryview", "array")
    if inplace:
        assert _bits(source) == _bits(wanted)
        if layout != "array":
            assert result is source
    else:
        assert _bits(source) == before


@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("with_mean,with_std", [(True, True), (True, False), (False, True), (False, False)])
def test_partial_fit_and_npz_roundtrip_are_native(missing, weighted, with_mean, with_std):
    x = _data(missing)
    model = ml.StandardScaler(with_mean=with_mean, with_std=with_std)
    w = np.linspace(.5, 1.5, len(x), dtype=np.float32) if weighted else None
    for part in (slice(0, 17), slice(17, None)):
        audit_native_call(model.partial_fit, x[part], sample_weight=None if w is None else w[part]).assert_native()
    archive = io.BytesIO()
    assert audit_native_call(model.save, archive).assert_native() is archive
    archive.seek(0)
    restored = audit_native_call(ml.StandardScaler.load, archive).assert_native()
    for name in ("mean_", "var_", "scale_", "n_samples_seen_"):
        expected = getattr(model, name)
        if name == "n_samples_seen_" and type(expected) is float:
            # The existing NPZ schema saves a scalar weight sum as float32.
            # Count-only partial_fit can retain a float64 sum between saves.
            expected = float(np.float32(expected))
        assert _bits(getattr(restored, name)) == _bits(expected), name
    assert _bits(audit_native_call(restored.transform, x).assert_native()) == _bits(model.transform(x))


def test_count_only_weighted_archive_keeps_existing_float32_storage():
    x = _data()
    weights = np.linspace(.5, 1.5, len(x), dtype=np.float32)
    model = ml.StandardScaler(with_mean=False, with_std=False)
    for part in (slice(0, 17), slice(17, None)):
        audit_native_call(model.partial_fit, x[part], sample_weight=weights[part]).assert_native()
    count = model.n_samples_seen_
    assert count == 36.999999046325684  # Retained CPU/Metal fixture, before migration.
    assert count != float(np.float32(count))
    archive = io.BytesIO()
    audit_native_call(model.save, archive).assert_native()
    with np.load(io.BytesIO(archive.getvalue()), allow_pickle=False) as fields:
        assert fields["n_samples_seen"].dtype == np.dtype("<f4")
        assert fields["n_samples_seen"].tobytes() == np.array([count], np.float32).tobytes()
    archive.seek(0)
    restored = audit_native_call(ml.StandardScaler.load, archive).assert_native()
    assert restored.n_samples_seen_ == float(np.float32(count))


def test_explicit_host_unfitted_error_uses_its_native_exception_context():
    from mojolearn._classical_host import HostStandardScaler
    from mojolearn.exceptions import NotFittedError
    model = audit_native_call(HostStandardScaler).assert_native()
    audit = audit_native_call(model.transform, _data())
    assert isinstance(audit.exception, NotFittedError)
    assert not audit.frames
    assert not audit.wrapper_work


@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("with_mean,with_std", [(True, True), (True, False), (False, True)])
def test_unseen_column_mask_survives_finite_transform_input(inverse, with_mean, with_std):
    x = _data()
    x[:, 1] = np.nan
    model = ml.StandardScaler(with_mean=with_mean, with_std=with_std).fit(x)
    source = np.nan_to_num(x)
    operation = model.inverse_transform if inverse else model.transform
    result = np.asarray(audit_native_call(operation, source).assert_native())
    assert np.isnan(result[:, 1]).all()
    assert np.isfinite(result[:, [0, 2, 3]]).all()


def test_model_protocol_hooks_are_refused_before_they_run():
    calls = []

    class Input:
        def __array__(self, *args, **kwargs):
            calls.append("array")
            return _data()

    model = ml.StandardScaler().fit(_data())
    audit = audit_native_call(model.transform, Input())
    assert isinstance(audit.exception, TypeError)
    assert not audit.frames
    assert calls == []


def test_unfitted_transform_retains_native_not_fitted_error():
    from mojolearn.exceptions import NotFittedError
    model = ml.StandardScaler()
    audit = audit_native_call(model.transform, _data())
    assert isinstance(audit.exception, NotFittedError)
    assert not audit.frames
    assert not audit.wrapper_work


def test_native_transform_override_and_state_refusals():
    x = _data()
    model = ml.StandardScaler().fit(x)
    with pytest.raises(ValueError):
        model.transform(x, copy=1)
    with pytest.raises(ValueError, match="feature count"):
        model.transform(x[:, :-1])
    object.__getattribute__(model, "__dict__")["scale_"] = np.ones(4, np.float32)
    with pytest.raises(ValueError, match="feature vector"):
        model.transform(x)


def test_numpy_free_memoryview_writeback_and_state_are_native():
    script = r'''
import io, json, sys
# Model an environment without the optional dependency before package import.
import importlib.abc
class BlockNumPy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "numpy" or fullname.startswith("numpy."):
            raise ModuleNotFoundError("NumPy is absent in this fixture", name="numpy")
sys.meta_path.insert(0, BlockNumPy())
import mojolearn as ml
from mojolearn._verify_native_runtime import audit_native_call
from array import array
assert "numpy" not in sys.modules
store = array("f", [1, 2, 3, 4, 5, 6])
x = memoryview(store).cast("B").cast("f", (3, 2))
m = audit_native_call(ml.StandardScaler, copy=False).assert_native()
audit_native_call(m.fit, x).assert_native()
r = audit_native_call(m.transform, x).assert_native()
assert r is x
assert list(store) != [1, 2, 3, 4, 5, 6]
f = io.BytesIO()
audit_native_call(m.save, f).assert_native()
f.seek(0)
audit_native_call(ml.StandardScaler.load, f).assert_native()
assert "numpy" not in sys.modules
print(json.dumps({"passed": True}))
'''
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=os.environ)
    assert result.returncode == 0, result.stderr[-4000:]
    assert json.loads(result.stdout.splitlines()[-1])["passed"]
