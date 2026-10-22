# SPDX-License-Identifier: Apache-2.0
"""Whole-fit boundary and preserved scaler semantics; test oracles stay here."""
import array
import os
import subprocess
import sys

import numpy as np
import pytest

import mojolearn as ml
from mojolearn._array import Array
from mojolearn._verify_native_runtime import audit_native_call


@pytest.mark.parametrize("layout", ["c", "fortran", "strided", "list", "array", "memoryview", "bigendian"])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("missing", [False, True])
def test_whole_fit_is_native_and_preserves_statistics(layout, weighted, missing):
    rng = np.random.default_rng(407)
    x = rng.normal(size=(259, 4)).astype(np.float32)
    x[:, 3] = 2.5
    if missing:
        x[::7, 0] = np.nan
        x[:, 2] = np.nan
    if layout == "fortran":
        source = np.asfortranarray(x)
    elif layout == "strided":
        base = np.zeros((518, 8), np.float32)
        base[::2, ::2] = x
        source = base[::2, ::2]
    elif layout == "list":
        source = x.tolist()
    elif layout == "array":
        source = Array.from_list(x.tolist(), "<f4")
    elif layout == "memoryview":
        source = memoryview(x)
    elif layout == "bigendian":
        source = x.astype(">f4")
    else:
        source = x
    weight = np.linspace(.25, 2, len(x), dtype=np.float32) if weighted else None
    reference = ml.StandardScaler().fit(x, sample_weight=weight)
    model = ml.StandardScaler()
    assert audit_native_call(model.fit, source, sample_weight=weight).assert_native() is model
    for name in ("mean_", "var_", "scale_", "n_samples_seen_"):
        actual = getattr(model, name)
        wanted = getattr(reference, name)
        assert np.asarray(actual).tobytes() == np.asarray(wanted).tobytes(), name
    assert isinstance(model.mean_, Array)


@pytest.mark.parametrize("with_mean,with_std", [(True, True), (True, False), (False, True), (False, False)])
def test_native_flags_and_scalar_weights(with_mean, with_std):
    x = np.arange(36, dtype=np.float32).reshape(12, 3)
    model = ml.StandardScaler(with_mean=with_mean, with_std=with_std)
    audit_native_call(model.fit, x, sample_weight=2.0).assert_native()
    assert (model.mean_ is None) == (not with_mean and not with_std)
    assert (model.var_ is None) == (not with_std)
    assert (model.scale_ is None) == (not with_std)
    assert model.n_samples_seen_ == 24.0


@pytest.mark.parametrize("bad,exception", [
    (np.ones((4, 2), np.float64), TypeError),
    (np.ones((4, 2), np.int32), TypeError),
    (np.array([[np.inf, 1]], np.float32), ValueError),
    (np.array([[-np.inf, 1]], np.float32), ValueError),
    (np.empty((0, 2), np.float32), ValueError),
])
def test_native_refusals(bad, exception):
    with pytest.raises(exception):
        ml.StandardScaler().fit(bad)


def test_native_pandas_nan_and_dtype_contract():
    pd = pytest.importorskip("pandas")
    x = np.array([[1, np.nan], [3, 4], [5, 6]], np.float32)
    model = ml.StandardScaler()
    audit_native_call(model.fit, pd.DataFrame(x)).assert_native()
    np.testing.assert_allclose(np.asarray(model.mean_), [3, 5])
    with pytest.raises(TypeError, match="float32"):
        model.fit(pd.DataFrame(x.astype(np.float64)))


def test_fresh_import_then_numpy_first_fit_is_native():
    code = '''
import sys
import mojolearn as ml
from mojolearn._verify_native_runtime import audit_native_call
assert "numpy" not in sys.modules
import numpy as np
x = np.arange(20, dtype=np.float32).reshape(10, 2)
m = ml.StandardScaler()
audit_native_call(m.fit, x).assert_native()
assert m.n_samples_seen_ == 10
'''
    run = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True)
    assert run.returncode == 0, run.stdout + run.stderr
