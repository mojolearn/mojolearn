# SPDX-License-Identifier: Apache-2.0
"""Neighbors public boundary: the caller's buffers, admission and returned objects."""
import os

import numpy as np

# These are public-caller contracts. Public wrappers tell a user call from a library-internal one by the
# caller's module (providers/python/estimator_lifecycle_pyapi_unpack.mojo:16 estimator_outside): this module
# lives in mojolearn.tests, so without the marker its calls count as internal.
__mojolearn_user_boundary__ = True
import pytest

os.environ.setdefault("MOJOLEARN_NUMERIC_MODE", "identical")

import mojolearn as ml  # noqa: E402


# ---------------------------------------------------------------- moved from test_c3_c5_no_copies.py (vdv-pytest-a)
# 0.8.66 regression C3: a k-NN fit keeps a caller's float32 C matrix by reference (as scikit-learn does) and never
# freezes or writes it.


def _c3_x(rows=64, cols=6, seed=3):
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(rng.standard_normal((rows, cols)).astype(np.float32))


@pytest.mark.parametrize("cls", ["NearestNeighbors", "KNeighborsRegressor", "KNeighborsClassifier"])
def test_knn_fit_keeps_the_float32_matrix_by_reference_and_never_changes_it(cls):
    X = _c3_x()
    y = (np.arange(len(X)) % 3).astype(np.int64)
    reference = X.copy()
    a = getattr(ml, cls)(n_neighbors=3)
    b = getattr(ml, cls)(n_neighbors=3)
    a.fit(X) if cls == "NearestNeighbors" else a.fit(X, y)
    b.fit(reference.copy()) if cls == "NearestNeighbors" else b.fit(reference.copy(), y)
    # The caller's array is never frozen or written.
    assert X.flags.writeable
    assert X.tobytes() == reference.tobytes()
    da, ia = a.kneighbors(reference[:5])
    db, ib = b.kneighbors(reference[:5])
    assert np.asarray(ia).tobytes() == np.asarray(ib).tobytes()
    assert np.asarray(da).tobytes() == np.asarray(db).tobytes()
    # The documented contract: after an in-place edit, fit again.
    X[:] = X[::-1].copy()
    a.fit(X) if cls == "NearestNeighbors" else a.fit(X, y)
    c = getattr(ml, cls)(n_neighbors=3)
    c.fit(X.copy()) if cls == "NearestNeighbors" else c.fit(X.copy(), y)
    assert np.asarray(a.kneighbors(reference[:5])[1]).tobytes() == np.asarray(c.kneighbors(reference[:5])[1]).tobytes()


def test_knn_fit_casts_a_strided_view_and_leaves_its_owner_alone():
    base = _c3_x(rows=128)
    before_bytes = base.tobytes()
    model = ml.NearestNeighbors(n_neighbors=2).fit(base[::2])
    assert base.flags.writeable and base.tobytes() == before_bytes
    query = base[::2][:4].copy()
    before = np.asarray(model.kneighbors(query)[1]).tobytes()
    base[:] = 0.0  # the strided view was cast into the model's own Array
    assert np.asarray(model.kneighbors(query)[1]).tobytes() == before


# ---------------------------------------------------------------- moved from test_knn_metrics.py (vdv-pytest-a g2)
# Brute-force k-NN under the extra metrics (lane x-neighbors-metrics): the refusals of a metric the build does
# not support, of input a metric needs non-negative, of a tree algorithm for a brute-force-only metric and of a
# similarity under distance weights.

def _knn_metrics_data(n=200, d=7, seed=6):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d)).astype(np.float32)


def test_knn_metric_refusals():
    x = _knn_metrics_data()
    with pytest.raises(ValueError, match="haversine"):
        ml.NearestNeighbors(metric="haversine").fit(x[:, :2])
    with pytest.raises(ValueError, match="X >= 0"):
        ml.NearestNeighbors(metric="jensenshannon").fit(x)
    with pytest.raises(ValueError, match="brute force only"):
        ml.NearestNeighbors(metric="canberra", algorithm="rbc").fit(x)
    with pytest.raises(ValueError, match="similarity"):
        ml.KNeighborsRegressor(metric="inner_product", weights="distance").fit(x, x[:, 0])


# ---- appended by vdv-pytest-b (2026-11-24) ----
# moved from python/mojolearn/tests/test_native_scalar_decode_public.py
# Native buffer scalar conversion of KernelDensity metadata through the public save.
import io
import struct

import numpy as np
import pytest

from mojolearn._verify_native_runtime import audit_native_call


@pytest.mark.parametrize('dtype,bits,array', [
    pytest.param('f4', 0x7f800001, False, id='f4-snan'),
    pytest.param('f8', 0x7ff0000000000001, False, id='f8-snan'),
    pytest.param('f4', 0x80000000, False, id='f4-negative-zero'),
    pytest.param('f8', 1, False, id='f8-smallest-subnormal'),
    pytest.param('u8', 0xffffffffffffffff, False, id='u8-max'),
    pytest.param('i8', 0x8000000000000000, False, id='i8-min'),
    pytest.param('f4', 0x7f800001, True, id='zero-d-f4-snan'),
    pytest.param('f8', 0x8000000000000000, True, id='zero-d-f8-negative-zero'),
])
def test_save_consumes_exact_native_scalar_conversion_bytes(dtype, bits, array):
    from mojolearn.tests.test_host_density_save_native import _prepared

    # Raw storage preserves signaling NaNs until the conversion under test.
    storage_type = np.dtype(dtype).newbyteorder('<')
    scalar = np.frombuffer(bits.to_bytes(storage_type.itemsize, 'little'),
                           dtype=storage_type).copy()[0]
    value = np.array(scalar) if array else scalar
    expected = struct.pack('<d', float(value))
    model = _prepared('kde')
    model.bandwidth = value
    output = io.BytesIO()
    audit = audit_native_call(model.save, output)
    assert audit.assert_native() is output
    assert model.bandwidth is value
    # Save accepts these metadata values; inference/load validation is a
    # separate interface. Consume the saved bytes, not just error/truth state.
    with np.load(io.BytesIO(output.getvalue()), allow_pickle=False) as saved:
        assert saved['bandwidth'].tobytes() == expected


@pytest.mark.parametrize('value', ['negative_zero', 'adjacent_half', 'smallest_normal'])
def test_save_consumes_non_native_longdouble_scalar_bytes(value):
    from mojolearn.tests.test_host_density_save_native import _prepared

    scalar = (np.longdouble('-0') if value == 'negative_zero' else
              np.nextafter(np.longdouble('.5'), np.longdouble(1)) if value == 'adjacent_half' else
              np.finfo(np.longdouble).tiny)
    supplied = np.asarray(scalar, dtype=np.dtype('g').newbyteorder('>'))
    expected = struct.pack('<d', float(supplied))
    before = supplied.tobytes()
    model = _prepared('kde')
    model.bandwidth = supplied
    output = io.BytesIO()
    assert audit_native_call(model.save, output).assert_native() is output
    assert model.bandwidth is supplied and supplied.tobytes() == before
    with np.load(io.BytesIO(output.getvalue()), allow_pickle=False) as saved:
        assert saved['bandwidth'].tobytes() == expected
