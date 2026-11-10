# SPDX-License-Identifier: Apache-2.0
"""Generic Ridge/OLS archive routing and writable coefficient ownership.

Typed numerical schema/refusal coverage remains in archive_linear.mojo.
These seams exercise the public owner after its source and old buffer die.
"""
import io
import zipfile

import numpy as np
import pytest

import mojolearn
from mojolearn import Array, _classical_host as host
from mojolearn._verify_native_runtime import audit_native_call


def _saved_bytes(estimator, order):
    fields = dict(
        format="mojolearn-linear-1", estimator=estimator, numeric_mode="identical",
        meta=np.array([2, 1], dtype="<i8"), coef=np.array([1.0, -2.0], dtype="<f4"),
        intercept=np.array([0.5], dtype="<f8"),
    )
    if estimator == "Ridge":
        fields["alpha"] = np.array([1.25], dtype="<f8")
    stream = io.BytesIO()
    np.savez(stream, **fields)
    if order == "C":
        return stream.getvalue()
    vector = io.BytesIO()
    np.lib.format.write_array_header_1_0(vector, {
        "descr": "<f4", "fortran_order": True, "shape": (2,),
    })
    vector.write(fields["coef"].tobytes())
    rewritten = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(stream.getvalue())) as original:
        with zipfile.ZipFile(rewritten, "w") as saved:
            for name in original.namelist():
                saved.writestr(name, vector.getvalue() if name == "coef.npy" else original.read(name))
    return rewritten.getvalue()


@pytest.mark.parametrize("estimator", ["Ridge", "LinearRegression"])
@pytest.mark.parametrize("route", ["host_model", "classical_host_model", "requested_class"])
@pytest.mark.parametrize("order", ["C", "F"])
def test_generic_linear_keeps_writable_and_rebound_coefficient_owners(estimator, route, order):
    source = io.BytesIO(_saved_bytes(estimator, order))
    cls = getattr(host, "Host" + estimator)
    if route == "host_model":
        model = audit_native_call(mojolearn.host_model, source).assert_native()
    elif route == "classical_host_model":
        model = audit_native_call(host.host_model, source).assert_native()
    else:
        model = audit_native_call(host._native_classical, cls, "host_archive_load",
                                  (source,), host._CLASSICAL).assert_native()
    assert type(model) is cls
    assert not source.closed
    source.close()
    assert type(model.coef_) is Array
    assert model.coef_.shape == (2,) and model.coef_.order == order
    assert np.asarray(model.coef_).flags.writeable
    np.testing.assert_array_equal(np.asarray(model.coef_), [1.0, -2.0])
    assert model.fit_intercept and model.intercept_ == 0.5 and model.n_features_in_ == 2
    if estimator == "Ridge":
        assert model.alpha == 1.25 and model.solver_ == "eig"
    # This exercises reconciliation of a published writable owner with native
    # fitted state, followed by public rebinding and release of the old owner.
    x = np.array([[1.0, 0.0]], dtype="<f4")
    model.coef_[0] = 3.0
    np.testing.assert_array_equal(audit_native_call(model.predict, x).assert_native(), [3.5])
    model.coef_ = np.array([4.0, -5.0], dtype="<f4")
    np.testing.assert_array_equal(audit_native_call(model.predict, x).assert_native(), [4.5])
    saved = io.BytesIO()
    audit_native_call(model.save, saved).assert_native()
    assert not saved.closed
    saved.seek(0)
    with np.load(saved, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["coef"], [4.0, -5.0])
        np.testing.assert_array_equal(archive["intercept"], [0.5])
