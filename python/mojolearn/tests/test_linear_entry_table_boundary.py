# SPDX-License-Identifier: Apache-2.0
"""Canonical connection and public buffer seams for the native linear table.

These are boundary controls, not proof that sys.setprofile observes native
C-API calls. Numerical/native handle contracts have separate Mojo fixtures.

An estimator admits its connection ONCE, when its device route is resolved
(its first operation; providers/python/linear_route_boundary_door.mojo). A
substitute placed in the editable entry dictionary is therefore refused by
the next estimator that resolves a route there, and an estimator whose route
is already resolved keeps running on the images it admitted: the edited
dictionary is no longer consulted per call.
"""
import numpy as np
import pytest

from mojolearn import LinearRegression, Ridge
from mojolearn._native_classical_bridge import connect


def _fit_operation(cls):
    # The classical entry names its fit per estimator (bindings/classical_native_pack.mojo);
    # it never had a generic "fit" operation.
    return "ridge_fit" if cls is Ridge else "ols_fit"


def setup_case(cls):
    entry, context = connect(Ridge, LinearRegression)
    x = np.array([[0, 1], [1, 0], [1, 2], [2, -1], [3, 2]], dtype="f4")
    y = np.array([1, 2, 4, 3, 8], dtype="f4")
    model = cls()
    entry(model, _fit_operation(cls), (x, y, None), context)
    mode = context["backend_fields"]["_DEFAULT_MODE"]
    return entry, context, context["classical_bindings"][mode], model, x, y


@pytest.mark.parametrize("cls", [Ridge, LinearRegression])
@pytest.mark.parametrize("fake", [(1,), (1, 0), 1, None])
def test_connection_refuses_editable_key_or_pin_substitutes(cls, fake, monkeypatch):
    entry, context, entries, model, x, y = setup_case(cls)
    before = np.asarray(entry(model, "predict", (x,), context)).copy()
    monkeypatch.setitem(entries, "_native_entry_connection", fake)
    # The resolved estimator does not read the edited dictionary.
    np.testing.assert_array_equal(np.asarray(entry(model, "predict", (x,), context)), before)
    with pytest.raises(TypeError, match="canonical connection owner"):
        entry(cls(), _fit_operation(cls), (x, y, None), context)


@pytest.mark.parametrize("cls", [Ridge, LinearRegression])
def test_connection_refuses_callback_object_without_hooks(cls, monkeypatch):
    entry, context, entries, model, x, y = setup_case(cls)
    calls = []

    class Hook:
        def __getattribute__(self, name):
            calls.append(name)
            raise AssertionError("connection callback executed")

        def __iter__(self):
            calls.append("iter")
            raise AssertionError("connection callback executed")

    monkeypatch.setitem(entries, "_native_entry_connection", Hook())
    entry(model, "predict", (x,), context)
    with pytest.raises(TypeError, match="canonical connection owner"):
        entry(cls(), _fit_operation(cls), (x, y, None), context)
    assert calls == []


@pytest.mark.parametrize("cls", [Ridge, LinearRegression])
def test_connection_refuses_other_native_family_after_cached_success(cls, monkeypatch):
    entry, context, entries, model, x, y = setup_case(cls)
    entry(model, "predict", (x,), context)
    monkeypatch.setitem(entries, "_native_entry_connection", entries["_native_weighted_connection"])
    entry(model, "predict", (x,), context)
    with pytest.raises(TypeError, match="connection family differs"):
        entry(cls(), _fit_operation(cls), (x, y, None), context)


@pytest.mark.parametrize("cls", [Ridge, LinearRegression])
def test_connection_replacement_rechecks_selected_numeric_mode(cls, monkeypatch):
    entry, context, entries, model, x, y = setup_case(cls)
    mode = context["backend_fields"]["_DEFAULT_MODE"]
    other = context["classical_bindings"]["fast" if mode == "identical" else "identical"]
    if isinstance(other, str):
        pytest.skip("The other numerical tier is not installed on this target")
    entry(model, "predict", (x,), context)
    monkeypatch.setitem(entries, "_native_entry_connection", other["_native_entry_connection"])
    entry(model, "predict", (x,), context)
    with pytest.raises(RuntimeError, match="mode differs"):
        entry(cls(), _fit_operation(cls), (x, y, None), context)


def test_weighted_connection_refuses_estimator_owner(monkeypatch):
    entry, context, entries, _, x, y = setup_case(Ridge)
    monkeypatch.setitem(entries, "_native_weighted_connection", entries["_native_entry_connection"])
    with pytest.raises(TypeError, match="connection family differs"):
        entry(Ridge(), "ridge_fit", (x, y, np.ones(len(x), dtype="f4")), context)


@pytest.mark.parametrize("cls", [Ridge, LinearRegression])
def test_score_connection_replacement_rechecks_selected_metrics_mode(cls, monkeypatch):
    entry, context, _, model, x, y = setup_case(cls)
    mode = context["backend_fields"]["_DEFAULT_MODE"]
    metrics = context["classical_metrics"][mode]
    other = context["classical_metrics"]["fast" if mode == "identical" else "identical"]
    if isinstance(other, str):
        pytest.skip("The other metrics tier is not installed on this target")
    entry(model, "score", (x, y), context)
    monkeypatch.setitem(metrics, "_native_entry_connection", other["_native_entry_connection"])
    with pytest.raises(RuntimeError, match="metrics mode differs"):
        entry(model, "score", (x, y), context)


@pytest.mark.parametrize("cls", [Ridge, LinearRegression])
def test_score_connection_rechecks_selected_metrics_vendor(cls, monkeypatch):
    entry, context, _, model, x, y = setup_case(cls)
    entry(model, "score", (x, y), context)
    vendor = context.get("classical_metric_vendor", context["vendor"])
    # Retain the real canonical metrics owner while changing only the selected
    # metrics vendor. The estimator context vendor remains valid and unchanged.
    monkeypatch.setitem(context, "classical_metric_vendor", "metal" if vendor == "cpu" else "cpu")
    with pytest.raises(RuntimeError, match="metrics vendor differs"):
        entry(model, "score", (x, y), context)


@pytest.mark.parametrize("cls", [Ridge, LinearRegression])
def test_imported_connection_ignores_legacy_capsule_factories(cls, monkeypatch):
    from mojolearn import _backend

    entry, context, entries, _, x, y = setup_case(cls)
    mode = context["backend_fields"]["_DEFAULT_MODE"]
    calls = []

    def forbidden(*args, **kwargs):
        calls.append("legacy factory")
        raise AssertionError("operation reconnected a legacy capsule factory")

    for image, factory in [("_mojolearn_estimators", "linear_fit_native_api_v1"),
                           ("_mojolearn_estimators", "linear_predict_native_api_v1"),
                           ("_mojolearn_x_linear", "ridge_fit_native_api_v1"),
                           ("_mojolearn_x_decomp", "x_decomp_native_api_v1"),
                           ("_mojolearn_x_metrics", "metrics_program_native_api_v1")]:
        # Test setup only: retain and mutate the actual compiled function owner,
        # including when the package exposes a host-binding import proxy.
        function = getattr(_backend.binding(image, mode), factory)
        monkeypatch.setattr(function.__self__, factory, forbidden)
    model = cls()
    entry(model, _fit_operation(cls), (x, y, None), context)
    prediction = entry(model, "predict_public", (x,), context)
    score = entry(model, "score", (x, y), context)
    assert prediction.shape == (len(x),)
    # predict returns the kernel's float32 words to every caller (lane predict-dtype, 2026-11-22).
    assert prediction.dtype == np.dtype("f4")
    assert isinstance(score, float)
    assert calls == []


@pytest.mark.parametrize("cls", [Ridge, LinearRegression])
def test_handle_refresh_preserves_public_coefficient_replacement(cls):
    x = np.array([[0, 1], [1, 0], [1, 2], [2, -1], [3, 2]], dtype="f4")
    y = np.array([1, 2, 4, 3, 8], dtype="f4")
    model = cls().fit(x, y)
    model.coef_ = np.array([2, -3], dtype="f4")
    model.intercept_ = 0.5
    expected = (x @ model.coef_ + np.float32(0.5)).astype("f8")
    np.testing.assert_allclose(np.asarray(model.predict(x)), expected, rtol=1e-6, atol=1e-6)
    # An in-place edit of the replaced coefficient owner reaches the next predict.
    model.coef_[0] = 4
    expected = (x @ model.coef_ + np.float32(0.5)).astype("f8")
    np.testing.assert_allclose(np.asarray(model.predict(x)), expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("position", ["mode", "vendor"])
def test_import_connector_refuses_python_metadata_before_callbacks(position):
    from mojolearn import _backend
    from mojolearn.tokenizer import _binding

    module = _backend.binding("_mojolearn_estimators")
    mode_entry = module.estimators_numeric_mode
    vendor_entry = module.estimators_vendor
    calls = []

    def forbidden():
        calls.append(position)
        raise AssertionError("metadata callback executed")

    if position == "mode":
        mode_entry = forbidden
    else:
        vendor_entry = forbidden
    with pytest.raises(TypeError, match="compiled metadata"):
        _binding()._module.linear_entry_connection_v1(
            0, mode_entry, vendor_entry, module.linear_fit_native_api_v1)
    assert calls == []


def test_import_connector_checks_compiled_owner_before_calling_metadata():
    from mojolearn import _backend
    from mojolearn.tokenizer import _binding

    module = _backend.binding("_mojolearn_estimators")
    # len is a real compiled builtin, but belongs to another exact module.
    # Calling it without arguments would produce a different TypeError.
    with pytest.raises(TypeError, match="share its compiled factory image"):
        _binding()._module.linear_entry_connection_v1(
            0, len, module.estimators_vendor, module.linear_fit_native_api_v1)
