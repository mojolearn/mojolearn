# SPDX-License-Identifier: Apache-2.0
"""PCA compiled-provider, mutable-public-state and native-cache boundary seams."""
import copy
import gc
import weakref

import numpy as np
import pytest
import mojolearn as ml


def _data():
    return np.array([
        [-4, 1, 0, 2], [-2, -1, 1, 0], [-1, 2, -2, 1], [0, -2, 2, -1],
        [1, 0, -1, 3], [2, 3, 2, 1], [3, -1, 0, -2], [5, 2, 1, 0],
    ], dtype=np.float32)


def _model(**settings):
    options = dict(n_components=2, svd_solver="covariance_eigh", numeric_mode="identical", random_state=7)
    options.update(settings)
    return ml.PCA(**options)


@pytest.mark.parametrize("solver", ["covariance_eigh", "full", "randomized"])
@pytest.mark.parametrize("method", ["fit", "fit_transform"])
def test_pca_uses_native_provider_without_legacy_python_dispatch(solver, method, monkeypatch):
    from mojolearn import _backend

    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("legacy Python PCA dispatch executed")

    binding = _backend.binding("_mojolearn_estimators", "identical")
    for name in (
        "pca_fit_owned", "pca_fit_full", "pca_fit_randomized", "pca_transform", "inverse_transform",
        "pca_whiten_transform", "pca_whiten_inverse_transform",
    ):
        if hasattr(binding, name):
            monkeypatch.setattr(binding, name, forbidden)
    model = _model(svd_solver=solver, whiten=True)
    result = getattr(model, method)(_data())
    scores = np.asarray(model.transform(_data()))
    assert scores.shape == (8, 2)
    assert np.isfinite(scores).all()
    assert np.asarray(model.inverse_transform(scores)).shape == (8, 4)
    if method == "fit":
        assert result is model
    else:
        np.testing.assert_allclose(np.asarray(result), scores, rtol=2e-4, atol=2e-4)
    assert model.numeric_mode_used() == "identical"
    assert isinstance(model.vendor_used(), str)
    assert calls == []


@pytest.mark.parametrize("field", ["mean_", "components_"])
@pytest.mark.parametrize("rebind", [False, True])
def test_cached_query_observes_mutable_fitted_arrays(field, rebind):
    model = _model().fit(_data())
    model.transform(_data())  # populate the immutable admission cache
    value = getattr(model, field)
    if rebind:
        value = np.asarray(value).copy()
        setattr(model, field, value)
    value.flat[0] += np.float32(0.25)
    expected = (_data() - np.asarray(model.mean_)) @ np.asarray(model.components_).T
    np.testing.assert_allclose(np.asarray(model.transform(_data())), expected, rtol=2e-4, atol=2e-4)


def test_mutable_whitening_setting_is_decoded_on_each_query():
    model = _model().fit(_data())
    value = np.array(False)
    model.whiten = value
    plain = np.asarray(model.transform(_data()))
    value[()] = True
    expected = plain * np.sqrt(np.float32(model.n_samples_ - 1)) / np.asarray(model.singular_values_)
    np.testing.assert_allclose(np.asarray(model.transform(_data())), expected, rtol=2e-4, atol=2e-4)


def test_mutable_dimension_setting_is_not_cached_by_identity():
    model = _model().fit(_data())
    value = np.array(model.n_features_in_, dtype=np.int64)
    model.n_features_in_ = value
    model.transform(_data())
    value[()] = 5
    with pytest.raises(ValueError, match="features"):
        model.transform(_data())


@pytest.mark.parametrize("operation", ["fit", "transform", "inverse_transform"])
def test_replaced_native_factory_is_refused_before_python_callback(operation, monkeypatch):
    from mojolearn.decomposition import _PCA_CONTEXT

    calls = []

    def forbidden():
        calls.append(True)
        raise AssertionError("Python provider factory executed")

    model = _model().fit(_data())
    scores = model.transform(_data())  # populate both connection and query caches
    monkeypatch.setitem(_PCA_CONTEXT["pca_bindings"], "identical", forbidden)
    source = scores if operation == "inverse_transform" else _data()
    with pytest.raises(TypeError, match="compiled extension entry"):
        getattr(model, operation)(source)
    assert calls == []


@pytest.mark.parametrize("copier", [copy.copy, copy.deepcopy])
def test_native_query_cache_does_not_serialize_or_keep_model_alive(copier):
    model = _model().fit(_data())
    expected = np.asarray(model.transform(_data()))
    restored = copier(model)
    np.testing.assert_allclose(np.asarray(restored.transform(_data())), expected, rtol=2e-4, atol=2e-4)
    reference = weakref.ref(model)
    del model
    gc.collect()
    assert reference() is None


@pytest.mark.parametrize("field", ["mean_", "components_", "singular_values_", "explained_variance_"])
def test_whitened_query_rechecks_in_place_fitted_values(field):
    model = _model(whiten=True).fit(_data())
    model.transform(_data())
    getattr(model, field).flat[0] = np.nan
    with pytest.raises(ValueError):
        model.transform(_data())


def test_changed_factory_keeps_field_admission_before_provider_refusal(monkeypatch):
    from mojolearn.decomposition import _PCA_CONTEXT

    calls = []

    def factory():
        calls.append("factory")
        raise AssertionError("Python factory executed")

    class Scalar:
        def __int__(self):
            calls.append("scalar")
            raise AssertionError("Python scalar executed")

    model = _model().fit(_data())
    model.transform(_data())
    model.n_features_in_ = Scalar()
    monkeypatch.setitem(_PCA_CONTEXT["pca_bindings"], "identical", factory)
    with pytest.raises(TypeError) as error:
        model.transform(_data())
    assert "compiled extension entry" not in str(error.value)
    assert calls == []


def test_query_width_refusal_precedes_fitted_buffer_layout():
    model = _model().fit(_data())
    model.components_ = np.asarray(model.components_, dtype=np.float64)
    with pytest.raises(ValueError, match="features"):
        model.transform(_data()[:, :3])


def test_whitening_fitted_scan_precedes_fit_row_refusal():
    model = _model(whiten=True).fit(_data())
    model.n_samples_ = 1
    model.components_.flat[0] = np.nan
    with pytest.raises(ValueError, match="finite components_"):
        model.transform(_data())
