# SPDX-License-Identifier: Apache-2.0
"""KDE public admission, snapshots and native ownership; numeric suites are Mojo."""
import copy
import gc
import weakref

import numpy as np
import pytest
import mojolearn as ml


def _x():
    return np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)


def _model(**options):
    settings = dict(bandwidth=0.75, numeric_mode="identical")
    settings.update(options)
    return ml.KernelDensity(**settings)


def test_kde_public_calls_do_not_dispatch_legacy_argument_lists(monkeypatch):
    from mojolearn import _backend

    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("legacy KDE entry was called")

    binding = _backend.binding("_mojolearn_estimators", "identical")
    for name in ("kde_fit_prepare", "kde_fit_release", "kde_fast_immutable_fit", "kde_score_samples", "kde_score_samples_resident"):
        monkeypatch.setattr(binding, name, forbidden, raising=False)
    model = _model().fit(_x(), sample_weight=np.array([1, 2, 1], dtype=np.float32))
    assert np.asarray(model.score_samples(_x())).shape == (3,)
    assert isinstance(model.score(_x()), float)
    assert model._release_resident_fit() is None
    assert model.fit(_x()) is model
    assert calls == []


@pytest.mark.parametrize("operation", ["fit", "score_samples", "score"])
def test_kde_python_descriptor_factory_is_refused_before_invocation(operation, monkeypatch):
    from mojolearn import _backend

    model = _model().fit(_x())
    calls = []

    def forbidden():
        calls.append(True)
        raise AssertionError("Python KDE descriptor factory was called")

    binding = _backend.binding("_mojolearn_estimators", "identical")
    monkeypatch.setattr(binding, "kde_native_api_v1", forbidden)
    with pytest.raises(TypeError, match="compiled extension entry"):
        getattr(model, operation)(_x())
    assert calls == []


@pytest.mark.parametrize("field", ["bandwidth", "algorithm", "kernel", "metric", "atol", "rtol", "breadth_first", "leaf_size", "n_features_in_"])
def test_kde_cached_state_refuses_rebound_scalar_protocols(field):
    model = _model().fit(_x())
    model.score_samples(_x())
    calls = []

    class Protocol:
        @property
        def __class__(self):
            calls.append("class")
            return float

        def __float__(self):
            calls.append("float")
            return 0.75

        def __int__(self):
            calls.append("int")
            return 2

        def __bool__(self):
            calls.append("bool")
            return True

        def __eq__(self, other):
            calls.append("eq")
            return True

        def __str__(self):
            calls.append("str")
            return "gaussian"

    setattr(model, field, Protocol())
    with pytest.raises(TypeError, match="native|hooks"):
        model.score_samples(_x())
    assert calls == []


def test_kde_cached_metric_mapping_mutation_is_readmitted_without_hooks():
    params = {"p": 2.0}
    model = _model(metric_params=params).fit(_x())
    model.score_samples(_x())
    calls = []

    class Conversion:
        def __float__(self):
            calls.append(True)
            return 2.0

    params["p"] = Conversion()
    with pytest.raises(TypeError, match="native|hooks"):
        model.score_samples(_x())
    assert calls == []


def test_kde_invalid_array_metric_parameter_avoids_numpy_formatters():
    calls = []

    def formatter(value):
        calls.append(True)
        raise AssertionError("NumPy array formatting callback was called")

    with np.printoptions(formatter={"all": formatter}):
        with pytest.raises(NotImplementedError, match="Minkowski p=3"):
            _model(metric_params={"p": np.array(3.0)}).fit(_x())
    assert calls == []


def test_kde_fit_keeps_caller_snapshots_and_readonly_public_storage():
    x = _x()
    weights = np.array([1, 2, 1], dtype=np.float32)
    model = _model().fit(x, sample_weight=weights)
    before = np.asarray(model.score_samples(_x())).copy()
    x[:] = 100
    weights[:] = 10
    np.testing.assert_array_equal(np.asarray(model.score_samples(_x())), before)
    assert not np.asarray(model._x).flags.writeable
    assert not np.asarray(model._w).flags.writeable


def test_kde_rebound_public_fit_arrays_and_bandwidth_invalidate_residency():
    model = _model().fit(_x())
    model.score_samples(_x())
    x = np.array([[1, 1], [2, 2], [4, 4]], dtype=np.float32)
    weights = np.array([2, 1, 3], dtype=np.float32)
    model._x, model._w, model.bandwidth = x, weights, np.float64(1.25)
    expected = _model(bandwidth=1.25).fit(x, sample_weight=weights)
    np.testing.assert_allclose(np.asarray(model.score_samples(_x())), np.asarray(expected.score_samples(_x())), rtol=1e-6, atol=1e-6)
    assert model._x is x and model._w is weights


def test_kde_pickle_state_omits_owner_and_release_remains_reusable():
    model = _model().fit(_x())
    before = np.asarray(model.score_samples(_x())).copy()
    assert "_resident" not in model.__getstate__()
    restored = copy.deepcopy(model)
    assert vars(restored).get("_resident") is None
    assert model._release_resident_fit() is None
    assert vars(model).get("_resident") is None
    np.testing.assert_array_equal(np.asarray(restored.score_samples(_x())), before)
    np.testing.assert_array_equal(np.asarray(model.score_samples(_x())), before)
    assert model._release_resident_fit() is None
    assert model._release_resident_fit() is None


def test_kde_private_tuple_cannot_inject_a_handle_or_release_callback():
    model = _model().fit(_x())
    calls = []

    class Provider:
        def kde_fit_release(self, handle):
            calls.append(handle)

    model._resident = ((), 0, Provider())
    with pytest.raises(TypeError, match="native owner"):
        model._release_resident_fit()
    assert calls == []
    del model._resident


def test_kde_native_cache_and_resident_owner_do_not_keep_model_alive():
    model = _model().fit(_x())
    model.score_samples(_x())
    reference = weakref.ref(model)
    del model
    gc.collect()
    assert reference() is None


@pytest.mark.parametrize("bandwidth,kernel,metric", [(np.float32(0.75), np.str_("gaussian"), np.str_("euclidean")), (np.array(0.75), "gaussian", "euclidean")])
def test_kde_native_scalar_forms_remain_admitted(bandwidth, kernel, metric):
    model = _model(bandwidth=bandwidth, kernel=kernel, metric=metric).fit(_x())
    assert np.asarray(model.score_samples(_x())).shape == (3,)
