# SPDX-License-Identifier: Apache-2.0
"""Installed TSVD provider, storage, publication and callback boundary seams."""
import copy
import gc
import weakref

import numpy as np
import pytest
import mojolearn as ml


def _data():
    return np.array([[1, 2, 0], [3, 0, 1], [0, 1, 4], [2, 3, 1], [1, 0, 2], [4, 1, 0]], dtype=np.float32)


def _model(**settings):
    options = dict(n_components=2, algorithm="covariance_eigh", random_state=7, numeric_mode="identical")
    options.update(settings)
    return ml.TruncatedSVD(**options)


@pytest.mark.parametrize("algorithm", ["covariance_eigh", "jacobi", "randomized"])
@pytest.mark.parametrize("method", ["fit", "fit_transform"])
def test_tsvd_uses_descriptors_without_legacy_python_dispatch(algorithm, method, monkeypatch):
    from mojolearn import _backend

    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("legacy Python TSVD dispatch executed")

    for basename, names in (
        ("_mojolearn_estimators", ("tsvd_fit", "tsvd_fit_explained", "tsvd_explained", "tsvd_transform",
                                    "inverse_transform", "tsvd_resident_lease", "tsvd_transform_resident",
                                    "tsvd_inverse_transform_resident")),
        ("_mojolearn_x_decomp", ("x_decomp_qfix_flags", "x_decomp_tsvd_randomized", "x_decomp_tsvd_tsqr")),
    ):
        binding = _backend.binding(basename, "identical")
        for name in names:
            if hasattr(binding, name):
                monkeypatch.setattr(binding, name, forbidden)
    model = _model(algorithm=algorithm)
    result = getattr(model, method)(_data())
    scores = np.asarray(model.transform(_data()))
    assert scores.shape == (6, 2)
    assert np.isfinite(scores).all()
    assert np.asarray(model.inverse_transform(scores)).shape == (6, 3)
    if method == "fit":
        assert result is model
    else:
        np.testing.assert_allclose(np.asarray(result), scores, rtol=2e-4, atol=2e-4)
    reference = weakref.ref(model)
    if result is model:
        result = None
    del model
    gc.collect()
    assert reference() is None
    assert calls == []


@pytest.mark.parametrize("inverse", [False, True])
def test_rebound_writable_components_are_observed_in_place(inverse):
    model = _model().fit(_data())
    model.transform(_data())
    components = np.asarray(model.components_).copy()
    model.components_ = components
    source = np.ones((2, 2), dtype=np.float32) if inverse else _data()
    operation = model.inverse_transform if inverse else model.transform
    operation(source)
    components[0, 0] += np.float32(0.5)
    expected = source @ components if inverse else source @ components.T
    np.testing.assert_allclose(np.asarray(operation(source)), expected, rtol=2e-4, atol=2e-4)


@pytest.mark.parametrize("operation", ["fit", "fit_transform", "transform", "inverse_transform"])
def test_replaced_estimator_factory_cannot_call_python(operation, monkeypatch):
    from mojolearn.decomposition import _TSVD_CONTEXT

    calls = []

    def forbidden():
        calls.append(True)
        raise AssertionError("Python provider factory executed")

    model = _model().fit(_data())
    scores = model.transform(_data())
    monkeypatch.setitem(_TSVD_CONTEXT["tsvd_bindings"], "identical", forbidden)
    source = scores if operation == "inverse_transform" else _data()
    with pytest.raises(TypeError, match="compiled extension entry"):
        getattr(model, operation)(source)
    assert calls == []


@pytest.mark.parametrize("field", ["tol", "n_components", "n_iter", "n_oversamples", "random_state", "algorithm"])
@pytest.mark.parametrize("kind", ["object", "int_subclass", "object_array"])
def test_rebound_fit_scalars_refuse_conversion_and_repr_hooks(field, kind):
    calls = []

    class Hooks:
        def __int__(self):
            calls.append("int")
            return 2

        def __float__(self):
            calls.append("float")
            return 0.0

        def __repr__(self):
            calls.append("repr")
            return "hook"

        @property
        def __class__(self):
            calls.append("class")
            return int

    class Integer(int):
        def __int__(self):
            calls.append("int")
            return 2

        def __float__(self):
            calls.append("float")
            return 0.0

        def __repr__(self):
            calls.append("repr")
            return "hook"

    value = Hooks() if kind == "object" else Integer(2)
    if kind == "object_array":
        value = np.empty((), dtype=object)
        value[()] = Hooks()
    model = _model(algorithm="randomized")
    vars(model)[field] = value
    with pytest.raises(TypeError, match="native values|integer seed"):
        model.fit(_data())
    assert calls == []


@pytest.mark.parametrize("field", ["n_features_in_", "n_components_"])
def test_rebound_query_dimensions_refuse_integer_hooks(field):
    calls = []

    class Integer(int):
        def __int__(self):
            calls.append(True)
            return 2

    model = _model().fit(_data())
    vars(model)[field] = Integer(getattr(model, field))
    with pytest.raises(TypeError, match="native values"):
        model.transform(_data())
    assert calls == []


def test_tol_refusal_precedes_algorithm_repr_and_input_admission():
    calls = []

    class Algorithm:
        def __repr__(self):
            calls.append(True)
            return "arpack"

    model = _model(tol=1)
    vars(model)["algorithm"] = Algorithm()
    with pytest.raises(NotImplementedError, match="tol is ARPACK"):
        model.fit(_data())
    assert "input_copied_" not in vars(model)
    assert calls == []


def test_component_refusal_keeps_input_copied_publication():
    model = _model(n_components=0)
    with pytest.raises(ValueError, match="n_components"):
        model.fit(_data())
    assert model.input_copied_ is False
    assert "components_" not in vars(model)


@pytest.mark.parametrize("algorithm", ["covariance_eigh", "randomized"])
def test_failed_finite_fit_preserves_existing_publication_stages(algorithm):
    model = _model(algorithm=algorithm)
    source = _data()
    source[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN|infinity|finite"):
        model.fit(source)
    fields = vars(model)
    assert fields["input_copied_"] is False
    assert ("components_" in fields) is (algorithm == "covariance_eigh")
    assert ("explained_variance_" in fields) is (algorithm == "covariance_eigh")
    assert "n_components_" not in fields


@pytest.mark.parametrize("copier", [copy.copy, copy.deepcopy])
def test_resident_owner_stays_out_of_serialization_and_does_not_retain_model(copier):
    model = _model().fit(_data())
    expected = np.asarray(model.transform(_data()))
    restored = copier(model)
    np.testing.assert_allclose(np.asarray(restored.transform(_data())), expected, rtol=2e-4, atol=2e-4)
    reference = weakref.ref(model)
    del model
    gc.collect()
    assert reference() is None


def test_resident_lease_reuses_then_releases_on_failed_refit():
    from mojolearn import _backend

    binding = _backend.binding("_mojolearn_estimators", "identical")
    if not hasattr(binding, "tsvd_resident_stats"):
        pytest.skip("CPU provider has no device-resident TSVD store")
    gc.collect()
    before = binding.tsvd_resident_stats()
    model = _model().fit(_data())
    scores = model.transform(_data())
    first = binding.tsvd_resident_stats()
    assert first[0] == before[0] + 1
    assert first[3] == before[3] + 1
    model.transform(_data())
    model.inverse_transform(scores)
    assert binding.tsvd_resident_stats()[3] == first[3]
    model.tol = 1
    with pytest.raises(NotImplementedError, match="tol is ARPACK"):
        model.fit(_data())
    assert binding.tsvd_resident_stats()[0] == before[0]


def test_explicit_host_fit_does_not_call_python_bind(monkeypatch):
    from mojolearn._classical_host import HostTruncatedSVD, _HostBound

    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("Python host binding dispatch executed")

    monkeypatch.setattr(_HostBound, "_bind", forbidden)
    model = HostTruncatedSVD(n_components=2, algorithm="covariance_eigh", numeric_mode="identical")
    assert model.fit(_data()) is model
    assert np.asarray(model.components_).shape == (2, 3)
    assert calls == []
