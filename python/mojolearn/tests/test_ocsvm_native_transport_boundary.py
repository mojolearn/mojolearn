# SPDX-License-Identifier: Apache-2.0
"""OneClassSVM public storage, scalar admission and native provider seams."""
import gc
import pathlib
import pickle

import numpy as np
import pytest

from mojolearn import OneClassSVM
from mojolearn._array import Array
from mojolearn._buffer import frombytes

__mojolearn_user_boundary__ = True


def _train():
    return np.array([[-1, 0], [1, 0]], dtype=np.float32)


def _query():
    return np.array([[-2, 0], [-0.5, 0], [0.5, 0], [2, 0]], dtype=np.float32)


def _model(**options):
    return OneClassSVM(nu=0.5, tol=1e-6, numeric_mode="identical", **options)


def _scores(gamma):
    x = _query()[:, 0].astype(np.float64)
    return 0.5 * (np.exp(-gamma * (x - 1) ** 2) + np.exp(-gamma * (x + 1) ** 2))


@pytest.mark.parametrize("gamma,resolved", [
    (np.array("auto"), 0.5),
    (np.array("scale"), 1.0),
    (np.array(0.5), 0.5),
    (np.array(0.5, dtype=object), 0.5),
])
def test_ocsvm_native_scalar_parameters_keep_existing_conversion(gamma, resolved):
    model = _model(gamma=gamma).fit(_train())
    assert model._gamma == resolved
    np.testing.assert_allclose(model.score_samples(_query()), _scores(resolved), rtol=0, atol=2e-4)


def test_ocsvm_public_storage_survives_native_scratch_and_pickle():
    model = _model(gamma=0.5).fit(_train())
    dual = np.asarray(model.dual_coef_)
    support = np.asarray(model.support_)
    vectors = np.asarray(model.support_vectors_)
    assert dual.dtype == np.dtype("float32") and dual.shape == (1, 2)
    assert support.dtype == np.dtype("int32") and support.shape == (2,)
    assert model.n_features_in_ == 2 and model.fit_status_ == 0
    assert isinstance(model.n_iter_, int)
    restored = pickle.loads(pickle.dumps(model))
    del model
    gc.collect()
    np.testing.assert_allclose(dual, [[0.5, 0.5]], rtol=0, atol=2e-4)
    np.testing.assert_array_equal(support, [0, 1])
    np.testing.assert_array_equal(vectors, _train())
    expected = _scores(0.5)
    rho = (1 + np.exp(-2)) / 2
    np.testing.assert_allclose(restored.score_samples(_query()), expected, rtol=0, atol=2e-4)
    np.testing.assert_allclose(restored.decision_function(_query()), expected - rho, rtol=0, atol=2e-4)
    labels = np.asarray(restored.predict(_query()))
    assert labels.dtype == np.dtype("int64")
    np.testing.assert_array_equal(labels, [-1, 1, 1, -1])


def test_ocsvm_zero_weight_compaction_publishes_original_row_indices():
    source = np.array([[-5, 0], [-1, 0], [1, 0], [5, 0]], dtype=np.float32)
    model = _model(gamma=0.5).fit(source, sample_weight=[0, 1, 1, 0])
    np.testing.assert_array_equal(model.support_, [1, 2])
    np.testing.assert_array_equal(model.support_vectors_, _train())
    np.testing.assert_allclose(model.score_samples(_query()), _scores(0.5), rtol=0, atol=2e-4)


def test_ocsvm_replaced_coefficients_keep_weak_cache_identity_semantics():
    model = _model(gamma=0.5).fit(_train())
    first = np.asarray(model.score_samples(_query())).copy()
    replacement = np.asarray(model.dual_coef_).copy() * np.float32(2)
    model.dual_coef_ = frombytes(replacement.tobytes(), "<f4", replacement.shape)
    np.testing.assert_allclose(model.score_samples(_query()), first * 2, rtol=0, atol=2e-4)
    np.testing.assert_allclose(model.score_samples(_query()), first * 2, rtol=0, atol=2e-4)


def test_ocsvm_precomputed_index_conversion_does_not_call_array_tolist(monkeypatch):
    model = _model(kernel="precomputed").fit(np.eye(2, dtype=np.float32))
    indices = np.asarray(model.support_, dtype=np.int64)
    model.support_ = frombytes(indices.tobytes(), "<i8", indices.shape)
    calls = []

    def hook(*args):
        calls.append(args)
        raise AssertionError("Python Array.tolist invoked")

    monkeypatch.setattr(Array, "tolist", hook)
    query = np.array([[0.2, 0.2], [1.2, 0], [1, 1]], dtype=np.float32)
    np.testing.assert_allclose(model.score_samples(query), [0.2, 0.6, 1], rtol=0, atol=2e-4)
    np.testing.assert_array_equal(model.predict(query), [-1, 1, 1])
    assert calls == []


@pytest.mark.parametrize("entry", ["x_neighbors_native_api_v1", "ocsvm_resident_native_api_v1"])
def test_ocsvm_rebound_factory_is_refused_without_python_call(monkeypatch, entry):
    monkeypatch.setenv("MOJOLEARN_CLASSICAL_NATIVE_DISPATCH", "0")
    model = _model(gamma="auto")
    calls = []

    def hook():
        calls.append(True)
        raise AssertionError("Python provider invoked")

    monkeypatch.setattr(model._bind(), entry, hook)
    with pytest.raises(TypeError, match="compiled extension entry"):
        model.fit(_train())
    assert calls == []
    assert "_gamma" in vars(model)
    assert "support_" not in vars(model)


@pytest.mark.parametrize("boxed", [False, True])
def test_ocsvm_path_scalar_is_refused_before_comparison_hook(monkeypatch, boxed):
    value = pathlib.PosixPath("not-a-gamma")
    value = np.array(value, dtype=object) if boxed else value
    calls = []

    def hook(*args):
        calls.append(args)
        raise AssertionError("Python parameter comparison invoked")

    monkeypatch.setattr(pathlib.PosixPath, "__eq__", hook)
    with pytest.raises(TypeError, match="native|scalar|protocol|owner"):
        _model(gamma=value).fit(_train())
    assert calls == []


def test_ocsvm_invalid_scalar_array_diagnostic_avoids_numpy_formatter():
    calls = []

    def hook(value):
        calls.append(value)
        raise AssertionError("NumPy formatter invoked")

    with np.printoptions(formatter={"all": hook}):
        with pytest.raises(NotImplementedError, match="native scalar array"):
            _model(kernel=np.array("unsupported")).fit(_train())
    assert calls == []


def test_ocsvm_refusals_keep_gamma_and_fitted_publication_stages():
    negative = _model(gamma="auto")
    with pytest.raises(ValueError, match="negative sample_weight"):
        negative.fit(_train(), sample_weight=[1, -1])
    assert "_gamma" not in vars(negative)
    assert "support_" not in vars(negative)
    iterations = _model(gamma="auto", max_iter="invalid")
    with pytest.raises(ValueError, match="invalid literal"):
        iterations.fit(_train())
    assert iterations._gamma == 0.5
    assert "support_" not in vars(iterations)


def test_ocsvm_rebound_ndarray_type_is_refused_before_dtype_hook(monkeypatch):
    from mojolearn import _expansion_neighbors

    calls = []

    class PretendArray:
        @property
        def dtype(self):
            calls.append("dtype")
            raise AssertionError("Python dtype property invoked")

    model = _model()
    vars(model)["kernel"] = PretendArray()
    monkeypatch.setitem(_expansion_neighbors._NATIVE_CONTEXT, "ndarray", PretendArray)
    with pytest.raises(TypeError, match="native|scalar|protocol|owner"):
        model.fit(_train())
    assert calls == []
    assert "_gamma" not in vars(model)


@pytest.mark.parametrize("rank", [0, 1, 2])
def test_ocsvm_precomputed_array_rank_preserves_native_route(rank):
    kernel = np.array("precomputed").reshape((1,) * rank)
    calls = []

    def formatter(value):
        calls.append(value)
        raise AssertionError("NumPy formatter invoked")

    model = _model(kernel=kernel)
    gram = np.eye(2, dtype=np.float32)
    query = np.array([[0.2, 0.2], [1.2, 0], [1, 1]], dtype=np.float32)
    with np.printoptions(formatter={"all": formatter}):
        if rank == 0:
            model.fit(gram)
            np.testing.assert_allclose(model.score_samples(query), [0.2, 0.6, 1], rtol=0, atol=2e-4)
        else:
            with pytest.raises(TypeError, match="unhashable"):
                model.fit(gram)
            assert model._gamma == 2.0  # non-precomputed gamma publication precedes key refusal
            fitted = _model(gamma=0.5).fit(_train())
            vars(fitted)["kernel"] = kernel
            with pytest.raises(TypeError, match="unhashable"):
                fitted.score_samples(query)
    assert calls == []


@pytest.mark.parametrize("field,operation,gamma_published", [
    ("nu", "fit", False),
    ("degree", "fit", False),
    ("coef0", "fit", False),
    ("max_iter", "fit", True),
    ("tol", "fit", True),
    ("_gamma", "score_samples", True),
    ("offset_", "decision_function", True),
    ("offset_", "predict", True),
])
def test_ocsvm_scalar_conversions_refuse_python_hooks_at_existing_stage(
    monkeypatch, field, operation, gamma_published
):
    calls = []

    def hook(*args):
        calls.append(args)
        raise AssertionError("Python scalar conversion invoked")

    model = _model(gamma="auto")
    if operation != "fit":
        model.fit(_train())
    # Exact Path is admitted by the shared family boundary; its Python
    # conversion hooks are not valid scalar execution inside OneClassSVM.
    vars(model)[field] = pathlib.PosixPath("not-a-number")
    monkeypatch.setattr(pathlib.PosixPath, "__float__", hook, raising=False)
    monkeypatch.setattr(pathlib.PosixPath, "__int__", hook, raising=False)
    with pytest.raises(TypeError, match="native|scalar|protocol|owner"):
        getattr(model, operation)(_train() if operation == "fit" else _query())
    assert calls == []
    assert ("_gamma" in vars(model)) is gamma_published
    if operation == "fit":
        assert "support_" not in vars(model)


@pytest.mark.parametrize("storage", ["scalar", "array", "object_array"])
def test_ocsvm_native_numeric_options_keep_existing_conversion(storage):
    def scalar(value):
        if storage == "scalar":
            return np.int64(value) if isinstance(value, int) else np.float64(value)
        return np.array(value, dtype=object if storage == "object_array" else None)

    model = OneClassSVM(nu=scalar(0.5), degree=scalar(3), coef0=scalar(0.0),
                       max_iter=scalar(1000), tol=scalar(1e-6), gamma=scalar(0.5),
                       numeric_mode="identical").fit(_train())
    vars(model)["_gamma"] = scalar(0.5)
    vars(model)["offset_"] = scalar((1 + np.exp(-2)) / 2)
    np.testing.assert_allclose(model.score_samples(_query()), _scores(0.5), rtol=0, atol=2e-4)
    np.testing.assert_array_equal(model.predict(_query()), [-1, 1, 1, -1])
