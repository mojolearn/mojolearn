"""Observable LinearSVC storage, label PACK, weight and provider seams."""
import gc
import pickle

import numpy as np
import pytest

from mojolearn.svm import LinearSVC, LinearSVR

__mojolearn_user_boundary__ = True


def _points():
    return np.array([[-1, 0], [1, 0]] * 4, dtype=np.float32)


def _fit(labels=(-5, 9), **options):
    model = LinearSVC(C=2, tol=1e-6, max_iter=200, numeric_mode="identical", **options)
    model.penalty_normalized = False
    return model.fit(_points(), [labels[i % 2] for i in range(8)])


@pytest.mark.parametrize("labels", [(-5, 9), (2**70, 2**70 + 1), ("low", "top")])
def test_native_codes_pack_original_label_dtype_and_huge_integer_semantics(labels):
    model = _fit(labels)
    actual = model.predict(_points())
    expected = [labels[i % 2] for i in range(8)]
    assert list(actual) == expected
    if isinstance(labels[0], str) or labels[0] > 2**63 - 1:
        assert isinstance(actual, list)
    else:
        assert np.asarray(actual).dtype == np.int64


def test_query_reads_mutated_native_weight_storage_and_rebound_class_labels():
    model = _fit()
    # Warm the weak label table, then mutate the actual historical query
    # owner. coef_ is a separate public copy, as before the migration.
    model.predict(_points())
    weights = np.asarray(model._w)
    weights[:] = [2, -1, 0.5]
    model.classes_[:] = [100, 200]
    query = np.array([[1, 3], [2, 1]], dtype=np.float32)
    np.testing.assert_allclose(model.decision_function(query), [-0.5, 3.5], rtol=0, atol=1e-6)
    np.testing.assert_array_equal(model.predict(query), [100, 200])
    assert model._w is vars(model)["_w"]
    assert not np.shares_memory(np.asarray(model.coef_), weights)


@pytest.mark.parametrize("class_weight", ["balanced", {-5: 1.0, 9: 1.0}])
def test_native_all_one_class_weight_retains_scalar_public_result_forms(class_weight):
    model = _fit(class_weight=class_weight)
    np.testing.assert_allclose(model.coef_, [[0.8, 0]], rtol=0, atol=2e-4)
    np.testing.assert_allclose(model.intercept_, [0], rtol=0, atol=2e-4)
    assert isinstance(model.objective_, float)
    assert isinstance(model.retcode_, int)
    assert abs(model.objective_ - 0.2) < 2e-4
    assert np.asarray(model.n_iter_).shape == (1,)
    assert np.asarray(model.n_iter_).dtype == np.int64


def test_sample_and_dictionary_weights_keep_existing_multiply_objective():
    model = LinearSVC(C=2, class_weight={-5: 1.0, 9: 3.0}, tol=1e-6,
                      max_iter=200, numeric_mode="identical")
    model.penalty_normalized = False
    model.fit(_points(), [-5, 9] * 4, sample_weight=np.full(8, 2, dtype=np.float32))
    # Weights2/6 give A=3 after optimizing b=(1-w)/2; lambda=1/C=1/2.
    np.testing.assert_allclose(model.coef_, [[12 / 13, 0]], rtol=0, atol=2e-4)
    np.testing.assert_allclose(model.intercept_, [1 / 26], rtol=0, atol=2e-4)
    assert abs(model.objective_ - 3 / 13) < 2e-4


def test_multiclass_public_metadata_and_exporter_lifetime_survive_plan_release():
    points = np.tile(np.eye(3, dtype=np.float32), (4, 1))
    model = LinearSVC(C=2, fit_intercept=False, tol=1e-6, max_iter=200,
                      numeric_mode="identical")
    model.penalty_normalized = False
    model.fit(points, ["a", "b", "c"] * 4)
    assert isinstance(model.objective_, list) and len(model.objective_) == 3
    assert isinstance(model.retcode_, list) and len(model.retcode_) == 3
    coefficients = np.asarray(model.coef_)
    restored = pickle.loads(pickle.dumps(model))
    del model
    gc.collect()
    expected = np.full((3, 3), -4 / 7, dtype=np.float32)
    np.fill_diagonal(expected, 4 / 7)
    np.testing.assert_allclose(coefficients, expected, rtol=0, atol=2e-4)
    assert list(restored.predict(points)) == ["a", "b", "c"] * 4


def test_early_parameter_refusal_keeps_original_publication_stage():
    model = LinearSVC(multi_class="bad", C=0, numeric_mode="identical")
    with pytest.raises(ValueError, match="multi_class must be 'ovr'"):
        model.fit(_points(), [-5, 9] * 4)
    assert "input_copied_" not in vars(model)
    assert "classes_" not in vars(model)
    assert "_w" not in vars(model)


@pytest.mark.parametrize("operation", ["fit", "predict", "decision_function"])
def test_linear_svc_native_factory_refuses_python_replacement(operation, monkeypatch):
    model = _fit()
    calls = []

    def forbidden():
        calls.append(True)
        raise AssertionError("Python LinearSVC provider invoked")

    monkeypatch.setattr(model._bind(), "linear_svc_native_api_v1", forbidden)
    with pytest.raises(TypeError, match="compiled extension entry"):
        if operation == "fit":
            model.fit(_points(), [-5, 9] * 4)
        else:
            getattr(model, operation)(_points())
    assert calls == []


def test_shared_linear_svr_decision_transport_keeps_public_weight_mutation():
    model = LinearSVR(numeric_mode="identical").fit(_points(), np.arange(8, dtype=np.float32))
    np.asarray(model._w)[:] = [2, -1, 0.5]
    query = np.array([[1, 3], [2, 1]], dtype=np.float32)
    np.testing.assert_allclose(model.predict(query), [-0.5, 3.5], rtol=0, atol=1e-6)
