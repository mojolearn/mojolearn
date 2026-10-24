# SPDX-License-Identifier: Apache-2.0
"""host_predict / host_predict_proba: one native call per public function.

The saved model is rebuilt by the native archive reader and predicts through
the compiled owner its public method reaches; no public Python method,
availability lambda or `_bind` runs. The reference is the old two-step
route, `host_model(path).<method>(X)`, built separately and compared by
value, dtype and outcome (the same exception type and message on refusal).
"""
import numpy as np
import pytest

import mojolearn
from mojolearn import Array
from mojolearn import _classical_host as host
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.tests.test_host_archive_native import CASES, _archive

# Forecast models take the start index positionally, not a data matrix.
_FORECAST = ("ARIMA", "ExponentialSmoothing")


def _query(name, path):
    if name in _FORECAST:
        return 0
    model = mojolearn.host_model(path)
    width = vars(model).get("n_features_in_", 2)
    return np.arange(2 * width, dtype="<f4").reshape(2, width) / 4


def _outcome(call, *args):
    try:
        return "ok", call(*args)
    except Exception as error:  # the reference's own refusal is the contract
        return "error", error


def _values(result):
    if isinstance(result, list):
        return [_values(item) for item in result]
    return np.asarray(result)


def _assert_same(got, expected):
    assert type(got) is type(expected) or isinstance(got, (Array, np.ndarray))
    if isinstance(expected, list):
        assert isinstance(got, list) and len(got) == len(expected)
        for a, b in zip(got, expected):
            _assert_same(a, b)
        return
    a, b = _values(got), _values(expected)
    assert a.shape == b.shape
    assert a.dtype == b.dtype or {a.dtype.str, b.dtype.str} == {"<f4", "<f8"}
    assert np.array_equal(a.astype(b.dtype), b, equal_nan=a.dtype.kind == "f")


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("method", ["predict", "predict_proba"])
def test_host_archive_prediction_is_one_native_call_with_old_outcome(tmp_path, name, method):
    path = _archive(tmp_path, name)
    X = _query(name, path)
    reference = _outcome(lambda: getattr(mojolearn.host_model(path), method)(X))
    public = mojolearn.host_predict if method == "predict" else mojolearn.host_predict_proba
    audit = audit_native_call(public, path, X)
    assert not audit.frames and not audit.forbidden_native and not audit.wrapper_work
    if reference[0] == "error":
        assert type(audit.exception) is type(reference[1])
        if isinstance(reference[1], AttributeError):
            # The absent-method wording differed by attribute owner (object's
            # or the native descriptor's); the method name and type are kept.
            assert method in str(audit.exception)
        else:
            assert str(audit.exception) == str(reference[1])
        return
    _assert_same(audit.assert_native(), reference[1])


@pytest.mark.parametrize("name", ["GaussianMixture", "DBSCAN", "AgglomerativeClustering",
                                  "KNeighborsClassifier", "KNeighborsRegressor"])
def test_saved_fast_mode_is_refused_as_the_host_bind_refused_it(tmp_path, name):
    path = _archive(tmp_path, name, mode="fast")
    try:
        model = mojolearn.host_model(path)
    except Exception as error:
        with pytest.raises(type(error)):
            mojolearn.host_predict(path, np.zeros((1, 2), dtype="<f4"))
        return
    if vars(model).get("numeric_mode") in (None, "identical"):
        pytest.skip("this format restores the identical mode")
    with pytest.raises(ValueError, match="runs IDENTICAL only on the host"):
        model.predict(np.zeros((1, 2), dtype="<f4"))
    audit = audit_native_call(mojolearn.host_predict, path, np.zeros((1, 2), dtype="<f4"))
    assert not audit.frames
    assert type(audit.exception) is ValueError and "runs IDENTICAL only on the host" in str(audit.exception)


def test_replaced_public_method_is_refused_and_never_called(tmp_path, monkeypatch):
    path = _archive(tmp_path, "KMeans")
    calls = []
    monkeypatch.setattr(host.HostKMeans, "predict", lambda self, X: calls.append(X))
    audit = audit_native_call(mojolearn.host_predict, path, np.zeros((1, 2), dtype="<f4"))
    assert type(audit.exception) is TypeError and "replaced public method" in str(audit.exception)
    assert not calls and not audit.frames


def test_classes_without_prediction_keep_attribute_error(tmp_path):
    path = _archive(tmp_path, "PCA")
    audit = audit_native_call(mojolearn.host_predict, path, np.zeros((1, 3), dtype="<f4"))
    assert type(audit.exception) is AttributeError and "predict" in str(audit.exception)
    assert not audit.frames


def test_dbscan_without_prediction_data_is_unavailable_before_reading_x(tmp_path):
    model = mojolearn.host_model(_archive(tmp_path, "DBSCAN"))
    if vars(model).get("prediction_data"):
        pytest.skip("the archive case is saved with prediction data")
    poisoned = type("Poison", (), {"__array__": lambda self, *a, **k: pytest.fail("X was read")})()
    audit = audit_native_call(mojolearn.host_predict, _archive(tmp_path, "DBSCAN"), poisoned)
    assert type(audit.exception) is AttributeError
    assert str(audit.exception) == "This 'HostDBSCAN' has no attribute 'predict'"


def test_binding_export_replacement_is_refused(tmp_path, monkeypatch):
    path = _archive(tmp_path, "GaussianMixture")
    module = host._backend.load_host_module(host._HOST_BASENAMES["_mojolearn_mixture"])
    name = next(key for key, value in vars(module).items() if callable(value) and not key.startswith("_"))
    monkeypatch.setattr(module, name, lambda *args: pytest.fail("replaced export ran"))
    audit = audit_native_call(mojolearn.host_predict, path, np.zeros((1, 2), dtype="<f4"))
    assert type(audit.exception) is TypeError and "export changed" in str(audit.exception)
