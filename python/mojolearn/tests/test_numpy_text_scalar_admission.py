"""Native text scalar identities and weighted public calls share admission."""
import numpy as np
import pytest
import mojolearn as ml
from mojolearn._verify_native_runtime import audit_native_call


@pytest.mark.parametrize("partial", [False, True])
def test_weighted_gaussian_nb_admits_numpy_inputs(partial):
    x = np.arange(48, dtype="f4").reshape(16, 3) / 16
    y = np.arange(16, dtype="i4") % 2
    weights = np.linspace(0.25, 2, 16, dtype="f4")
    model = ml.GaussianNB()
    if partial:
        model.partial_fit(x[:8], y[:8], classes=[0, 1])
        model.partial_fit(x[8:], y[8:], sample_weight=weights[8:])
    else:
        model.fit(x, y, sample_weight=weights)
    assert np.isfinite(np.asarray(model.predict_proba(x))).all()


def test_weighted_bayesian_ridge_admits_numpy_inputs():
    x = np.arange(48, dtype="f4").reshape(16, 3) / 16
    y = x[:, 0] * 0.3 + np.arange(16, dtype="f4") % 3
    model = ml.BayesianRidge().fit(x, y, sample_weight=np.linspace(0.25, 2, 16, dtype="f4"))
    assert np.isfinite(np.asarray(model.predict(x))).all()


def test_numpy_string_labels_keep_native_identity_and_order():
    values = [np.str_("z"), np.str_("a"), np.str_("z"), np.str_("a")]
    x = np.array([[3, 4], [0, 1], [4, 5], [1, 2]], dtype="f4")
    model = ml.GaussianNB().fit(x, values)
    assert list(model.classes_) == ["a", "z"]
    assert list(model.predict(x)) == list(values)


@pytest.mark.parametrize("name", ["str_", "dtype"])
def test_replaced_numpy_text_metadata_refuses_without_callback(monkeypatch, name):
    values = [np.str_("z"), np.str_("a"), np.str_("z"), np.str_("a")]
    x = np.array([[3, 4], [0, 1], [4, 5], [1, 2]], dtype="f4")
    fit = ml.GaussianNB().fit
    calls = []

    def replacement(*args, **kwargs):
        calls.append(name)
        raise AssertionError("NumPy metadata replacement executed")

    with monkeypatch.context() as patch:
        patch.setattr(np, name, replacement)
        result = audit_native_call(fit, x, values)
    assert type(result.exception) is TypeError, result.report()
    assert "native" in str(result.exception)
    assert not calls
    for key in ("python_frames", "forbidden_native_calls", "wrapper_work", "unobserved_existing_threads"):
        assert not result.report()[key], result.report()
