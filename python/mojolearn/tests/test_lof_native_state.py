# SPDX-License-Identifier: Apache-2.0
"""Public LOF mutation, owner and refusal seams; numerical oracles are native.

Each parametrized case is independent for the repository's parallel pytest
controller. These fixtures retain Python only to exercise its public boundary.
"""
import numpy as np
import pytest

from mojolearn import LocalOutlierFactor
from mojolearn._native_family_bridge import shared_context


def _data():
    return np.array([[0, 0], [1, 0], [2, 0], [9, 0]], dtype="f4")


def _query():
    return np.array([[0.5, 0], [8, 0]], dtype="f4")


def _model():
    return LocalOutlierFactor(n_neighbors=2, novelty=True).fit(_data())


@pytest.mark.parametrize("unfused,old_items", [(False, False), (False, True), (True, False)])
def test_lof_runtime_ignores_replaced_legacy_numerical_entries(monkeypatch, unfused, old_items):
    _model()
    context = shared_context()
    module = context["family_binding_connections"]["_mojolearn_x_neighbors"]["identical"][0]
    monkeypatch.setitem(context["family_xn_loc_flags"], "unfused_knn", unfused)
    monkeypatch.setitem(context["family_xn_loc_flags"], "old_items", old_items)
    called = []

    def old_entry(*args):
        called.append(args)
        raise AssertionError("legacy LOF Python numerical entry executed")

    for name in ("xn_sqdist", "xn_knn_select", "xn_knn_sq", "xn_knn_sq_tiled", "xn_unary",
                 "xn_lof_lrd", "xn_lof_score", "xn_p2m_row_sort", "xn_p2m_sign_label"):
        monkeypatch.setattr(module, name, old_entry)
    fit_predict = LocalOutlierFactor(n_neighbors=2, contamination=0.25).fit_predict(_data())
    model = LocalOutlierFactor(n_neighbors=2, contamination=0.25, novelty=True).fit(_data())
    model.score_samples(_query())
    model.decision_function(_query())
    predicted = model.predict(_query())
    assert np.asarray(fit_predict).dtype == np.dtype("int64")
    assert np.asarray(predicted).dtype == np.dtype("int64")
    assert called == []


@pytest.mark.parametrize("in_place", [False, True])
def test_lof_query_rereads_public_density(in_place):
    model = _model()
    before = np.asarray(model.score_samples(_query()))
    replacement = np.asarray(model._lrd).copy()
    model._lrd = replacement
    if in_place:
        replacement *= np.float32(2)
    else:
        model._lrd = replacement * np.float32(2)
    np.testing.assert_array_equal(model.score_samples(_query()), before * 2)


@pytest.mark.parametrize("offset", [-0.0, np.float32(-1.25), np.array(-1.25, dtype="f4"), np.uint8(2)])
def test_lof_decision_rereads_native_scalar_offset(offset):
    model = _model()
    scores = np.asarray(model.score_samples(_query()), dtype="f4")
    model.offset_ = offset
    # This is a scalar-boundary oracle, including NumPy unsigned negation.
    with np.errstate(over="ignore"):
        shift = np.float32(float(-offset))
        expected = np.add(scores, shift, dtype=np.float32).astype("f8")
        actual = model.decision_function(_query())
    np.testing.assert_array_equal(actual, expected)


def test_lof_negative_rebound_neighbor_count_keeps_array_shape_refusal():
    model = _model()
    model.n_neighbors_ = -1
    with pytest.raises(ValueError, match="negative dimensions are not allowed"):
        model.score_samples(_query())


@pytest.mark.parametrize("method", ["decision_function", "predict"])
def test_lof_rebound_scalar_hooks_are_refused_before_execution(method):
    model = _model()
    called = []

    class Hook:
        def __float__(self):
            called.append("float")
            raise AssertionError("Python scalar conversion hook executed")

        def __neg__(self):
            called.append("neg")
            raise AssertionError("Python scalar negation hook executed")

        def __bool__(self):
            called.append("bool")
            raise AssertionError("Python scalar truth hook executed")

    model.offset_ = Hook()
    with pytest.raises(TypeError, match="native|Python|hook"):
        getattr(model, method)(_query())
    assert called == []


def test_lof_failed_finite_refit_keeps_previous_published_state():
    model = _model()
    names = ("_fit_X", "_fit_dist", "_lrd", "negative_outlier_factor_", "offset_",
             "n_neighbors_", "n_samples_fit_", "n_features_in_")
    before = {name: getattr(model, name) for name in names}
    bad = _data()
    bad[1, 0] = np.nan
    with pytest.raises(ValueError, match="NaN|infinity|finite"):
        model.fit(bad)
    assert all(getattr(model, name) is value for name, value in before.items())


@pytest.mark.parametrize("method", ["score_samples", "decision_function", "predict"])
def test_lof_novelty_refusal_precedes_query_finite_check(method):
    model = _model()
    model.novelty = False
    bad = _query()
    bad[0, 0] = np.nan
    with pytest.raises(AttributeError, match="novelty=False"):
        getattr(model, method)(bad)


@pytest.mark.parametrize("field", ["metric", "p"])
def test_lof_invalid_scalar_array_diagnostic_does_not_enter_numpy_formatter(field):
    kwargs = {"metric": "unsupported", field: np.array(7.0)}
    called = []

    def formatter(value):
        called.append(value)
        raise AssertionError("Python NumPy formatter executed")

    with np.printoptions(formatter={"float_kind": formatter}):
        with pytest.raises(NotImplementedError, match="<native scalar array>"):
            LocalOutlierFactor(**kwargs).fit(_data())
    assert called == []
