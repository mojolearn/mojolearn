# SPDX-License-Identifier: Apache-2.0
"""Python boundary ownership, live fitted state and legacy-entry removal.

Arithmetic/plan oracles are native Mojo fixtures. These cases exercise public
Python attributes and owners, with independent cases for pytest workers.
"""
import gc

import numpy as np
import pytest

from mojolearn import Array, NearestCentroid
from mojolearn._native_family_bridge import shared_context


def _model():
    x = np.array([[-2, -1], [-2, 1], [2, -1], [2, 1]], dtype="f4")
    model = NearestCentroid().fit(x, [0, 0, 1, 1])
    model.centroids_ = Array.from_list([[-1, 0], [1, 0]], "<f4")
    model.within_class_std_dev_ = Array.from_list([1, 1], "<f4")
    model.class_prior_ = Array.from_list([0.5, 0.5], "<f8")
    return model


@pytest.mark.parametrize("field", ["class_prior_", "centroids_", "within_class_std_dev_"])
@pytest.mark.parametrize("mutation", ["inplace", "rebind"])
def test_centroid_native_plan_observes_live_fitted_buffers(field, mutation):
    model = _model()
    query = np.array([[0, 0]], dtype="f4")
    if field == "class_prior_":
        replacement = np.array([0.1, 0.9], dtype="f8")
    elif field == "centroids_":
        replacement = np.array([[-10, 0], [1, 0]], dtype="f4")
    else:
        model.class_prior_ = Array.from_list([0.4, 0.6], "<f8")
        query[0, 0] = -0.25
        replacement = np.array([2, 1], dtype="f4")
    assert np.asarray(model.predict(query)).tolist() == [0]
    if mutation == "inplace":
        np.asarray(getattr(model, field))[...] = replacement
    else:
        setattr(model, field, Array.from_buffer(replacement))
    assert np.asarray(model.predict(query)).tolist() == [1]
    assert np.asarray(model.decision_function(query))[0] > 0
    assert np.asarray(model.predict_proba(query))[0, 1] > 0.5
    assert np.asarray(model.predict_log_proba(query))[0, 1] > np.asarray(model.predict_log_proba(query))[0, 0]


@pytest.mark.parametrize("method", ["predict", "decision_function", "predict_proba", "predict_log_proba"])
def test_centroid_empty_query_keeps_native_null_buffer_refusal(method):
    model = _model()
    with pytest.raises(ValueError, match="x_neighbors: null buffer address"):
        getattr(model, method)(np.empty((0, 2), dtype="f4"))


def test_centroid_query_outputs_detach_from_inputs_and_temporary_prior_state():
    model = _model()
    model.class_prior_ = Array.from_list([0.1, 0.9], "<f8")
    query = np.array([[-2, 0], [0, 0], [2, 0]], dtype="f4")
    output = model.predict_proba(query)
    expected = np.asarray(output).copy()
    query[:] = 100
    model.class_prior_ = Array.from_list([0.9, 0.1], "<f8")
    del model, query
    gc.collect()
    np.testing.assert_array_equal(np.asarray(output), expected)


def test_centroid_runtime_ignores_replaced_legacy_numerical_entries(monkeypatch):
    # Resolve the import-connected exact module before the public operation;
    # patched legacy exports must not be numerical dispatch or capability doors.
    _model()
    record = shared_context()["family_binding_connections"]["_mojolearn_x_neighbors"]["identical"]
    module = record[0]
    called = []

    def old_entry(*args):
        called.append(args)
        raise AssertionError("legacy numerical Python entry executed")

    names = ["xn_p2m_class_counts", "xn_group_mean", "xn_nc_median", "xn_nc_stats", "xn_nc_std",
             "xn_nc_med_std", "xn_nc_shrink_d", "xn_nc_shrink", "xn_nc_decision", "xn_sqdist",
             "xn_l1dist", "xn_knn_select", "xn_matmul", "xn_row_argmax", "xn_softmax", "xn_log_softmax",
             "x_neighbors_nc_labels", "x_neighbors_nc_proba"]
    for name in names:  # test controller: fixed legacy export inventory
        monkeypatch.setattr(module, name, old_entry)
    model = _model()
    query = np.array([[0, 0]], dtype="f4")
    assert np.asarray(model.predict(query)).tolist() == [0]
    model.class_prior_ = Array.from_list([0.1, 0.9], "<f8")
    assert np.asarray(model.predict(query)).tolist() == [1]
    assert np.asarray(model.decision_function(query))[0] > 0
    assert np.asarray(model.predict_proba(query))[0, 1] > 0.5
    assert np.asarray(model.predict_log_proba(query))[0, 1] > -1
    assert called == []
