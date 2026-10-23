# SPDX-License-Identifier: Apache-2.0
# The estimators binding's resident entries of `labeled_reference_predict`
# (core/labeled_reference_resident.mojo): DBSCAN and AgglomerativeClustering
# keep their fitted reference rows on the device under a handle minted at
# fit, and predict uploads only the queries. Same kernel, same launches,
# same bits as `labeled_reference_predict`.
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from core.hostptr import f32_ptr, i32_ptr
from core.labeled_reference_resident import (
    labeled_reference_prepare,
    labeled_reference_release,
    labeled_reference_predict_resident,
)


def labeled_reference_prepare_binding(addrs: PythonObject, params: PythonObject) raises -> PythonObject:
    """Upload a reference set once; returns its handle. `addrs`: refs
    (float32 n_refs x n_features), keys (int32 n_refs), ref_labels (int32
    n_refs). `params`: n_refs, n_features, metric (0 L2, 1 L1, 2 cosine).
    Release with `labeled_reference_release`."""
    if len(addrs) != 3:
        raise Error("labeled_reference_prepare: addrs must contain 3 addresses, got " + String(len(addrs)))
    if len(params) != 3:
        raise Error("labeled_reference_prepare: params must contain 3 values, got " + String(len(params)))
    var rp = f32_ptr(Int(py=addrs[0]))
    var kp = i32_ptr(Int(py=addrs[1]))
    var lp = i32_ptr(Int(py=addrs[2]))
    var n_refs = Int(py=params[0])
    var n_features = Int(py=params[1])
    var metric = Int(py=params[2])
    var handle: Int
    with GILReleased(Python()):
        handle = labeled_reference_prepare(rp, n_refs, kp, lp, n_features, metric)
    return PythonObject(handle)


def labeled_reference_release_binding(handle: PythonObject) raises -> PythonObject:
    """Drop a resident reference set."""
    labeled_reference_release(Int(py=handle))
    return PythonObject(None)


def labeled_reference_predict_resident_binding(
    handle: PythonObject, addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`labeled_reference_predict` over a resident reference set. `addrs`:
    queries (float32 n_queries x n_features), out_labels (int32 n_queries).
    `params`: n_queries, n_features, metric, eps (float, read only when
    has_thresh), has_thresh (0/1). Returns 0."""
    if len(addrs) != 2:
        raise Error("labeled_reference_predict_resident: addrs must contain 2 addresses, got " + String(len(addrs)))
    if len(params) != 5:
        raise Error("labeled_reference_predict_resident: params must contain 5 values, got " + String(len(params)))
    var h = Int(py=handle)
    var qp = f32_ptr(Int(py=addrs[0]))
    var olp = i32_ptr(Int(py=addrs[1]))
    var n_queries = Int(py=params[0])
    var n_features = Int(py=params[1])
    var metric = Int(py=params[2])
    var eps = Float64(py=params[3])
    var has_thresh = Int(py=params[4]) != 0
    with GILReleased(Python()):
        labeled_reference_predict_resident(h, qp, n_queries, n_features, metric, eps, has_thresh, olp)
    return PythonObject(0)
