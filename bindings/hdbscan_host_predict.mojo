# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""HDBSCAN's prediction entries of the hdbscan host bindings,
`approximate_predict`, `membership_vector` and
`all_points_membership_vectors`,
written once (the neighbors and density inference lane, 2026-09-15).

`bindings/_mojolearn_hdbscan_host.mojo` (the reference binding: fit,
prediction data and prediction) and `bindings/_mojolearn_hdbscan_infer_host.mojo`
(the inference binding a wheel ships: prediction only) both register this
function. The address and params contract is the GPU binding's
(`bindings/_mojolearn_hdbscan.mojo`), word for word.
"""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased

from bindings.hostptr import f32_ptr, i32_ptr, read_f32, read_i32
from hdbscan.host.hdbscan_host_oracle import (
    hdbh_all_points_membership_vectors,
    hdbh_approximate_predict,
    hdbh_membership_vector,
)


def hdbscan_approximate_predict_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """The GPU binding's `hdbscan_approximate_predict`, the same 10
    addresses and 7 params, through `hdbh_approximate_predict`."""
    if len(addrs) != 10:
        raise Error(
            "hdbscan_approximate_predict: addrs must contain 10 addresses, got "
            + String(len(addrs))
        )
    if len(params) != 7:
        raise Error(
            "hdbscan_approximate_predict: params must contain 7 values (m, d,"
            " n_edges, n_clusters, n_selected, nq, min_samples), got "
            + String(len(params))
        )
    var m = Int(py=params[0])
    var d = Int(py=params[1])
    var n_edges = Int(py=params[2])
    var n_clusters = Int(py=params[3])
    var n_selected = Int(py=params[4])
    var nq = Int(py=params[5])
    var min_samples = Int(py=params[6])
    if m < 2 or d < 1 or n_edges < 1 or n_clusters < 1 or n_selected < 0 or nq < 1:
        raise Error(
            "hdbscan_approximate_predict: shape refused by name (m=" + String(m)
            + ", d=" + String(d) + ", n_edges=" + String(n_edges)
            + ", n_clusters=" + String(n_clusters) + ", n_selected="
            + String(n_selected) + ", nq=" + String(nq) + ")"
        )
    var x = read_f32(Int(py=addrs[0]), m * d)
    var core = read_f32(Int(py=addrs[1]), m)
    var labels = read_i32(Int(py=addrs[2]), m)
    var lambdas = read_f32(Int(py=addrs[3]), n_edges)
    var deaths = read_f32(Int(py=addrs[4]), n_clusters)
    var selected = read_i32(Int(py=addrs[5]), n_selected)
    var iic = read_i32(Int(py=addrs[6]), n_edges + 1)
    var q = read_f32(Int(py=addrs[7]), nq * d)
    var lo = i32_ptr(Int(py=addrs[8]))
    var po = f32_ptr(Int(py=addrs[9]))
    with GILReleased(Python()):
        var out = hdbh_approximate_predict(
            x, m, d, core, labels, lambdas, m, deaths, selected, iic, q, nq,
            min_samples,
        )
        for i in range(nq):
            lo.unsafe_store(i, out.labels[i])
            po.unsafe_store(i, out.probabilities[i])
        _ = out^
    return PythonObject(nq)


def _soft_shape_refusal(where: String, vals: List[Int]) raises:
    """The GPU binding's refusal: every count positive, m at least 2."""
    var ok = vals[0] >= 2
    for i in range(1, len(vals)):
        if vals[i] < 1:
            ok = False
    if not ok:
        var s = String(where) + ": shape refused by name ("
        for i in range(len(vals)):
            if i > 0:
                s += ", "
            s += String(vals[i])
        raise Error(s + ")")


def hdbscan_membership_vector_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """The GPU binding's `hdbscan_membership_vector`, the same 12 addresses
    and 8 params, through `hdbh_membership_vector`."""
    if len(addrs) != 12:
        raise Error(
            "hdbscan_membership_vector: addrs must contain 12 addresses, got "
            + String(len(addrs))
        )
    if len(params) != 8:
        raise Error(
            "hdbscan_membership_vector: params must contain 8 values (m, d,"
            " n_edges, n_clusters, n_selected, n_exemplars, nq, min_samples),"
            " got " + String(len(params))
        )
    var m = Int(py=params[0])
    var d = Int(py=params[1])
    var n_edges = Int(py=params[2])
    var n_clusters = Int(py=params[3])
    var n_selected = Int(py=params[4])
    var n_ex = Int(py=params[5])
    var nq = Int(py=params[6])
    var min_samples = Int(py=params[7])
    _soft_shape_refusal(
        "hdbscan_membership_vector",
        [m, d, n_edges, n_clusters, n_selected, n_ex, nq],
    )
    var x = read_f32(Int(py=addrs[0]), m * d)
    var core = read_f32(Int(py=addrs[1]), m)
    var labels = read_i32(Int(py=addrs[2]), m)
    var parents = read_i32(Int(py=addrs[3]), n_edges)
    var lambdas = read_f32(Int(py=addrs[4]), n_edges)
    var deaths = read_f32(Int(py=addrs[5]), n_clusters)
    var selected = read_i32(Int(py=addrs[6]), n_selected)
    var iic = read_i32(Int(py=addrs[7]), n_edges + 1)
    var ex_idx = read_i32(Int(py=addrs[8]), n_ex)
    var offsets = read_i32(Int(py=addrs[9]), n_selected + 1)
    var q = read_f32(Int(py=addrs[10]), nq * d)
    var out = f32_ptr(Int(py=addrs[11]))
    with GILReleased(Python()):
        var mv = hdbh_membership_vector(
            x, m, d, core, labels, parents, lambdas, n_edges, n_clusters,
            deaths, selected, iic, ex_idx, offsets, q, nq, min_samples,
        )
        for i in range(nq * n_selected):
            out.unsafe_store(i, mv[i])
        _ = mv^
    return PythonObject(nq)


def hdbscan_all_points_membership_vectors_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """The GPU binding's `hdbscan_all_points_membership_vectors`, the same 9
    addresses and 8 params, through `hdbh_all_points_membership_vectors`."""
    if len(addrs) != 9:
        raise Error(
            "hdbscan_all_points_membership_vectors: addrs must contain 9"
            " addresses, got " + String(len(addrs))
        )
    if len(params) != 8:
        raise Error(
            "hdbscan_all_points_membership_vectors: params must contain 8"
            " values (m, d, n_edges, n_clusters, n_selected, n_exemplars,"
            " row0, count), got " + String(len(params))
        )
    var m = Int(py=params[0])
    var d = Int(py=params[1])
    var n_edges = Int(py=params[2])
    var n_clusters = Int(py=params[3])
    var n_selected = Int(py=params[4])
    var n_ex = Int(py=params[5])
    var row0 = Int(py=params[6])
    var count = Int(py=params[7])
    _soft_shape_refusal(
        "hdbscan_all_points_membership_vectors",
        [m, d, n_edges, n_clusters, n_selected, n_ex, count],
    )
    if row0 < 0 or row0 + count > m:
        raise Error(
            "hdbscan_all_points_membership_vectors: rows " + String(row0)
            + " + " + String(count) + " exceed m=" + String(m)
            + "; refused by name"
        )
    var x = read_f32(Int(py=addrs[0]), m * d)
    var parents = read_i32(Int(py=addrs[1]), n_edges)
    var lambdas = read_f32(Int(py=addrs[2]), n_edges)
    var deaths = read_f32(Int(py=addrs[3]), n_clusters)
    var selected = read_i32(Int(py=addrs[4]), n_selected)
    var iic = read_i32(Int(py=addrs[5]), n_edges + 1)
    var ex_idx = read_i32(Int(py=addrs[6]), n_ex)
    var offsets = read_i32(Int(py=addrs[7]), n_selected + 1)
    var out = f32_ptr(Int(py=addrs[8]))
    with GILReleased(Python()):
        var mv = hdbh_all_points_membership_vectors(
            x, m, d, parents, lambdas, n_edges, n_clusters, deaths, selected,
            iic, ex_idx, offsets, row0, count,
        )
        for i in range(count * n_selected):
            out.unsafe_store(i, mv[i])
        _ = mv^
    return PythonObject(count)
