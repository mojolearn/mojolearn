# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""HDBSCAN's `approximate_predict` entry of the hdbscan host bindings,
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
from hdbscan.host.hdbscan_host_oracle import hdbh_approximate_predict


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
