# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU binding for the `_mojolearn_hdbscan` family: HDBSCAN (CPU training for
the workstream D estimators, 2026-09-15).

HOST ONLY. No DeviceContext, no kernel launch. The fit is
`hdbscan/host/hdbscan_host_oracle.mojo`, the device path of
`hdbscan/estimator.mojo::hdbscan_fit_host` restated on the host: the k-NN
core distances, the dense mutual reachability graph, the Boruvka MST with
its round count, the dendrogram, the condensed tree, the stabilities, the
selection and the labels. So `labels_`, `core_distances_` and the four
counts are meant to be the GPU columns' bytes.

THE EXPORTED NAMES ARE THE GPU BINDING'S NAMES, with the GPU binding's
address and params contract word for word (`bindings/_mojolearn_hdbscan.mojo`,
mirrored in `python/mojolearn/hdbscan.py`), so `HDBSCAN` runs unchanged on a
CPU-only install through `_backend._HOST_MODULES`
(`"_mojolearn_hdbscan": "_mojolearn_hdbscan_host"`): `hdbscan_fit`
(4 addresses, 10 params), `hdbscan_vendor` answering "cpu" and
`hdbscan_numeric_mode`. ABSENT, and so refused BY NAME through
`_HostBinding`: `hdbscan_rows_parallel_available`, the multi-GPU driver's
probe.

The sabotage arm (`hdbscan_host_sabotage`) is
`hdbscan/host/hdbscan_host_oracle.mojo::HDBH_HOST_SABOTAGE`
(`-D MOJOLEARN_HOST_SABOTAGE=1`): every core distance is read one neighbor
early.
"""
from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

from bindings.hostptr import f32_ptr, i32_ptr, read_f32, read_i32
from checks.kernel_matrix import (
    COLUMN_CPU,
    TARGET_COLUMN,
    column_name,
)
from checks.numerics import GLOBAL_NUMERIC_MODE
from bindings.hdbscan_host_predict import (
    hdbscan_all_points_membership_vectors_binding,
    hdbscan_approximate_predict_binding,
    hdbscan_membership_vector_binding,
)
from hdbscan.host.hdbscan_host_oracle import (
    HDBH_HOST_SABOTAGE,
    HDBH_PREDICT_SABOTAGE,
    hdbh_fit,
)
from hdbscan.impl.prediction_data import generate_prediction_data


def hdbscan_host_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(GLOBAL_NUMERIC_MODE)


def hdbscan_host_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def hdbscan_host_column_binding() raises -> PythonObject:
    """`column_name(TARGET_COLUMN)`, the comptime assert's witness: "cpu"."""
    comptime assert TARGET_COLUMN == COLUMN_CPU, (
        "hdbscan host: this binding compiles the CPU column only; pass"
        " -D MOJOLEARN_COLUMN_CPU (bindings/build_hdbscan_host.sh does)"
    )
    return PythonObject(column_name(TARGET_COLUMN))


def hdbscan_host_sabotage_binding() raises -> PythonObject:
    """Whether this binary reads every core distance one neighbor early on
    purpose (-D MOJOLEARN_HOST_SABOTAGE=1, the gate's negative control)."""
    return PythonObject(HDBH_HOST_SABOTAGE)


def hdbscan_vendor_binding() raises -> PythonObject:
    return PythonObject(String("cpu"))


def hdbscan_numeric_mode_binding() raises -> PythonObject:
    return PythonObject(GLOBAL_NUMERIC_MODE)


def hdbscan_fit_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`HDBSCAN(...).fit(X)` on the host. Returns the cluster count.
    `addrs`: 0 x, 1 labels_out (n int32), 2 core_dists_out (n float32),
    3 info_out (4 int32: n_clusters, n_outliers, Boruvka rounds,
    n_condensed_clusters). `params`: 0 n, 1 d, 2 min_samples,
    3 min_cluster_size, 4 max_cluster_size, 5 alpha, 6 allow_single_cluster,
    7 cluster_selection_method, 8 cluster_selection_epsilon, 9 metric."""
    if len(addrs) != 4 and len(addrs) != 9:
        raise Error(
            "hdbscan_fit: addrs must contain 4 addresses (x, labels_out,"
            " core_dists_out, info_out) or 9 (plus the condensed tree's"
            " parents, children, lambdas, sizes and inverse_label_map), got "
            + String(len(addrs))
        )
    if len(params) != 10:
        raise Error(
            "hdbscan_fit: params must contain 10 values (n, d, min_samples,"
            " min_cluster_size, max_cluster_size, alpha,"
            " allow_single_cluster, cluster_selection_method,"
            " cluster_selection_epsilon, metric), got "
            + String(len(params))
        )
    var lp = i32_ptr(Int(py=addrs[1]))
    var cp = f32_ptr(Int(py=addrs[2]))
    var ip = i32_ptr(Int(py=addrs[3]))
    var n = Int(py=params[0])
    var d = Int(py=params[1])
    var min_samples = Int(py=params[2])
    var min_cluster_size = Int(py=params[3])
    var max_cluster_size = Int(py=params[4])
    var alpha = Float32(Float64(py=params[5]))
    var allow_single = Int(py=params[6]) != 0
    var method = Int(py=params[7])
    var eps = Float32(Float64(py=params[8]))
    var metric = Int(py=params[9])
    var x = read_f32(Int(py=addrs[0]), max(0, n * d))
    var want_tree = len(addrs) == 9
    var tpp = lp
    var tcp = lp
    var tlp = cp
    var tsp = lp
    var invp = lp
    if want_tree:
        tpp = i32_ptr(Int(py=addrs[4]))
        tcp = i32_ptr(Int(py=addrs[5]))
        tlp = f32_ptr(Int(py=addrs[6]))
        tsp = i32_ptr(Int(py=addrs[7]))
        invp = i32_ptr(Int(py=addrs[8]))
    var n_clusters = 0
    with GILReleased(Python()):
        var out = hdbh_fit(
            x, n, d, min_samples, min_cluster_size, max_cluster_size, alpha,
            allow_single, method, eps, metric,
        )
        for i in range(n):
            lp.unsafe_store(i, out.labels[i])
            cp.unsafe_store(i, out.core_dists[i])
        ip.unsafe_store(0, Int32(out.n_clusters))
        ip.unsafe_store(1, Int32(out.n_outliers))
        ip.unsafe_store(2, Int32(out.n_boruvka_rounds))
        ip.unsafe_store(3, Int32(out.n_condensed_clusters))
        if want_tree:
            var ne = out.tree.n_edges
            if ne > 2 * n:
                raise Error(
                    "hdbscan_fit: the condensed tree has " + String(ne)
                    + " edges, more than the 2 * n_rows the caller sized;"
                    " refused by name"
                )
            ip.unsafe_store(4, Int32(ne))
            for e in range(ne):
                tpp.unsafe_store(e, out.tree.parents[e])
                tcp.unsafe_store(e, out.tree.children[e])
                tlp.unsafe_store(e, out.tree.lambdas[e])
                tsp.unsafe_store(e, out.tree.sizes[e])
            for c in range(out.n_clusters):
                invp.unsafe_store(c, out.inverse_label_map[c])
        n_clusters = out.n_clusters
        _ = out^
    _ = x^
    return PythonObject(n_clusters)


def hdbscan_generate_prediction_data_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """The GPU binding's `hdbscan_generate_prediction_data`, the same 11
    addresses and 4 params, calling the same host function
    (`hdbscan/impl/prediction_data.mojo`)."""
    if len(addrs) != 11:
        raise Error(
            "hdbscan_generate_prediction_data: addrs must contain 11 addresses,"
            " got " + String(len(addrs))
        )
    if len(params) != 4:
        raise Error(
            "hdbscan_generate_prediction_data: params must contain 4 values"
            " (n_leaves, n_edges, n_clusters, n_selected), got "
            + String(len(params))
        )
    var n_leaves = Int(py=params[0])
    var n_edges = Int(py=params[1])
    var n_clusters = Int(py=params[2])
    var n_selected = Int(py=params[3])
    if n_leaves < 2 or n_edges < 1 or n_clusters < 1 or n_selected < 0:
        raise Error(
            "hdbscan_generate_prediction_data: shape refused by name"
            " (n_leaves=" + String(n_leaves) + ", n_edges=" + String(n_edges)
            + ", n_clusters=" + String(n_clusters) + ", n_selected="
            + String(n_selected) + ")"
        )
    var labels = read_i32(Int(py=addrs[0]), n_leaves)
    var parents = read_i32(Int(py=addrs[1]), n_edges)
    var children = read_i32(Int(py=addrs[2]), n_edges)
    var lambdas = read_f32(Int(py=addrs[3]), n_edges)
    var sizes = read_i32(Int(py=addrs[4]), n_edges)
    var inv = read_i32(Int(py=addrs[5]), n_selected)
    var dp = f32_ptr(Int(py=addrs[6]))
    var sp = i32_ptr(Int(py=addrs[7]))
    var ep = i32_ptr(Int(py=addrs[8]))
    var op = i32_ptr(Int(py=addrs[9]))
    var iicp = i32_ptr(Int(py=addrs[10]))
    var n_ex = 0
    with GILReleased(Python()):
        var pd = generate_prediction_data(
            parents, children, lambdas, sizes, n_edges, n_leaves, n_clusters,
            labels, inv, n_selected,
        )
        for c in range(n_clusters):
            dp.unsafe_store(c, pd.deaths[c])
        for c in range(n_selected):
            sp.unsafe_store(c, pd.selected_clusters[c])
            op.unsafe_store(c, pd.exemplar_label_offsets[c])
        op.unsafe_store(n_selected, pd.exemplar_label_offsets[n_selected])
        for j in range(pd.n_exemplars):
            ep.unsafe_store(j, pd.exemplar_idx[j])
        for e in range(n_edges + 1):
            iicp.unsafe_store(e, pd.index_into_children[e])
        n_ex = pd.n_exemplars
        _ = pd^
    return PythonObject(n_ex)


def hdbscan_host_predict_sabotage_binding() raises -> PythonObject:
    """Whether this binary flips the lowest bit of every approximate_predict
    probability on purpose (`HDBH_PREDICT_SABOTAGE`)."""
    return PythonObject(HDBH_PREDICT_SABOTAGE)


@export
def PyInit__mojolearn_hdbscan_host() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_hdbscan_host")
        module.def_function[hdbscan_host_numeric_mode_binding]("hdbscan_host_numeric_mode")
        module.def_function[hdbscan_host_vendor_binding]("hdbscan_host_vendor")
        module.def_function[hdbscan_host_column_binding]("hdbscan_host_column")
        module.def_function[hdbscan_host_sabotage_binding]("hdbscan_host_sabotage")
        module.def_function[hdbscan_vendor_binding]("hdbscan_vendor")
        module.def_function[hdbscan_numeric_mode_binding]("hdbscan_numeric_mode")
        module.def_function[hdbscan_fit_binding]("hdbscan_fit")
        module.def_function[hdbscan_generate_prediction_data_binding]("hdbscan_generate_prediction_data")
        module.def_function[hdbscan_approximate_predict_binding]("hdbscan_approximate_predict")
        module.def_function[hdbscan_host_predict_sabotage_binding]("hdbscan_host_predict_sabotage")
        module.def_function[hdbscan_membership_vector_binding]("hdbscan_membership_vector")
        module.def_function[hdbscan_all_points_membership_vectors_binding]("hdbscan_all_points_membership_vectors")
        return module.finalize()
    except e:
        abort(String("failed to create _mojolearn_hdbscan_host: ", e))
