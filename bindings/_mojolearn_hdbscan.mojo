# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPython boundary for the HDBSCAN lane (workstream D, 2026-09-14).

A separate extension module, for `bindings/_mojolearn_gp.mojo`'s reason.
`hdbscan/estimator.mojo::hdbscan_fit_host` is reached and nothing is
re-decided: its header lists every refusal and where each is raised
(metric, build algorithm, selection method and epsilon, min_samples,
min_cluster_size, alpha, the row bounds, a non-finite cell, and
`probabilities_`, DEVIATION 1610). This file refuses a null address and a
list of the wrong length and nothing else.

THE ABI IS THE GP'S: two length-checked lists, orders written out below
and mirrored in `python/mojolearn/hdbscan.py`. The fit writes the labels,
the core distances and four integers (`info_ptr`) into caller-sized
buffers and returns the cluster count. There is no model to carry: HDBSCAN
is transductive, as `DBSCAN` is, and the harness's model column is
`n/a:no-save`.

THE GIL is released around the device call, and nothing inside the
`GILReleased` block touches a `PythonObject`.
"""

from std.os import abort
from bindings.hostptr import f32_ptr, i32_ptr, read_f32, read_i32
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder
from max.gpu.host import DeviceContext

from checks.numerics import GLOBAL_NUMERIC_MODE
from checks.vendor import COMPILED_VENDOR
from hdbscan.estimator import (
    hdbscan_approximate_predict_host,
    hdbscan_fit_host_output,
)
from hdbscan.impl.prediction_data import generate_prediction_data


def _f32_ptr(addr: Int) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    return f32_ptr(addr)


def _i32_ptr(addr: Int) raises -> MutPointer[Int32, MutUntrackedOrigin]:
    return i32_ptr(addr)


def hdbscan_numeric_mode_binding() raises -> PythonObject:
    """THE BUILD'S TIER as the `NUMERIC_*` code: 0 FAST, 1 IDENTICAL, 2
    DETERMINISTIC."""
    return PythonObject(GLOBAL_NUMERIC_MODE)


def hdbscan_vendor_binding() raises -> PythonObject:
    """'metal', 'cuda', 'hip' or 'none', from `checks/vendor.mojo`."""
    return PythonObject(String(COMPILED_VENDOR))


def hdbscan_rows_parallel_available() raises -> PythonObject:
    """1: the core-distance k-NN reads MOJOLEARN_NEIGHBORS_DEVICE_COUNT
    (neighbors/impl/multi_gpu.mojo) and the dense pairwise distance matrix
    reads MOJOLEARN_HIERARCHY_DEVICE_COUNT (hierarchy/impl/cluster/detail/
    multi_gpu.mojo); parallel_classical.fit_hdbscan sets both in its worker."""
    return PythonObject(1)


def _hdbscan_fit_run(
    xp: MutPointer[Float32, MutUntrackedOrigin],
    lp: MutPointer[Int32, MutUntrackedOrigin],
    cp: MutPointer[Float32, MutUntrackedOrigin],
    ip: MutPointer[Int32, MutUntrackedOrigin],
    want_tree: Bool,
    tpp: MutPointer[Int32, MutUntrackedOrigin],
    tcp: MutPointer[Int32, MutUntrackedOrigin],
    tlp: MutPointer[Float32, MutUntrackedOrigin],
    tsp: MutPointer[Int32, MutUntrackedOrigin],
    invp: MutPointer[Int32, MutUntrackedOrigin],
    n: Int,
    d: Int,
    min_samples: Int,
    min_cluster_size: Int,
    max_cluster_size: Int,
    alpha: Float32,
    allow_single_cluster: Bool,
    cluster_selection_method: Int,
    cluster_selection_epsilon: Float32,
    metric: Int,
) raises -> Int:
    var ctx = DeviceContext()
    var out = hdbscan_fit_host_output(
        ctx,
        xp,
        n,
        d,
        min_samples,
        min_cluster_size,
        max_cluster_size,
        alpha,
        allow_single_cluster,
        cluster_selection_method,
        cluster_selection_epsilon,
        metric,
    )
    ctx.synchronize()
    for i in range(n):
        lp.unsafe_store(i, out.labels[i])
        cp.unsafe_store(i, out.core_dists[i])
    ip.unsafe_store(0, Int32(out.n_clusters))
    ip.unsafe_store(1, Int32(out.n_outliers))
    ip.unsafe_store(2, Int32(out.n_boruvka_rounds))
    ip.unsafe_store(3, Int32(out.condensed.n_clusters))
    if want_tree:
        var ne = out.condensed.n_edges
        if ne > 2 * n:
            raise Error(
                "hdbscan_fit: the condensed tree has " + String(ne)
                + " edges, more than the 2 * n_rows the caller sized; refused"
                " by name"
            )
        ip.unsafe_store(4, Int32(ne))
        for e in range(ne):
            tpp.unsafe_store(e, out.condensed.parents[e])
            tcp.unsafe_store(e, out.condensed.children[e])
            tlp.unsafe_store(e, out.condensed.lambdas[e])
            tsp.unsafe_store(e, out.condensed.sizes[e])
        for c in range(out.n_clusters):
            invp.unsafe_store(c, out.inverse_label_map[c])
    var n_clusters = out.n_clusters
    _ = out^
    # DEVIATION 1946: the context dies LAST, after every value built on it.
    _ = ctx^
    return n_clusters


def hdbscan_fit_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`HDBSCAN(...).fit(X)` (`hdbscan_fit_host`, cuML `runner.h:152-234`).
    Returns the cluster count.

    `addrs`, in this exact order:

        0  x                n * d float32, row-major, read
        1  labels_out       n int32, WRITTEN (0 .. n_clusters-1, -1 noise)
        2  core_dists_out   n float32, WRITTEN (distance to the
                             min_samples-th neighbor excluding self)
        3  info_out         4 int32, WRITTEN: n_clusters, n_outliers,
                             Boruvka round count, n_condensed_clusters

    `params`, in this exact order:

        0  n
        1  d
        2  min_samples
        3  min_cluster_size
        4  max_cluster_size            (0 means no cap)
        5  alpha                       (float)
        6  allow_single_cluster        (0/1)
        7  cluster_selection_method    (0 eom, 1 leaf)
        8  cluster_selection_epsilon   (float; only 0.0 is implemented,
                                        refused by name otherwise)
        9  metric                      (1 L2SqrtExpanded, the only one)

    `prediction_data=True` passes NINE addresses: the four above (info_out
    then holds FIVE int32, [4] = the condensed tree's edge count) and

        4  tree_parents_out   2 * n int32, WRITTEN (first n_edges)
        5  tree_children_out  2 * n int32, WRITTEN
        6  tree_lambdas_out   2 * n float32, WRITTEN
        7  tree_sizes_out     2 * n int32, WRITTEN
        8  inverse_label_map_out  n int32, WRITTEN (first n_clusters)
    """
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
    var xp = _f32_ptr(Int(py=addrs[0]))
    var lp = _i32_ptr(Int(py=addrs[1]))
    var cp = _f32_ptr(Int(py=addrs[2]))
    var ip = _i32_ptr(Int(py=addrs[3]))
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
    var want_tree = len(addrs) == 9
    var tpp = lp
    var tcp = lp
    var tlp = cp
    var tsp = lp
    var invp = lp
    if want_tree:
        tpp = _i32_ptr(Int(py=addrs[4]))
        tcp = _i32_ptr(Int(py=addrs[5]))
        tlp = _f32_ptr(Int(py=addrs[6]))
        tsp = _i32_ptr(Int(py=addrs[7]))
        invp = _i32_ptr(Int(py=addrs[8]))
    var n_clusters = 0
    with GILReleased(Python()):
        n_clusters = _hdbscan_fit_run(
            xp,
            lp,
            cp,
            ip,
            want_tree,
            tpp,
            tcp,
            tlp,
            tsp,
            invp,
            n,
            d,
            min_samples,
            min_cluster_size,
            max_cluster_size,
            alpha,
            allow_single,
            method,
            eps,
            metric,
        )
    return PythonObject(n_clusters)


def hdbscan_generate_prediction_data_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`generate_prediction_data` (cuML `prediction_data.cu:92-239`,
    `hdbscan/impl/prediction_data.mojo`, host code on either binding).
    Returns the exemplar count.

    `addrs`: 0 labels (n_leaves int32), 1 parents, 2 children, 3 lambdas
    (float32), 4 sizes (n_edges each), 5 inverse_label_map (n_selected),
    6 deaths_out (n_clusters float32), 7 selected_clusters_out
    (n_selected), 8 exemplar_idx_out (n_leaves), 9
    exemplar_label_offsets_out (n_selected + 1), 10
    index_into_children_out (n_edges + 1), every out int32 unless named.
    `params`: 0 n_leaves, 1 n_edges, 2 n_clusters (condensed), 3 n_selected.
    """
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
    var dp = _f32_ptr(Int(py=addrs[6]))
    var sp = _i32_ptr(Int(py=addrs[7]))
    var ep = _i32_ptr(Int(py=addrs[8]))
    var op = _i32_ptr(Int(py=addrs[9]))
    var iicp = _i32_ptr(Int(py=addrs[10]))
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


def _hdbscan_predict_run(
    x: List[Float32],
    m: Int,
    d: Int,
    core: List[Float32],
    labels: List[Int32],
    lambdas: List[Float32],
    n_edges: Int,
    n_clusters: Int,
    deaths: List[Float32],
    selected: List[Int32],
    iic: List[Int32],
    q: List[Float32],
    nq: Int,
    min_samples: Int,
    lo: MutPointer[Int32, MutUntrackedOrigin],
    po: MutPointer[Float32, MutUntrackedOrigin],
) raises:
    var ctx = DeviceContext()
    hdbscan_approximate_predict_host(
        ctx, x, m, d, core, labels, lambdas, n_edges, n_clusters, deaths,
        selected, iic, q, nq, min_samples, lo, po,
    )
    ctx.synchronize()
    # DEVIATION 1946: the context dies LAST, after every value built on it.
    _ = ctx^


def hdbscan_approximate_predict_binding(
    addrs: PythonObject, params: PythonObject
) raises -> PythonObject:
    """`approximate_predict(clusterer, points_to_predict)` (cuML
    `hdbscan.pyx:1264`, `predict.cuh:220-262`). Returns the query count.

    `addrs`: 0 x_train (m * d float32), 1 core_distances (m float32), 2
    labels (m int32), 3 condensed lambdas (n_edges float32), 4 deaths
    (n_clusters float32), 5 selected_clusters (n_selected int32), 6
    index_into_children (n_edges + 1 int32), 7 points_to_predict (nq * d
    float32), 8 labels_out (nq int32, WRITTEN), 9 probabilities_out (nq
    float32, WRITTEN).
    `params`: 0 m, 1 d, 2 n_edges, 3 n_clusters (condensed), 4 n_selected,
    5 nq, 6 min_samples (the estimator's, before the fit's + 1).
    """
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
    var lo = _i32_ptr(Int(py=addrs[8]))
    var po = _f32_ptr(Int(py=addrs[9]))
    with GILReleased(Python()):
        _hdbscan_predict_run(
            x, m, d, core, labels, lambdas, n_edges, n_clusters, deaths,
            selected, iic, q, nq, min_samples, lo, po,
        )
    return PythonObject(nq)


@export
def PyInit__mojolearn_hdbscan() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_hdbscan")
        m.def_function[hdbscan_rows_parallel_available]("hdbscan_rows_parallel_available")
        m.def_function[hdbscan_vendor_binding]("hdbscan_vendor")
        m.def_function[hdbscan_numeric_mode_binding]("hdbscan_numeric_mode")
        m.def_function[hdbscan_fit_binding]("hdbscan_fit")
        m.def_function[hdbscan_generate_prediction_data_binding]("hdbscan_generate_prediction_data")
        m.def_function[hdbscan_approximate_predict_binding]("hdbscan_approximate_predict")
        return m.finalize()
    except e:
        abort(String("failed to create _mojolearn_hdbscan: ", e))
