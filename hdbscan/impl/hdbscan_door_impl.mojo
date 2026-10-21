# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Typed bodies of the HDBSCAN GPU binding's doors (`bindings/_mojolearn_hdbscan.mojo`):
the fit run and the prediction-data run with their result copies, moved
here with no Python import (lane thin-core, 2026-10-21: a pure move, no bit
change)."""
from std.memory import memcpy
from std.sys.compile import is_defined
from core.neural_context import process_ctx
from checks.numerics import GLOBAL_NUMERIC_MODE as _DEVCTX_MODE, NUMERIC_IDENTICAL as _DEVCTX_IDENTICAL
from hdbscan.estimator import hdbscan_fit_host_output
from hdbscan.impl.prediction_data import generate_prediction_data_device

#: The HDBSCAN binding's ONE process-lifetime DeviceContext (core/neural_context.mojo,
#: lane/devctx-lifetime): a context per call exhausts Metal command queues.
comptime HDB_DEVCTX_SLOT = "MojoHdbscanContextIdentical" if _DEVCTX_MODE == _DEVCTX_IDENTICAL else "MojoHdbscanContextFast"
comptime _DEVCTX_SLOT = HDB_DEVCTX_SLOT

#: fg-tsne-dbscan H5 (every mode and column, DEFAULT ON, lane fg-tsne-dbscan
#: 2026-10-09): the fit's host lists (labels and core distances, n each; the
#: condensed tree's four arrays; the inverse map; the probabilities) go into
#: the caller's arrays by one memcpy each instead of an element loop per
#: array (5 n + 4 n_edges scalar stores). The same bytes.
#: `-D MOJOLEARN_IDN_HDB_OUT_MEMCPY_OFF` restores the loops.
comptime IDN_HDB_OUT_MEMCPY = not is_defined["MOJOLEARN_IDN_HDB_OUT_MEMCPY_OFF"]()


def hdbscan_fit_door_run(
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
    want_probs: Bool,
    pp: MutPointer[Float32, MutUntrackedOrigin],
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
    var ctx = process_ctx[_DEVCTX_SLOT]()
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
    comptime if IDN_HDB_OUT_MEMCPY:
        memcpy(dest=lp, src=out.labels.unsafe_ptr(), count=n)
        memcpy(dest=cp, src=out.core_dists.unsafe_ptr(), count=n)
    else:
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
        comptime if IDN_HDB_OUT_MEMCPY:
            memcpy(dest=tpp, src=out.condensed.parents.unsafe_ptr(), count=ne)
            memcpy(dest=tcp, src=out.condensed.children.unsafe_ptr(), count=ne)
            memcpy(dest=tlp, src=out.condensed.lambdas.unsafe_ptr(), count=ne)
            memcpy(dest=tsp, src=out.condensed.sizes.unsafe_ptr(), count=ne)
            memcpy(dest=invp, src=out.inverse_label_map.unsafe_ptr(), count=out.n_clusters)
        else:
            for e in range(ne):
                tpp.unsafe_store(e, out.condensed.parents[e])
                tcp.unsafe_store(e, out.condensed.children[e])
                tlp.unsafe_store(e, out.condensed.lambdas[e])
                tsp.unsafe_store(e, out.condensed.sizes[e])
            for c in range(out.n_clusters):
                invp.unsafe_store(c, out.inverse_label_map[c])
    if want_probs:
        comptime if IDN_HDB_OUT_MEMCPY:
            memcpy(dest=pp, src=out.probabilities.unsafe_ptr(), count=n)
        else:
            for i in range(n):
                pp.unsafe_store(i, out.probabilities[i])
    var n_clusters = out.n_clusters
    _ = out^
    # DEVIATION 1946: the context dies LAST, after every value built on it.
    _ = ctx^
    return n_clusters


def prediction_data_door_run(
    parents: List[Int32],
    children: List[Int32],
    lambdas: List[Float32],
    sizes: List[Int32],
    n_edges: Int,
    n_leaves: Int,
    n_clusters: Int,
    labels: List[Int32],
    inv: List[Int32],
    n_selected: Int,
    dp: MutPointer[Float32, MutUntrackedOrigin],
    sp: MutPointer[Int32, MutUntrackedOrigin],
    ep: MutPointer[Int32, MutUntrackedOrigin],
    op: MutPointer[Int32, MutUntrackedOrigin],
    iicp: MutPointer[Int32, MutUntrackedOrigin],
) raises -> Int:
    """`generate_prediction_data_device` and its result copies; returns the
    exemplar count."""
    var ctx = process_ctx[_DEVCTX_SLOT]()
    var pd = generate_prediction_data_device(
        ctx, parents, children, lambdas, sizes, n_edges, n_leaves,
        n_clusters, labels, inv, n_selected,
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
    var n_ex = pd.n_exemplars
    _ = pd^
    _ = ctx^
    return n_ex
