# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Typed bodies of the HDBSCAN host bindings' doors
(`bindings/_mojolearn_hdbscan_host.mojo`, `bindings/hdbscan_host_predict.mojo`):
the fit, prediction-data and prediction runs with their result copies, moved
here with no Python import (lane thin-core, 2026-10-21: a pure move, no bit
change)."""
from hdbscan.host.hdbscan_host_oracle import (
    hdbh_all_points_membership_vectors,
    hdbh_approximate_predict,
    hdbh_fit,
    hdbh_membership_vector,
)
from hdbscan.impl.prediction_data import generate_prediction_data_host
from hdbscan.impl.detail.extract import probabilities_from_labels_host


def hdbscan_fit_host_run(x: List[Float32], n: Int, d: Int, min_samples: Int, min_cluster_size: Int, max_cluster_size: Int, alpha: Float32, allow_single: Bool, method: Int, eps: Float32, metric: Int, lp: MutPointer[Int32, MutUntrackedOrigin], cp: MutPointer[Float32, MutUntrackedOrigin], ip: MutPointer[Int32, MutUntrackedOrigin], want_tree: Bool, tpp: MutPointer[Int32, MutUntrackedOrigin], tcp: MutPointer[Int32, MutUntrackedOrigin], tlp: MutPointer[Float32, MutUntrackedOrigin], tsp: MutPointer[Int32, MutUntrackedOrigin], invp: MutPointer[Int32, MutUntrackedOrigin], want_probs: Bool, pp: MutPointer[Float32, MutUntrackedOrigin]) raises -> Int:
    """The GIL-free body of `hdbscan_fit_binding` (moved word for word)."""
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
    if want_probs:
        var probs = probabilities_from_labels_host(
            out.tree, out.labels, out.inverse_label_map, n
        )
        for i in range(n):
            pp.unsafe_store(i, probs[i])
    var n_clusters = out.n_clusters
    _ = out^
    return n_clusters


def prediction_data_host_run(parents: List[Int32], children: List[Int32], lambdas: List[Float32], sizes: List[Int32], n_edges: Int, n_leaves: Int, n_clusters: Int, labels: List[Int32], inv: List[Int32], n_selected: Int, dp: MutPointer[Float32, MutUntrackedOrigin], sp: MutPointer[Int32, MutUntrackedOrigin], ep: MutPointer[Int32, MutUntrackedOrigin], op: MutPointer[Int32, MutUntrackedOrigin], iicp: MutPointer[Int32, MutUntrackedOrigin]) raises -> Int:
    """The GIL-free body of `hdbscan_generate_prediction_data_binding` (moved word for word)."""
    var pd = generate_prediction_data_host(
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
    var n_ex = pd.n_exemplars
    _ = pd^
    return n_ex


def approximate_predict_host_run(x: List[Float32], m: Int, d: Int, core: List[Float32], labels: List[Int32], lambdas: List[Float32], deaths: List[Float32], selected: List[Int32], iic: List[Int32], q: List[Float32], nq: Int, min_samples: Int, lo: MutPointer[Int32, MutUntrackedOrigin], po: MutPointer[Float32, MutUntrackedOrigin]) raises:
    """The GIL-free body of `hdbscan_approximate_predict_binding` (moved word for word)."""
    var out = hdbh_approximate_predict(
        x, m, d, core, labels, lambdas, m, deaths, selected, iic, q, nq,
        min_samples,
    )
    for i in range(nq):
        lo.unsafe_store(i, out.labels[i])
        po.unsafe_store(i, out.probabilities[i])
    _ = out^


def membership_vector_host_run(x: List[Float32], m: Int, d: Int, core: List[Float32], labels: List[Int32], parents: List[Int32], lambdas: List[Float32], n_edges: Int, n_clusters: Int, deaths: List[Float32], selected: List[Int32], iic: List[Int32], ex_idx: List[Int32], offsets: List[Int32], q: List[Float32], nq: Int, min_samples: Int, n_selected: Int, dst: MutPointer[Float32, MutUntrackedOrigin]) raises:
    """The GIL-free body of `hdbscan_membership_vector_binding` (moved word for word)."""
    var mv = hdbh_membership_vector(
        x, m, d, core, labels, parents, lambdas, n_edges, n_clusters,
        deaths, selected, iic, ex_idx, offsets, q, nq, min_samples,
    )
    for i in range(nq * n_selected):
        dst.unsafe_store(i, mv[i])
    _ = mv^


def all_points_membership_host_run(x: List[Float32], m: Int, d: Int, parents: List[Int32], lambdas: List[Float32], n_edges: Int, n_clusters: Int, deaths: List[Float32], selected: List[Int32], iic: List[Int32], ex_idx: List[Int32], offsets: List[Int32], row0: Int, count: Int, n_selected: Int, dst: MutPointer[Float32, MutUntrackedOrigin]) raises:
    """The GIL-free body of `hdbscan_all_points_membership_vectors_binding` (moved word for word)."""
    var mv = hdbh_all_points_membership_vectors(
        x, m, d, parents, lambdas, n_edges, n_clusters, deaths, selected,
        iic, ex_idx, offsets, row0, count,
    )
    for i in range(count * n_selected):
        dst.unsafe_store(i, mv[i])
    _ = mv^
