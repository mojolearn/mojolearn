# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# SHIPS: compiled into a CPU host binding (python/mojolearn/host_surface.py names which); product, not only a check.
"""HDBSCAN on the HOST: `hdbscan/estimator.mojo::hdbscan_fit_host` and
`hdbscan/impl/runner.mojo::fit_hdbscan` restated without a device (CPU
training for the workstream D estimators, 2026-09-15).

WHAT THIS IS. The device fit runs its distances, its mutual reachability
graph, its Boruvka MST and its stabilities in kernels, and its condensing,
selection and labelling on the host behind device buffers. This file runs
the same stages in the same order with no `DeviceContext` and no card. The
stages already written as host code on the device path (the condensing
BFS and collapse, the cluster tree and its CSR, the union-find labelling,
the order keys, the mutual reachability max) are CALLED; the kernels are
restated beside the device line they mirror:

    the core distances      reachability.mojo::compute_core_dists: the k-NN
                            of every row against X at k = min_samples + 1
                            (core/knn_host_predict.mojo::host_knn_search,
                            the knn host lane's restatement), slot k - 1
    the pairwise distances  connectivities.mojo::pairwise_distances:
                            row_norm_kernel's pinned fold and
                            pinned_distance_tile_kernel
                            (hierarchy/checks/linkage_oracle.mojo::
                            host_row_norms_pinned, host_pinned_distance),
                            FLOAT32_MAX on the diagonal, the NaN refusal
    mutual reachability     mutual_reachability_dense_kernel (mr_scale,
                            mr_max3), then the non-finite refusal
    the MST and its rounds  sparse/solver/mst_solver.mojo's Boruvka on the
                            dense graph, serially: each vertex's minimum
                            outgoing edge under (weight key, lo, hi), each
                            color's minimum, the pair de-duplication by the
                            larger color, the color as the smallest vertex
                            of its merged component, and the round count
                            with its final empty round
    the sort and orientation
                            coo_sort_by_weight's (weight key, lo, hi) order,
                            then edge_lo and edge_hi
    the dendrogram          agglomerative.mojo::build_dendrogram_host
    the condensed tree      condense.mojo::build_condensed_hierarchy
    stabilities             stabilities.mojo::births_init_kernel and
                            cluster_stability_kernel
    selection               select.mojo::excess_of_mass (and perform_bfs's
                            negation) or leaf
    labels                  extract.mojo::do_labelling_on_host and the
                            label map

WHAT IS REFUSED, BY NAME, AS ON THE DEVICE: fewer than two rows, no column,
min_samples above n_rows or below 1, a metric other than L2SqrtExpanded, a
selection method other than eom and leaf, a non-positive or non-finite
alpha, more rows than PAIRWISE_MAX_ROWS, min_cluster_size below 2 or above
n_rows, a non-zero cluster_selection_epsilon where the epsilon search would
run, and NaN distances or non-finite mutual reachability cells.

THE SABOTAGE. `-D MOJOLEARN_HOST_SABOTAGE=1` reads the core distance one
slot early (the (k - 1)-th neighbor instead of the k-th), so every core
distance, and with it the whole graph, differs.
"""

from std.sys.compile import is_defined
from max.algorithm import sync_parallelize

from core.knn_host_predict import KNN_HOST_METRIC_FROM_IS_SQRT, host_knn_search
from core.host_predict_threads import (
    host_list_ptr,
    host_predict_chunk,
    host_predict_task_count,
)
from hdbscan.checks.hdbscan_sabotage import HDB_SAB_NONE, mr_max3, mr_scale
from hdbscan.checks.mutual_reachability_dense import refuse_nonfinite_host
from hdbscan.impl.condensed_hierarchy import CondensedHierarchy
from hdbscan.impl.prediction_data import (
    PredictionData,
    prediction_neighborhood,
    refuse_nonfinite_queries,
    refuse_soft_clustering_inputs,
)
from hdbscan.impl.detail.condense import _add_edge, _collapse, bfs_from_node
from hdbscan.impl.detail.extract import do_labelling_on_host
from hdbscan.impl.detail.stabilities import (
    stability_order_key_bits,
    stability_order_unkey_bits,
)
from hdbscan.impl.detail.utils import (
    make_cluster_tree,
    select_parent_csr,
    utils_parent_csr,
)
from hierarchy.checks.edge_order import (
    edge_hi,
    edge_lo,
    pack_edge_key,
    triple_less,
    weight_order_key,
)
from hierarchy.checks.linkage_oracle import (
    host_pinned_distance,
    host_row_norms_pinned,
)
from hierarchy.impl.sparse.op.sort import merge_sort_u64_with_index
from checks.numerics import (
    ftz,
    identical_div,
    identical_exp,
    identical_mul,
    identical_mul_add,
    identical_sqrt,
)
from std.memory import bitcast

comptime HDBH_HOST_SABOTAGE = is_defined["MOJOLEARN_HOST_SABOTAGE"]()

comptime HDBH_SELECTION_EOM = 0
comptime HDBH_SELECTION_LEAF = 1
comptime HDBH_METRIC_L2_SQRT_EXPANDED = 1
comptime HDBH_PAIRWISE_MAX_ROWS = 46340
comptime HDBH_FLOAT32_MAX = Float32(3.4028234663852886e38)
comptime HDBH_KEY_SENTINEL: Int32 = 0x7FFFFFFF


@fieldwise_init
struct HdbscanHostFit(Movable):
    var labels: List[Int32]
    var core_dists: List[Float32]
    var n_clusters: Int
    var n_outliers: Int
    var n_boruvka_rounds: Int
    var n_condensed_clusters: Int
    var tree: CondensedHierarchy
    """The condensed tree, kept for `prediction_data=True`
    (`hdbscan/impl/prediction_data.mojo`). Nothing above reads it."""
    var inverse_label_map: List[Int32]
    """final label -> condensed cluster id counted from 0, ascending, as
    `extract.mojo::extract_clusters` builds it."""


@fieldwise_init
struct HdbscanHostPredict(Movable):
    var labels: List[Int32]
    var probabilities: List[Float32]
    var min_mr_inds: List[Int32]
    """The nearest mutual reachability neighbor of each query, which
    `hdbh_membership_vector` reads."""
    var prediction_lambdas: List[Float32]
    """`1 / min_mr_dist` (FLOAT32_MAX at zero), before any sabotage."""


comptime HDBH_PREDICT_SABOTAGE = (
    HDBH_HOST_SABOTAGE or is_defined["MOJOLEARN_HDBSCAN_PREDICT_SABOTAGE"]()
)
"""The prediction pass's negative control. It rides on the family's
`-D MOJOLEARN_HOST_SABOTAGE=1`, and `-D MOJOLEARN_HDBSCAN_PREDICT_SABOTAGE=1`
arms it ALONE, so a fit stays correct and only the query pass moves."""


def hdbh_approximate_predict(
    x: List[Float32],
    m: Int,
    n: Int,
    input_core: List[Float32],
    labels: List[Int32],
    tree_lambdas: List[Float32],
    n_leaves: Int,
    deaths: List[Float32],
    selected_clusters: List[Int32],
    index_into_children: List[Int32],
    queries: List[Float32],
    nq: Int,
    min_samples: Int,
) raises -> HdbscanHostPredict:
    """`hdbscan/impl/detail/predict.mojo::approximate_predict` (cuML
    `predict.cuh:220-262`) on the host, restated beside each device line:
    the k-NN at `(min_samples - 1) * 2` through `host_knn_search`, slot
    `min_samples - 1` for the query core distance, the strict `>` scan for
    the nearest mutual reachability neighbor (DEVIATION 1615), `1 / d`, and
    the label and probability rule of `cluster_probability_kernel`."""
    if nq < 1:
        raise Error(
            "hdbscan.approximate_predict: points_to_predict has no rows;"
            " refused by name"
        )
    refuse_nonfinite_queries(queries, nq, n)
    var k = prediction_neighborhood(min_samples, m)
    var dist = List[Float32](length=nq * k, fill=Float32(0.0))
    var idx = List[UInt32](length=nq * k, fill=UInt32(0))
    host_knn_search(x, m, queries, nq, n, k, KNN_HOST_METRIC_FROM_IS_SQRT, True, dist, idx)
    var out_labels = List[Int32](capacity=nq)
    var out_probs = List[Float32](capacity=nq)
    var out_inds = List[Int32](capacity=nq)
    var out_lams = List[Float32](capacity=nq)
    var nl = Int32(n_leaves)
    for q in range(nq):
        # core_distances_kernel, slot min_samples - 1
        var pcore = dist[q * k + (min_samples - 1)]
        # min_mutual_reachability_kernel
        var best = HDBH_FLOAT32_MAX
        var best_ind = -1
        for i in range(k):
            var mr = pcore
            var nb = Int(idx[q * k + i])
            if input_core[nb] > mr:
                mr = input_core[nb]
            if dist[q * k + i] > mr:
                mr = dist[q * k + i]
            if best > mr:
                best = mr
                best_ind = nb
        # prediction_lambda_kernel
        var lam = HDBH_FLOAT32_MAX
        if best > Float32(0.0):
            lam = identical_div(Float32(1.0), best)
        out_inds.append(Int32(best_ind))
        out_lams.append(lam)
        # cluster_probability_kernel
        var cl = labels[best_ind]
        var got = Int32(-1)
        if cl >= Int32(0):
            var sel = selected_clusters[Int(cl)]
            if sel > nl:
                if tree_lambdas[Int(index_into_children[Int(sel)])] < lam:
                    got = cl
            elif sel == nl:
                got = cl
        out_labels.append(got)
        var prob = Float32(0.0)
        if got >= Int32(0):
            var max_lambda = deaths[Int(selected_clusters[Int(cl)] - nl)]
            if max_lambda > Float32(0.0):
                var num = max_lambda if max_lambda < lam else lam
                prob = identical_div(num, max_lambda)
            else:
                prob = Float32(1.0)
        comptime if HDBH_PREDICT_SABOTAGE:
            # THE SABOTAGE ARM: the lowest bit of every probability. Wrong
            # on purpose.
            prob = bitcast[DType.float32](
                bitcast[DType.uint32](prob) ^ UInt32(1)
            )
        out_probs.append(prob)
    return HdbscanHostPredict(out_labels^, out_probs^, out_inds^, out_lams^)


comptime HDBH_SOFT_FLOAT32_MAX = Float32(3.4028234663852886e38)
comptime HDBH_SOFT_EPS = Float32(1e-8)
comptime HDBH_SOFT_PREDICT = 0
comptime HDBH_SOFT_ALL_POINTS = 1


def hdbh_soft_normalize(mut v: List[Float32], base: Int, n: Int):
    """`soft_clustering.mojo::soft_normalize_row`: the ascending L1 fold,
    saturated at FLT_MAX, a zero sum left at zero (DEVIATION 1616)."""
    var s = Float32(0.0)
    for c in range(n):
        s = ftz(s + v[base + c])
    if s > HDBH_SOFT_FLOAT32_MAX:
        s = HDBH_SOFT_FLOAT32_MAX
    if s == Float32(0.0):
        return
    for c in range(n):
        v[base + c] = identical_div(v[base + c], s)


def hdbh_soft_pass(
    mode: Int,
    rows: List[Float32],
    n_rows: Int,
    n: Int,
    x: List[Float32],
    parents: List[Int32],
    lambdas: List[Float32],
    pd: PredictionData,
    min_mr_inds: List[Int32],
    prediction_lambdas: List[Float32],
    row0: Int,
) raises -> List[Float32]:
    """`hdbscan/impl/detail/soft_clustering.mojo::_soft_pass` on the host,
    each kernel restated beside its device line."""
    var ns = pd.n_selected_clusters
    var nx = pd.n_exemplars
    var nl = Int(pd.n_leaves)
    var cells = n_rows * ns
    # copy_rows, then soft_row_norm_kernel over the rows and the exemplars
    var ex = List[Float32](capacity=nx * n)
    for j in range(nx):
        for f in range(n):
            ex.append(x[Int(pd.exemplar_idx[j]) * n + f])
    var rnorm = List[Float32](capacity=n_rows)
    for r in range(n_rows):
        var acc = Float32(0.0)
        for f in range(n):
            var v = ftz(rows[r * n + f])
            acc = ftz(identical_mul_add(v, v, acc))
        rnorm.append(acc)
    var enorm = List[Float32](capacity=nx)
    for j in range(nx):
        var acc = Float32(0.0)
        for f in range(n):
            var v = ftz(ex[j * n + f])
            acc = ftz(identical_mul_add(v, v, acc))
        enorm.append(acc)
    var dist_mv = List[Float32](length=cells, fill=Float32(0.0))
    var heights = List[Float32](length=cells, fill=Float32(0.0))
    var outlier = List[Float32](length=cells, fill=Float32(0.0))
    var out = List[Float32](length=cells, fill=Float32(0.0))
    for r in range(n_rows):
        var base = r * ns
        # exemplar_min_dist_kernel
        var rn = ftz(rnorm[r])
        for c in range(ns):
            var best = HDBH_SOFT_FLOAT32_MAX
            for j in range(
                Int(pd.exemplar_label_offsets[c]),
                Int(pd.exemplar_label_offsets[c + 1]),
            ):
                var acc = Float32(0.0)
                for f in range(n):
                    acc = ftz(
                        identical_mul_add(ftz(rows[r * n + f]), ftz(ex[j * n + f]), acc)
                    )
                var dist = ftz(
                    identical_mul_add(Float32(-2.0), acc, ftz(rn + ftz(enorm[j])))
                )
                if dist <= Float32(0.0):
                    dist = Float32(0.0)
                dist = ftz(identical_sqrt(dist))
                if dist < best:
                    best = dist
            # dist_membership_kernel
            if best > Float32(0.0):
                dist_mv[base + c] = identical_div(Float32(1.0), best)
            else:
                dist_mv[base + c] = identical_div(
                    HDBH_SOFT_FLOAT32_MAX, Float32(ns)
                )
        hdbh_soft_normalize(dist_mv, base, ns)
        # soft_row_prep_kernel
        var point = row0 + r
        if mode == HDBH_SOFT_PREDICT:
            point = Int(min_mr_inds[r])
        var edge = Int(pd.index_into_children[point])
        var lam = lambdas[edge]
        if mode == HDBH_SOFT_PREDICT:
            if prediction_lambdas[r] < lam:
                lam = prediction_lambdas[r]
        var death = pd.deaths[Int(parents[edge]) - nl]
        # soft_merge_height_kernel
        var leaf_parent = parents[edge]
        for c in range(ns):
            var right = pd.selected_clusters[c]
            var left = leaf_parent
            var took_r = False
            var took_l = False
            var last = Int32(0)
            while left != right:
                if left > right:
                    took_l = True
                    last = left
                    left = parents[Int(pd.index_into_children[Int(left)])]
                else:
                    took_r = True
                    last = right
                    right = parents[Int(pd.index_into_children[Int(right)])]
            if took_l and took_r:
                heights[base + c] = lambdas[Int(pd.index_into_children[Int(last)])]
            else:
                heights[base + c] = lam
        # soft_outlier_kernel
        for c in range(ns):
            var h = heights[base + c]
            var o: Float32
            if mode == HDBH_SOFT_PREDICT:
                var den = ftz(death - h)
                if den <= Float32(0.0):
                    den = HDBH_SOFT_EPS
                o = identical_div(death, den)
                if o > HDBH_SOFT_FLOAT32_MAX:
                    o = HDBH_SOFT_FLOAT32_MAX
            else:
                var t = identical_div(ftz(death + HDBH_SOFT_EPS), h)
                if t > HDBH_SOFT_FLOAT32_MAX:
                    t = HDBH_SOFT_FLOAT32_MAX
                o = identical_exp(-t)
            outlier[base + c] = o
        var mx = outlier[base]
        for c in range(1, ns):
            if outlier[base + c] > mx:
                mx = outlier[base + c]
        for c in range(ns):
            outlier[base + c] = identical_exp(ftz(outlier[base + c] - mx))
        hdbh_soft_normalize(outlier, base, ns)
        # soft_prob_kernel
        var best_c = 0
        var bh = heights[base]
        for c in range(1, ns):
            if heights[base + c] > bh:
                bh = heights[base + c]
                best_c = c
        var dsel = pd.deaths[Int(pd.selected_clusters[best_c]) - nl]
        var ml = lam
        if ml < dsel:
            ml = dsel
        if mode == HDBH_SOFT_PREDICT:
            ml = ftz(ml + HDBH_SOFT_EPS)
        var prob = Float32(0.0)
        if ml > Float32(0.0):
            prob = identical_div(bh, ml)
        # soft_combine_kernel
        for c in range(ns):
            var mo = outlier[base + c]
            var dm = dist_mv[base + c]
            if mode == HDBH_SOFT_PREDICT:
                out[base + c] = ftz(
                    identical_mul(ftz(identical_mul(mo, mo)), ftz(identical_sqrt(dm)))
                )
            else:
                out[base + c] = ftz(identical_mul(dm, mo))
        hdbh_soft_normalize(out, base, ns)
        for c in range(ns):
            var v = ftz(identical_mul(out[base + c], prob))
            comptime if HDBH_PREDICT_SABOTAGE:
                # THE SABOTAGE ARM: the lowest bit of every membership cell.
                # Wrong on purpose.
                v = bitcast[DType.float32](bitcast[DType.uint32](v) ^ UInt32(1))
            out[base + c] = v
    return out^


def _hdbh_soft_pd(
    m: Int,
    n_edges: Int,
    n_clusters: Int,
    deaths: List[Float32],
    selected_clusters: List[Int32],
    index_into_children: List[Int32],
    exemplar_idx: List[Int32],
    exemplar_label_offsets: List[Int32],
) -> PredictionData:
    return PredictionData(
        m, n_edges, n_clusters, len(selected_clusters), len(exemplar_idx),
        deaths.copy(), exemplar_idx.copy(), exemplar_label_offsets.copy(),
        selected_clusters.copy(), index_into_children.copy(),
    )


def hdbh_membership_vector(
    x: List[Float32],
    m: Int,
    n: Int,
    input_core: List[Float32],
    labels: List[Int32],
    parents: List[Int32],
    tree_lambdas: List[Float32],
    n_edges: Int,
    n_clusters: Int,
    deaths: List[Float32],
    selected_clusters: List[Int32],
    index_into_children: List[Int32],
    exemplar_idx: List[Int32],
    exemplar_label_offsets: List[Int32],
    queries: List[Float32],
    nq: Int,
    min_samples: Int,
) raises -> List[Float32]:
    """`soft_clustering.mojo::membership_vector` (cuML
    `soft_clustering.cuh:501-627`) on the host: the refusals in its order,
    `hdbh_approximate_predict` for the nearest neighbor and its lambda,
    then `hdbh_soft_pass`."""
    if nq < 1:
        raise Error(
            "hdbscan.membership_vector: points_to_predict has no rows;"
            " refused by name"
        )
    refuse_nonfinite_queries(queries, nq, n)
    var pd = _hdbh_soft_pd(
        m, n_edges, n_clusters, deaths, selected_clusters, index_into_children,
        exemplar_idx, exemplar_label_offsets,
    )
    refuse_soft_clustering_inputs(
        parents, tree_lambdas, pd, m, "hdbscan.membership_vector"
    )
    var near = hdbh_approximate_predict(
        x, m, n, input_core, labels, tree_lambdas, m, deaths,
        selected_clusters, index_into_children, queries, nq, min_samples,
    )
    return hdbh_soft_pass(
        HDBH_SOFT_PREDICT, queries, nq, n, x, parents, tree_lambdas, pd,
        near.min_mr_inds, near.prediction_lambdas, 0,
    )


def hdbh_all_points_membership_vectors(
    x: List[Float32],
    m: Int,
    n: Int,
    parents: List[Int32],
    tree_lambdas: List[Float32],
    n_edges: Int,
    n_clusters: Int,
    deaths: List[Float32],
    selected_clusters: List[Int32],
    index_into_children: List[Int32],
    exemplar_idx: List[Int32],
    exemplar_label_offsets: List[Int32],
    row0: Int,
    count: Int,
) raises -> List[Float32]:
    """`soft_clustering.mojo::all_points_membership_vectors` (cuML
    `soft_clustering.cuh:385-482`) on the host, training rows `row0 ..
    row0 + count - 1`."""
    if count < 1 or row0 < 0 or row0 + count > m:
        raise Error(
            "hdbscan.all_points_membership_vectors: rows " + String(row0)
            + " + " + String(count) + " are outside the " + String(m)
            + " training rows; refused by name"
        )
    var pd = _hdbh_soft_pd(
        m, n_edges, n_clusters, deaths, selected_clusters, index_into_children,
        exemplar_idx, exemplar_label_offsets,
    )
    refuse_soft_clustering_inputs(
        parents, tree_lambdas, pd, m, "hdbscan.all_points_membership_vectors"
    )
    var rows = List[Float32](capacity=count * n)
    for i in range(count * n):
        rows.append(x[row0 * n + i])
    return hdbh_soft_pass(
        HDBH_SOFT_ALL_POINTS, rows, count, n, x, parents, tree_lambdas, pd,
        List[Int32](), List[Float32](), row0,
    )


@fieldwise_init
struct HdbscanHostMst(Movable):
    var src: List[Int32]
    var dst: List[Int32]
    var weights: List[Float32]
    var rounds: Int


def hdbh_core_distances(
    x: List[Float32], m: Int, n: Int, k: Int
) raises -> List[Float32]:
    """`compute_core_dists`: the k-NN of X against itself, sqrt distances,
    then `core_distances_kernel`'s slot `k - 1`."""
    var dist = List[Float32](length=m * k, fill=Float32(0.0))
    var idx = List[UInt32](length=m * k, fill=UInt32(0))
    host_knn_search(x, m, x, m, n, k, KNN_HOST_METRIC_FROM_IS_SQRT, True, dist, idx)
    var slot = k - 1
    comptime if HDBH_HOST_SABOTAGE:
        # THE SABOTAGE ARM: one neighbor early. Wrong on purpose.
        if slot > 0:
            slot = slot - 1
    var core = List[Float32](capacity=m)
    for row in range(m):
        core.append(dist[row * k + slot])
    refuse_nonfinite_host(
        core, "hdbscan.compute_core_dists", "core distances", HDB_SAB_NONE
    )
    return core^


def hdbh_mutual_reachability(
    x: List[Float32], m: Int, n: Int, core: List[Float32], alpha: Float32
) raises -> List[Float32]:
    """`pairwise_distances` at L2SqrtExpanded, then
    `mutual_reachability_dense_kernel`, written cell by cell into one
    `m x m` array."""
    var norms = host_row_norms_pinned(x, m, n)
    var inv_alpha = identical_div(Float32(1.0), alpha)
    var mr = List[Float32](length=m * m, fill=Float32(0.0))
    var xp = host_list_ptr(x)
    var np = host_list_ptr(norms)
    var cp = host_list_ptr(core)
    var mp = host_list_ptr(mr)
    var tasks = host_predict_task_count(m)
    var chunk = host_predict_chunk(m, tasks)

    def _rows(c: Int) {imm xp, imm np, imm cp, imm mp, imm chunk, imm m, imm n, imm inv_alpha}:
        var lo = c * chunk
        var hi = min(lo + chunk, m)
        for row in range(lo, hi):
            mp.unsafe_store(row * m + row, HDBH_FLOAT32_MAX)
            # Expanded L2 is symmetric.  One task owns each undirected
            # edge and writes its two directed dense cells.
            for col in range(row + 1, m):
                # `host_pinned_distance`, over pointers so row tasks share
                # only immutable inputs and write disjoint dense cells.
                var acc = Float32(0.0)
                for f in range(n):
                    var qv = ftz(xp.unsafe_load(row * n + f))
                    var yv = ftz(xp.unsafe_load(col * n + f))
                    acc = ftz(identical_mul_add(qv, yv, acc))
                var d = ftz(identical_mul_add(
                    Float32(-2.0), acc,
                    ftz(ftz(np.unsafe_load(row)) + ftz(np.unsafe_load(col))),
                ))
                if d <= Float32(0.0):
                    d = Float32(0.0)
                d = ftz(identical_sqrt(d))
                var value = mr_max3(
                    cp.unsafe_load(row), cp.unsafe_load(col),
                    mr_scale(inv_alpha, d), HDB_SAB_NONE,
                )
                mp.unsafe_store(row * m + col, value)
                mp.unsafe_store(col * m + row, value)

    if tasks == 1:
        _rows(0)
    else:
        sync_parallelize(_rows, tasks)
    var n_nan = 0
    for cell in range(m * m):
        if mr[cell] != mr[cell]:
            n_nan += 1
    if n_nan != 0:
        raise Error(
            "hierarchy.pairwise_distances: " + String(n_nan) + " of "
            + String(m * m) + " distance cells are NaN (a non-finite input"
            " row, or two rows whose squared norms overflow Float32); refused"
            " by name (DEVIATION 623, IDENTITY_PATHS row 39)"
        )
    refuse_nonfinite_host(
        mr, "hdbscan.build_mr_linkage", "mutual reachability cells", HDB_SAB_NONE
    )
    return mr^


def hdbh_boruvka(mr: List[Float32], m: Int) raises -> HdbscanHostMst:
    """`MST_solver.solve` on the dense graph (`symmetrize_output=False`,
    colors initialized), serially. Returns the edges in the order
    `compact_new_edges_kernel` appends them (rounds ascending, vertex
    ascending within a round) and the round count, which includes the final
    round that finds nothing."""
    var color = List[Int](capacity=m)
    for v in range(m):
        color.append(v)
    var src = List[Int32]()
    var dst = List[Int32]()
    var weights = List[Float32]()
    var rounds = 0
    var cand = List[Int](length=m, fill=-1)
    var cand_wk = List[Int32](length=m, fill=HDBH_KEY_SENTINEL)
    var cand_lo = List[Int32](length=m, fill=Int32(0))
    var cand_hi = List[Int32](length=m, fill=Int32(0))
    var cmin_wk = List[Int32](length=m, fill=HDBH_KEY_SENTINEL)
    var cmin_lo = List[Int32](length=m, fill=Int32(0))
    var cmin_hi = List[Int32](length=m, fill=Int32(0))
    var mrp = rebind[MutPointer[Float32, MutUntrackedOrigin]](mr.unsafe_ptr())
    var colorp = rebind[MutPointer[Int, MutUntrackedOrigin]](color.unsafe_ptr())
    var candp = rebind[MutPointer[Int, MutUntrackedOrigin]](cand.unsafe_ptr())
    var candwkp = rebind[MutPointer[Int32, MutUntrackedOrigin]](cand_wk.unsafe_ptr())
    var candlop = rebind[MutPointer[Int32, MutUntrackedOrigin]](cand_lo.unsafe_ptr())
    var candhip = rebind[MutPointer[Int32, MutUntrackedOrigin]](cand_hi.unsafe_ptr())
    var tasks = host_predict_task_count(m)
    if m * m < (1 << 14):
        tasks = 1
    var chunk = host_predict_chunk(m, tasks)

    def _vertex_min(task: Int) {imm mrp, imm colorp, imm candp, imm candwkp, imm candlop, imm candhip, imm chunk, imm m}:
        var lo_u = task * chunk
        var hi_u = min(lo_u + chunk, m)
        for u in range(lo_u, hi_u):
            var best = -1
            var bwk = HDBH_KEY_SENTINEL
            var blo = Int32(0x7FFFFFFF)
            var bhi = Int32(0x7FFFFFFF)
            var cu = colorp.unsafe_load(u)
            var base = u * m
            for j in range(m):
                if colorp.unsafe_load(j) == cu:
                    continue
                var wk = weight_order_key(mrp.unsafe_load(base + j))
                var edge_l = edge_lo(Int32(u), Int32(j))
                var edge_h = edge_hi(Int32(u), Int32(j))
                if triple_less(wk, edge_l, edge_h, bwk, blo, bhi):
                    bwk = wk
                    blo = edge_l
                    bhi = edge_h
                    best = j
            if bwk == HDBH_KEY_SENTINEL:
                best = -1
            candp.unsafe_store(u, best)
            candwkp.unsafe_store(u, bwk)
            candlop.unsafe_store(u, blo)
            candhip.unsafe_store(u, bhi)

    for _it in range(m):
        # kernel_min_edge_per_vertex: the minimum triple over a vertex's
        # row, restricted to edges leaving its color.
        for c in range(m):
            cmin_wk[c] = HDBH_KEY_SENTINEL
            cmin_lo[c] = Int32(0x7FFFFFFF)
            cmin_hi[c] = Int32(0x7FFFFFFF)
        if tasks > 1:
            sync_parallelize(_vertex_min, tasks)
        else:
            _vertex_min(0)
        # Preserve min_edge_per_color's original ascending-vertex fold.  The
        # parallel region changes only where each independent row scan runs.
        for u in range(m):
            var best = cand[u]
            var bwk = cand_wk[u]
            var blo = cand_lo[u]
            var bhi = cand_hi[u]
            var cu = color[u]
            # min_edge_{,lo_,hi_}per_color: the color's minimum triple.
            if best >= 0 and triple_less(bwk, blo, bhi, cmin_wk[cu], cmin_lo[cu], cmin_hi[cu]):
                cmin_wk[cu] = bwk
                cmin_lo[cu] = blo
                cmin_hi[cu] = bhi
        # min_edge_per_supervertex, at symmetrize_output = False.
        var added = 0
        var new_src = List[Int32]()
        var new_dst = List[Int32]()
        for u in range(m):
            var j = cand[u]
            if j < 0:
                continue
            var cu = color[u]
            if not (
                cand_wk[u] == cmin_wk[cu]
                and cand_lo[u] == cmin_lo[cu]
                and cand_hi[u] == cmin_hi[cu]
            ):
                continue
            var cj = color[j]
            var add_edge = True
            if cand[j] == u:
                var j_is_min = (
                    cand_wk[j] == cmin_wk[cj]
                    and cand_lo[j] == cmin_lo[cj]
                    and cand_hi[j] == cmin_hi[cj]
                )
                if j_is_min and cu > cj:
                    add_edge = False
            if add_edge:
                new_src.append(Int32(u))
                new_dst.append(Int32(j))
                added += 1
        rounds += 1
        if added == 0:
            break
        for e in range(added):
            var u = Int(new_src[e])
            var j = Int(new_dst[e])
            src.append(Int32(u))
            dst.append(Int32(j))
            weights.append(mr[u * m + j])
        # label_prop: every vertex takes the smallest color of its merged
        # component. A union-find over the old colors and the new edges.
        var parent = List[Int](capacity=m)
        for v in range(m):
            parent.append(v)
        for e in range(added):
            var a = color[Int(new_src[e])]
            var b = color[Int(new_dst[e])]
            while parent[a] != a:
                a = parent[a]
            while parent[b] != b:
                b = parent[b]
            if a != b:
                if a < b:
                    parent[b] = a
                else:
                    parent[a] = b
        for v in range(m):
            var r = color[v]
            while parent[r] != r:
                r = parent[r]
            color[v] = r
    if len(src) != m - 1:
        raise Error(
            "hierarchy.build_sorted_mst: the MST has " + String(len(src))
            + " edges for " + String(m) + " vertices; a spanning tree has m - 1"
        )
    return HdbscanHostMst(src^, dst^, weights^, rounds)


@fieldwise_init
struct HdbscanHostDendrogram(Movable):
    var children: List[Int32]
    var deltas: List[Float32]
    var sizes: List[Int32]


def hdbh_dendrogram(
    lo: List[Int32], hi: List[Int32], w: List[Float32], n_edges: Int
) -> HdbscanHostDendrogram:
    """`build_dendrogram_host`: a union-find over the sorted MST."""
    var n_leaves = n_edges + 1
    var parent = List[Int](length=2 * n_leaves - 1, fill=-1)
    var size = List[Int](length=2 * n_leaves - 1, fill=0)
    for i in range(n_leaves):
        size[i] = 1
    var next_label = n_leaves
    var children = List[Int32](capacity=2 * n_edges)
    var deltas = List[Float32](capacity=n_edges)
    var sizes = List[Int32](capacity=n_edges)
    for i in range(n_edges):
        var aa = Int(lo[i])
        while parent[aa] != -1:
            aa = parent[aa]
        var bb = Int(hi[i])
        while parent[bb] != -1:
            bb = parent[bb]
        children.append(Int32(aa))
        children.append(Int32(bb))
        deltas.append(w[i])
        sizes.append(Int32(size[aa] + size[bb]))
        size[next_label] = size[aa] + size[bb]
        parent[aa] = next_label
        parent[bb] = next_label
        next_label += 1
    return HdbscanHostDendrogram(children^, deltas^, sizes^)


def hdbh_condense(
    h_children: List[Int32],
    h_delta: List[Float32],
    h_sizes: List[Int32],
    min_cluster_size: Int,
    n_leaves: Int,
) raises -> CondensedHierarchy:
    """`build_condensed_hierarchy` over host lists, its helpers called."""
    if min_cluster_size < 2:
        raise Error(
            "hdbscan.build_condensed_hierarchy: min_cluster_size="
            + String(min_cluster_size)
            + " refused by name; it must be at least 2"
        )
    if min_cluster_size > n_leaves:
        raise Error(
            "hdbscan.build_condensed_hierarchy: min_cluster_size="
            + String(min_cluster_size) + " > n_rows=" + String(n_leaves)
            + " refused by name; no subtree can reach that size"
        )
    var root = 2 * (n_leaves - 1)
    var n_samples = n_leaves
    refuse_nonfinite_host(
        h_delta, "hdbscan.build_condensed_hierarchy", "dendrogram deltas",
        HDB_SAB_NONE,
    )
    var next_label = n_samples + 1
    var node_list = List[Int32]()
    bfs_from_node(root, n_samples, h_children, node_list, HDB_SAB_NONE)
    var relabel = List[Int](length=root + 1, fill=0)
    var ignore = List[Int](length=root + 1, fill=0)
    relabel[root] = n_samples
    var out_parent = List[Int32]()
    var out_child = List[Int32]()
    var out_lambda = List[Float32]()
    var out_size = List[Int32]()
    for idx in range(len(node_list)):
        var node = Int(node_list[idx])
        if ignore[node] != 0 or node < n_samples:
            continue
        var left = Int(h_children[(node - n_samples) * 2])
        var right = Int(h_children[(node - n_samples) * 2 + 1])
        var distance = h_delta[node - n_samples]
        var lambda_value = HDBH_FLOAT32_MAX
        if distance > Float32(0.0):
            lambda_value = identical_div(Float32(1.0), distance)
        var left_count = 1
        if left >= n_samples:
            left_count = Int(h_sizes[left - n_samples])
        var right_count = 1
        if right >= n_samples:
            right_count = Int(h_sizes[right - n_samples])
        if left_count >= min_cluster_size and right_count >= min_cluster_size:
            relabel[left] = next_label
            next_label += 1
            _add_edge(
                out_parent, out_child, out_lambda, out_size,
                relabel[node], relabel[left], lambda_value, left_count,
            )
            relabel[right] = next_label
            next_label += 1
            _add_edge(
                out_parent, out_child, out_lambda, out_size,
                relabel[node], relabel[right], lambda_value, right_count,
            )
        elif left_count < min_cluster_size and right_count < min_cluster_size:
            _collapse(
                left, node, n_samples, h_children, relabel, ignore,
                out_parent, out_child, out_lambda, out_size, lambda_value,
                HDB_SAB_NONE,
            )
            _collapse(
                right, node, n_samples, h_children, relabel, ignore,
                out_parent, out_child, out_lambda, out_size, lambda_value,
                HDB_SAB_NONE,
            )
        elif left_count < min_cluster_size:
            relabel[right] = relabel[node]
            _collapse(
                left, node, n_samples, h_children, relabel, ignore,
                out_parent, out_child, out_lambda, out_size, lambda_value,
                HDB_SAB_NONE,
            )
        else:
            relabel[left] = relabel[node]
            _collapse(
                right, node, n_samples, h_children, relabel, ignore,
                out_parent, out_child, out_lambda, out_size, lambda_value,
                HDB_SAB_NONE,
            )
    var tree = CondensedHierarchy(n_leaves)
    tree.condense(out_parent, out_child, out_lambda, out_size)
    refuse_nonfinite_host(
        tree.lambdas, "hdbscan.build_condensed_hierarchy",
        "condensed tree lambdas", HDB_SAB_NONE,
    )
    return tree^


def hdbh_stabilities(tree: CondensedHierarchy) raises -> List[Float32]:
    """`compute_stabilities`: `births_init_kernel` then
    `cluster_stability_kernel`, one cluster at a time."""
    var n_clusters = tree.n_clusters
    var n_edges = tree.n_edges
    if n_clusters < 1:
        raise Error(
            "hdbscan.compute_stabilities: n_clusters=" + String(n_clusters)
            + " < 1; the condensed tree has no cluster to score"
        )
    var indptr = utils_parent_csr(tree)
    var births = List[Float32](length=n_clusters, fill=Float32(0.0))
    for i in range(n_edges):
        var child = Int(tree.children[i])
        if child >= tree.n_leaves:
            births[child - tree.n_leaves] = tree.lambdas[i]
    var stab = List[Float32](length=n_clusters, fill=Float32(0.0))
    for c in range(n_clusters):
        var lo = Int(indptr[c])
        var hi = Int(indptr[c + 1])
        var seg_key = stability_order_key_bits(
            bitcast[DType.uint32](HDBH_FLOAT32_MAX)
        )
        for i in range(lo, hi):
            var lam_key = stability_order_key_bits(
                bitcast[DType.uint32](tree.lambdas[i])
            )
            if lam_key < seg_key:
                seg_key = lam_key
        var birth_bits = bitcast[DType.uint32](births[c])
        if c > 0:
            if seg_key < stability_order_key_bits(birth_bits):
                birth_bits = stability_order_unkey_bits(seg_key)
            births[c] = bitcast[DType.float32](birth_bits)
        var birth = bitcast[DType.float32](birth_bits)
        var acc = Float32(0.0)
        for i in range(lo, hi):
            var term = ftz(tree.lambdas[i] - birth)
            var size_f = tree.sizes[i].cast[DType.float32]()
            acc = ftz(identical_mul_add(term, size_f, acc))
        stab[c] = acc
    return stab^


def hdbh_select(
    tree: CondensedHierarchy,
    stab_in: List[Float32],
    method: Int,
    allow_single_cluster: Bool,
    max_cluster_size: Int,
    cluster_selection_epsilon: Float32,
) raises -> List[Int32]:
    """`select_clusters`: `excess_of_mass` (with `perform_bfs`'s negation)
    or `leaf`, then the epsilon refusal. Returns `is_cluster`."""
    var n_clusters = tree.n_clusters
    var cluster_tree = make_cluster_tree(tree)
    var n_edges = cluster_tree.n_edges
    var is_cluster = List[Int32](length=n_clusters, fill=Int32(0))
    if method == HDBH_SELECTION_EOM:
        var cluster_sizes = List[Int](length=n_clusters, fill=0)
        for i in range(n_edges):
            var parent = Int(cluster_tree.parents[i])
            var child = Int(cluster_tree.children[i])
            var size = Int(cluster_tree.sizes[i])
            if parent == 0:
                cluster_sizes[0] += size
            if child < 0 or child >= n_clusters:
                raise Error(
                    "hdbscan.excess_of_mass: cluster-tree child " + String(child)
                    + " at edge " + String(i) + " is outside [0, "
                    + String(n_clusters) + ")"
                )
            cluster_sizes[child] = size
        var frontier = List[Int32](length=n_clusters, fill=Int32(0))
        for i in range(n_clusters):
            is_cluster[i] = Int32(1)
        is_cluster[0] = Int32(1) if allow_single_cluster else Int32(0)
        var indptr = select_parent_csr(cluster_tree)
        var stab = stab_in.copy()
        var tree_top = 0 if allow_single_cluster else 1
        var node = n_clusters - 1
        while node >= tree_top:
            var node_stability = stab[node]
            var lo = Int(indptr[node])
            var hi = Int(indptr[node + 1])
            var subtree_stability = Float32(0.0)
            if hi - lo > 0:
                for i in range(lo, hi):
                    var child = Int(cluster_tree.children[i])
                    subtree_stability = ftz(
                        identical_mul_add(Float32(1.0), stab[child], subtree_stability)
                    )
            if (
                subtree_stability > node_stability
                or cluster_sizes[node] > max_cluster_size
            ):
                stab[node] = subtree_stability
                is_cluster[node] = Int32(0)
            else:
                frontier[node] = Int32(1)
            node -= 1
        # perform_bfs
        var n_left = 0
        for i in range(n_clusters):
            n_left += Int(frontier[i])
        while n_left > 0:
            var next_frontier = List[Int32](length=n_clusters, fill=Int32(0))
            for cluster in range(n_clusters):
                if frontier[cluster] == Int32(0):
                    continue
                frontier[cluster] = Int32(0)
                for i in range(Int(indptr[cluster]), Int(indptr[cluster + 1])):
                    var child = Int(cluster_tree.children[i])
                    next_frontier[child] = Int32(1)
                    is_cluster[child] = Int32(0)
            frontier = next_frontier^
            n_left = 0
            for i in range(n_clusters):
                n_left += Int(frontier[i])
    elif method == HDBH_SELECTION_LEAF:
        if n_edges > 0:
            var is_parent = List[Int32](length=n_clusters, fill=Int32(0))
            for i in range(n_edges):
                var p = Int(cluster_tree.parents[i])
                if p >= 0 and p < n_clusters:
                    is_parent[p] = Int32(1)
            for i in range(n_edges):
                var c = Int(cluster_tree.children[i])
                if c >= 0 and c < n_clusters:
                    if is_parent[c] == Int32(0):
                        is_cluster[c] = Int32(1)
    else:
        raise Error(
            "hdbscan.select_clusters: cluster_selection_method="
            + String(method)
            + " refused by name; their enum has exactly two values, EOM=0"
            " and LEAF=1 (hdbscan.hpp:126)"
        )
    var n_selected = 0
    for i in range(n_clusters):
        if is_cluster[i] != Int32(0):
            n_selected += 1
    if cluster_selection_epsilon != Float32(0.0) and n_edges > 0:
        var epsilon_search = n_selected != 0
        if method == HDBH_SELECTION_EOM and n_selected == 1:
            if is_cluster[0] != Int32(0) and allow_single_cluster:
                epsilon_search = False
        if epsilon_search:
            raise Error(
                "hdbscan.cluster_epsilon_search: cluster_selection_epsilon"
                " refused by name; the epsilon search is NOT IMPLEMENTED"
                " (rung 2). Use cluster_selection_epsilon=0.0, their default"
            )
    return is_cluster^


def hdbh_fit(
    x: List[Float32],
    m: Int,
    n: Int,
    min_samples: Int,
    min_cluster_size: Int,
    max_cluster_size_in: Int,
    alpha: Float32,
    allow_single_cluster: Bool,
    method: Int,
    cluster_selection_epsilon: Float32,
    metric: Int,
) raises -> HdbscanHostFit:
    """`hdbscan_fit_host` then `fit_hdbscan`, in their order."""
    if m < 2:
        raise Error("hdbscan_fit_host needs n_rows >= 2, got " + String(m))
    if n < 1:
        raise Error("hdbscan_fit_host needs n_cols >= 1, got " + String(n))
    if min_samples > m:
        raise Error(
            "hdbscan.fit_hdbscan: min_samples must be at most the number of"
            " samples in X (min_samples=" + String(min_samples)
            + ", n_rows=" + String(m) + ")"
        )
    if metric != HDBH_METRIC_L2_SQRT_EXPANDED:
        raise Error(
            "hdbscan.fit_hdbscan: metric=" + String(metric)
            + " refused by name; Currently only L2 expanded distance is"
            " supported (their RAFT_EXPECTS, reachability.cuh:109)"
        )
    if method != HDBH_SELECTION_EOM and method != HDBH_SELECTION_LEAF:
        raise Error(
            "hdbscan.fit_hdbscan: cluster_selection_method=" + String(method)
            + " refused by name; their enum has exactly two values, EOM=0"
            " and LEAF=1 (hdbscan.hpp:126)"
        )
    # effective_min_samples
    if min_samples < 1:
        raise Error(
            "hdbscan: min_samples=" + String(min_samples) + " refused by"
            " name; it must be at least 1"
        )
    var k = min_samples + 1
    if min_samples + 1 > m:
        k = m
    if m > HDBH_PAIRWISE_MAX_ROWS:
        raise Error(
            "hdbscan.build_mr_linkage: n_rows=" + String(m) + " > "
            + String(HDBH_PAIRWISE_MAX_ROWS)
            + "; the dense mutual reachability graph is m * m cells"
        )
    if not (alpha > Float32(0.0)) or alpha > HDBH_FLOAT32_MAX:
        raise Error(
            "hdbscan.build_mr_linkage: alpha refused by name; alpha must be"
            " finite and strictly positive"
        )

    var core = hdbh_core_distances(x, m, n, k)
    var mr = hdbh_mutual_reachability(x, m, n, core, alpha)
    var mst = hdbh_boruvka(mr, m)
    _ = mr^

    # coo_sort_by_weight, then the orientation.
    var n_edges = m - 1
    var keys = List[UInt64](capacity=n_edges)
    var order = List[Int](capacity=n_edges)
    for i in range(n_edges):
        var u = mst.src[i]
        var v = mst.dst[i]
        keys.append(pack_edge_key(weight_order_key(mst.weights[i]), edge_lo(u, v), edge_hi(u, v)))
        order.append(i)
    merge_sort_u64_with_index(keys, order)
    var lo = List[Int32](capacity=n_edges)
    var hi = List[Int32](capacity=n_edges)
    var w = List[Float32](capacity=n_edges)
    for t in range(n_edges):
        var i = order[t]
        lo.append(edge_lo(mst.src[i], mst.dst[i]))
        hi.append(edge_hi(mst.src[i], mst.dst[i]))
        w.append(mst.weights[i])

    var dendro = hdbh_dendrogram(lo, hi, w, n_edges)
    var tree = hdbh_condense(
        dendro.children, dendro.deltas, dendro.sizes, min_cluster_size, m
    )
    var stab = hdbh_stabilities(tree)
    var max_cluster_size = max_cluster_size_in
    if max_cluster_size <= 0:
        max_cluster_size = m
    var is_cluster = hdbh_select(
        tree, stab, method, allow_single_cluster, max_cluster_size,
        cluster_selection_epsilon,
    )

    var n_clusters = tree.n_clusters
    var label_map = List[Int32](length=n_clusters, fill=Int32(-1))
    var n_selected = 0
    for i in range(n_clusters):
        if is_cluster[i] != Int32(0):
            label_map[i] = Int32(n_selected)
            n_selected += 1
    var in_clusters = List[Int32](length=m + n_clusters, fill=Int32(0))
    for i in range(n_clusters):
        if is_cluster[i] != Int32(0):
            in_clusters[i + m] = Int32(1)
    var raw = do_labelling_on_host(
        tree, in_clusters, m, allow_single_cluster, cluster_selection_epsilon
    )
    var labels = List[Int32](capacity=m)
    var n_outliers = 0
    for i in range(m):
        var l = raw[i]
        if l != Int32(-1):
            labels.append(label_map[Int(l)])
        else:
            labels.append(Int32(-1))
        if labels[i] == Int32(-1):
            n_outliers += 1
    var inverse_label_map = List[Int32](capacity=n_selected)
    for i in range(n_clusters):
        if is_cluster[i] != Int32(0):
            inverse_label_map.append(Int32(i))
    var n_condensed = tree.n_clusters
    return HdbscanHostFit(
        labels^, core^, n_selected, n_outliers, mst.rounds, n_condensed,
        tree^, inverse_label_map^,
    )
