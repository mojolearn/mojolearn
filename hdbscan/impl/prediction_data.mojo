# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`PredictionData` and `generate_prediction_data`, the `prediction_data=True`
half of a fit.

Reference: `cuml-v26.08.00/cpp/src/hdbscan/prediction_data.cu` (`allocate`
`:34-46`, `build_index_into_children` `:60-79`, `generate_prediction_data`
`:92-239`) and the container at `cpp/include/cuml/cluster/hdbscan.hpp:369-435`.
Steps run in the reference order.

WHAT IT HOLDS (their getters, `hdbscan.hpp:401-408`)
  deaths               n_clusters, the largest child lambda under each
                       condensed cluster (`:112-145`)
  exemplar_idx         the points of a LEAF cluster that are still in it
                       at its death, grouped by label (`:158-211`)
  exemplar_label_offsets  n_selected + 1, the CSR over those groups
                       (`:221-226`)
  selected_clusters    n_selected, condensed node id of each final label
                       (`:228-234`)
  index_into_children  n_edges + 1, node id -> its edge in the condensed
                       tree (`:236-237`)
  core_dists           theirs keeps a POINTER to the fit's; the Python
                       estimator keeps `core_distances_` for the same use.

======================================================================
DEVIATION BLOCK -- DEVIATION 1612. PREDICTION DATA IS BUILT ON THE DEVICE
BY THE DEVICE BINDING AND ON THE HOST BY THE CPU COLUMN, IN ONE ORDER.
======================================================================
WHAT THEIRS DOES. Thrust transforms and CUB segmented reductions over
the condensed tree's device vectors.

WHAT OURS DOES. The device binding calls `generate_prediction_data_device`
(lane cgr2-hdbscan, 2026-10-03): `deaths` an integer `Atomic.max` on
`weight_order_key` per edge, `is_leaf_cluster` and `is_exemplar` constant
writes per edge, the exemplar compaction an exclusive scan, the exemplar
order one stable radix sort keyed by condensed label over the ascending
indices, the label offsets an atomic count and a scan, `selected_clusters`
and `index_into_children` one thread per slot. The CPU binding calls
`generate_prediction_data`, the same passes as host loops. No pass does
arithmetic; `deaths` is a MAX over lambdas the fit has refused to be NaN
(DEVIATION 1607), and the key order and the float order agree on them
(lambdas are positive or FLT_MAX, so no signed-zero pair arises). The two
give the same arrays.

THE ONE ORDER CHOICE. Their `thrust::sort_by_key` of the exemplars by
original label (`:208-211`) is not specified stable. Ours sorts on the
packed key `(label << 32) | point index` (the `merge_sort_u64_with_index`
DEVIATION 1611 already uses), a total order that equals a stable sort of
the ascending `copy_if` output. The exemplar ORDER feeds nothing but a
minimum over each label's exemplars in `membership_vector`, which does not
depend on it; the array itself is what a caller could read.

THE UNWRITTEN SLOTS. Their `is_exemplar` and `index_into_children` are
`rmm::device_uvector`s, uninitialized, and `index_into_children[root]` is
never written (the root is never a child). Every leaf is a child exactly
once, so `is_exemplar` is fully written; the root slot of
`index_into_children` is filled with -1 here and no reader reaches it
(`kernels/predict.cuh:64-65` reads it only for a selected cluster above the
root, `kernels/soft_clustering.cuh:34-44` climbs only from a larger id).
======================================================================
"""

from std.atomic import Atomic
from max.gpu.host import DeviceContext

from hdbscan.impl.detail.tree_device import (
    F32P,
    I32P,
    TD_TPB,
    U32P,
    _gid,
    td_download_f32,
    td_download_i32,
    td_exclusive_scan,
    td_grid,
    td_read_i32,
    td_sort_pairs,
    td_upload_f32,
    td_upload_i32,
)
from hierarchy.checks.edge_order import weight_order_key, weight_order_unkey
from hierarchy.impl.sparse.op.sort import merge_sort_u64_with_index


comptime PD_FLOAT32_LOWEST = Float32(-3.4028234663852886e38)
"""`std::numeric_limits<float>::lowest()`, CUB `DeviceSegmentedReduce::Max`'s
initial value."""


struct PredictionData(Copyable, Movable):
    """`hdbscan.hpp:369-435`, host lists."""

    var n_leaves: Int
    var n_edges: Int
    var n_clusters: Int
    var n_selected_clusters: Int
    var n_exemplars: Int
    var deaths: List[Float32]
    var exemplar_idx: List[Int32]
    var exemplar_label_offsets: List[Int32]
    var selected_clusters: List[Int32]
    var index_into_children: List[Int32]

    def __init__(
        out self,
        n_leaves: Int,
        n_edges: Int,
        n_clusters: Int,
        n_selected_clusters: Int,
        n_exemplars: Int,
        var deaths: List[Float32],
        var exemplar_idx: List[Int32],
        var exemplar_label_offsets: List[Int32],
        var selected_clusters: List[Int32],
        var index_into_children: List[Int32],
    ):
        self.n_leaves = n_leaves
        self.n_edges = n_edges
        self.n_clusters = n_clusters
        self.n_selected_clusters = n_selected_clusters
        self.n_exemplars = n_exemplars
        self.deaths = deaths^
        self.exemplar_idx = exemplar_idx^
        self.exemplar_label_offsets = exemplar_label_offsets^
        self.selected_clusters = selected_clusters^
        self.index_into_children = index_into_children^


def generate_prediction_data(
    parents: List[Int32],
    children: List[Int32],
    lambdas: List[Float32],
    sizes: List[Int32],
    n_edges: Int,
    n_leaves: Int,
    n_clusters: Int,
    labels: List[Int32],
    inverse_label_map: List[Int32],
    n_selected_clusters: Int,
) raises -> PredictionData:
    """`prediction_data.cu:92-239`. `labels` are the FINAL labels (`0 ..
    n_selected-1` or -1), `inverse_label_map` maps a final label to its
    condensed cluster id counted from 0 (their `inverse_label_map`)."""
    if n_edges < 1 or len(parents) < n_edges or len(children) < n_edges:
        raise Error(
            "hdbscan.generate_prediction_data: the condensed tree has "
            + String(n_edges) + " edges but the arrays hold fewer; refused"
            " by name"
        )
    if len(labels) < n_leaves or len(inverse_label_map) < n_selected_clusters:
        raise Error(
            "hdbscan.generate_prediction_data: labels or inverse_label_map is"
            " shorter than n_leaves / n_selected_clusters; refused by name"
        )
    # `:112-145` the death of each cluster: parent CSR + segmented Max.
    # Their segments are the parent-sorted edges; a max over each segment
    # is the same value visited in any order.
    var deaths = List[Float32](length=n_clusters, fill=PD_FLOAT32_LOWEST)
    for e in range(n_edges):
        var p = Int(parents[e]) - n_leaves
        if p < 0 or p >= n_clusters:
            raise Error(
                "hdbscan.generate_prediction_data: parent "
                + String(Int(parents[e])) + " at edge " + String(e)
                + " is outside the condensed cluster ids; refused by name"
            )
        if deaths[p] < lambdas[e]:
            deaths[p] = lambdas[e]

    # `:147-156` is_leaf_cluster: a cluster with a cluster child is not a leaf.
    var is_leaf_cluster = List[Int32](length=n_clusters, fill=Int32(1))
    for e in range(n_edges):
        if sizes[e] > Int32(1):
            is_leaf_cluster[Int(parents[e]) - n_leaves] = Int32(0)

    # `:162-179` exemplar_op
    var is_exemplar = List[Int32](length=n_leaves, fill=Int32(0))
    for e in range(n_edges):
        var c = Int(children[e])
        if c < n_leaves:
            var p = Int(parents[e]) - n_leaves
            var ex = (
                labels[c] != Int32(-1)
                and is_leaf_cluster[p] != Int32(0)
                and lambdas[e] == deaths[p]
            )
            is_exemplar[c] = Int32(1) if ex else Int32(0)

    # `:181-182` count_if
    var n_exemplars = 0
    for i in range(n_leaves):
        if is_exemplar[i] != Int32(0):
            n_exemplars += 1

    # `:184` allocate
    var exemplar_idx = List[Int32](capacity=n_exemplars)
    var exemplar_label_offsets = List[Int32](
        length=n_selected_clusters + 1, fill=Int32(0)
    )
    var selected_clusters = List[Int32](
        length=n_selected_clusters, fill=Int32(0)
    )
    var index_into_children = List[Int32](length=n_edges + 1, fill=Int32(-1))

    # `:186-191` copy_if over the counting iterator (ascending, stable)
    for i in range(n_leaves):
        if is_exemplar[i] != Int32(0):
            exemplar_idx.append(Int32(i))

    # `:194-206` exemplar_labels through the inverse label map
    var exemplar_labels = List[Int32](capacity=n_exemplars)
    for j in range(n_exemplars):
        var label = labels[Int(exemplar_idx[j])]
        if label != Int32(-1):
            exemplar_labels.append(inverse_label_map[Int(label)])
        else:
            exemplar_labels.append(Int32(-1))

    # `:208-211` sort_by_key(exemplar_labels, exemplar_idx). DEVIATION 1612.
    if n_exemplars > 0:
        var keys = List[UInt64](capacity=n_exemplars)
        var order = List[Int](capacity=n_exemplars)
        for j in range(n_exemplars):
            var lab = UInt64(Int(exemplar_labels[j])) & UInt64(0xFFFFFFFF)
            keys.append((lab << UInt64(32)) | UInt64(Int(exemplar_idx[j])))
            order.append(j)
        merge_sort_u64_with_index(keys, order)
        var s_idx = List[Int32](capacity=n_exemplars)
        var s_lab = List[Int32](capacity=n_exemplars)
        for j in range(n_exemplars):
            s_idx.append(exemplar_idx[order[j]])
            s_lab.append(exemplar_labels[order[j]])
        exemplar_idx = s_idx^
        exemplar_labels = s_lab^

    # `:214-219` converted (final) labels of the sorted exemplars
    # `:221-238`, only when there is an exemplar
    if n_exemplars > 0:
        # raft sorted_coo_to_csr (sparse/convert/detail/csr.cuh:78-90):
        # row counts over m = n_selected + 1 rows, then an exclusive scan.
        var counts = List[Int32](length=n_selected_clusters + 1, fill=Int32(0))
        for j in range(n_exemplars):
            var cl = Int(labels[Int(exemplar_idx[j])])
            if cl < 0 or cl > n_selected_clusters:
                raise Error(
                    "hdbscan.generate_prediction_data: exemplar label "
                    + String(cl) + " is outside [0, "
                    + String(n_selected_clusters) + "]; refused by name"
                )
            counts[cl] += Int32(1)
        var acc = Int32(0)
        for c in range(n_selected_clusters + 1):
            exemplar_label_offsets[c] = acc
            acc += counts[c]
        # `:228-234` selected_clusters[c] = exemplar_labels[offsets[c]] + n_leaves
        for c in range(n_selected_clusters):
            var off = Int(exemplar_label_offsets[c])
            if off >= n_exemplars:
                raise Error(
                    "hdbscan.generate_prediction_data: selected cluster "
                    + String(c) + " has no exemplar, and their transform at"
                    " prediction_data.cu:228-234 would read past"
                    " exemplar_labels; refused by name"
                )
            selected_clusters[c] = exemplar_labels[off] + Int32(n_leaves)
        # `:236-237` build_index_into_children (`:60-79`)
        for e in range(n_edges):
            var ch = Int(children[e])
            if ch < 0 or ch > n_edges:
                raise Error(
                    "hdbscan.generate_prediction_data: child "
                    + String(ch) + " is outside [0, n_edges]; refused by name"
                )
            index_into_children[ch] = Int32(e)
    return PredictionData(
        n_leaves, n_edges, n_clusters, n_selected_clusters, n_exemplars,
        deaths^, exemplar_idx^, exemplar_label_offsets^,
        selected_clusters^, index_into_children^,
    )


comptime PD_ERR_PARENT = 1
comptime PD_ERR_LABEL = 2
comptime PD_ERR_NO_EXEMPLAR = 3
comptime PD_ERR_CHILD = 4


def pd_deaths_kernel(
    parents: I32P,
    lambdas: F32P,
    sizes: I32P,
    dkey: I32P,
    is_leaf: I32P,
    err: I32P,
    n_leaves: Int32,
    n_edges: Int32,
    n_clusters: Int32,
):
    """`:112-156` per edge: the death as an integer `Atomic.max` on
    `weight_order_key`, and `is_leaf_cluster` (a constant write)."""
    var e = _gid()
    if e >= Int(n_edges):
        return
    var p = Int(parents[e]) - Int(n_leaves)
    if p < 0 or p >= Int(n_clusters):
        _ = Atomic.max(err, Int32(PD_ERR_PARENT))
        return
    _ = Atomic.max(dkey.unsafe_offset(p), weight_order_key(lambdas[e]))
    if sizes[e] > Int32(1):
        is_leaf[p] = 0


def pd_unkey_kernel(dkey: I32P, deaths: F32P, n: Int32):
    var c = _gid()
    if c >= Int(n):
        return
    deaths[c] = weight_order_unkey(dkey[c])


def pd_exemplar_kernel(
    parents: I32P,
    children: I32P,
    lambdas: F32P,
    labels: I32P,
    is_leaf: I32P,
    deaths: F32P,
    flag: I32P,
    n_leaves: Int32,
    n_edges: Int32,
):
    """`:162-179` exemplar_op, one thread per edge (each leaf is a child
    once)."""
    var e = _gid()
    if e >= Int(n_edges):
        return
    var c = Int(children[e])
    if c < 0 or c >= Int(n_leaves):
        return
    var p = Int(parents[e]) - Int(n_leaves)
    var ex = (
        labels[c] != Int32(-1)
        and is_leaf[p] != Int32(0)
        and lambdas[e] == deaths[p]
    )
    flag[c] = Int32(1) if ex else Int32(0)


def pd_compact_kernel(
    off: I32P,
    labels: I32P,
    inv: I32P,
    keys: U32P,
    vals: U32P,
    n_leaves: Int32,
):
    """`:186-206` copy_if (ascending) and the exemplar's condensed label."""
    var i = _gid()
    if i >= Int(n_leaves):
        return
    var o = Int(off[i])
    if Int(off[i + 1]) == o:
        return
    var label = Int(labels[i])
    keys[o] = UInt32(Int(inv[label])) if label != -1 else UInt32(0xFFFFFFFF)
    vals[o] = UInt32(i)


def pd_count_kernel(
    vals: U32P, labels: I32P, counts: I32P, err: I32P, n_ex: Int32, n_sel: Int32
):
    var j = _gid()
    if j >= Int(n_ex):
        return
    var cl = Int(labels[Int(vals[j])])
    if cl < 0 or cl > Int(n_sel):
        _ = Atomic.max(err, Int32(PD_ERR_LABEL))
        return
    _ = Atomic.fetch_add(counts.unsafe_offset(cl), Int32(1))


def pd_selected_kernel(
    keys: U32P, offsets: I32P, selected: I32P, err: I32P,
    n_leaves: Int32, n_ex: Int32, n_sel: Int32,
):
    """`:228-234` selected_clusters[c] = exemplar_labels[offsets[c]] +
    n_leaves."""
    var c = _gid()
    if c >= Int(n_sel):
        return
    var o = Int(offsets[c])
    if o >= Int(n_ex):
        _ = Atomic.max(err, Int32(PD_ERR_NO_EXEMPLAR))
        return
    selected[c] = Int32(Int(keys[o]) + Int(n_leaves))


def pd_children_index_kernel(
    children: I32P, iic: I32P, err: I32P, n_edges: Int32
):
    """`:60-79` build_index_into_children."""
    var e = _gid()
    if e >= Int(n_edges):
        return
    var ch = Int(children[e])
    if ch < 0 or ch > Int(n_edges):
        _ = Atomic.max(err, Int32(PD_ERR_CHILD))
        return
    iic[ch] = Int32(e)


def generate_prediction_data_device(
    ctx: DeviceContext,
    parents: List[Int32],
    children: List[Int32],
    lambdas: List[Float32],
    sizes: List[Int32],
    n_edges: Int,
    n_leaves: Int,
    n_clusters: Int,
    labels: List[Int32],
    inverse_label_map: List[Int32],
    n_selected_clusters: Int,
) raises -> PredictionData:
    """`generate_prediction_data` on the device (DEVIATION 1612): the same
    passes as grid-wide launches, the exemplar order by one stable radix
    sort keyed by condensed label over the ascending exemplar indices (the
    packed-key order of the host function). Inputs are uploaded once and
    the five outputs downloaded once."""
    if n_edges < 1 or len(parents) < n_edges or len(children) < n_edges:
        raise Error(
            "hdbscan.generate_prediction_data: the condensed tree has "
            + String(n_edges) + " edges but the arrays hold fewer; refused"
            " by name"
        )
    if len(labels) < n_leaves or len(inverse_label_map) < n_selected_clusters:
        raise Error(
            "hdbscan.generate_prediction_data: labels or inverse_label_map is"
            " shorter than n_leaves / n_selected_clusters; refused by name"
        )
    var d_par = td_upload_i32(ctx, parents, n_edges)
    var d_ch = td_upload_i32(ctx, children, n_edges)
    var d_lam = td_upload_f32(ctx, lambdas, n_edges)
    var d_sz = td_upload_i32(ctx, sizes, n_edges)
    var d_lab = td_upload_i32(ctx, labels, n_leaves)
    var d_inv = td_upload_i32(ctx, inverse_label_map, n_selected_clusters)
    var err = ctx.enqueue_create_buffer[DType.int32](1)
    ctx.enqueue_memset(err, Int32(0))
    var dkey = ctx.enqueue_create_buffer[DType.int32](n_clusters)
    ctx.enqueue_memset(dkey, weight_order_key(PD_FLOAT32_LOWEST))
    var is_leaf = ctx.enqueue_create_buffer[DType.int32](n_clusters)
    ctx.enqueue_memset(is_leaf, Int32(1))
    ctx.enqueue_function[pd_deaths_kernel](
        d_par.unsafe_ptr(), d_lam.unsafe_ptr(), d_sz.unsafe_ptr(),
        dkey.unsafe_ptr(), is_leaf.unsafe_ptr(), err.unsafe_ptr(),
        Int32(n_leaves), Int32(n_edges), Int32(n_clusters),
        grid_dim=(td_grid(n_edges), 1, 1), block_dim=(TD_TPB, 1, 1),
    )
    var deaths = ctx.enqueue_create_buffer[DType.float32](n_clusters)
    ctx.enqueue_function[pd_unkey_kernel](
        dkey.unsafe_ptr(), deaths.unsafe_ptr(), Int32(n_clusters),
        grid_dim=(td_grid(n_clusters), 1, 1), block_dim=(TD_TPB, 1, 1),
    )
    if td_read_i32(ctx, err, 0) == PD_ERR_PARENT:
        raise Error(
            "hdbscan.generate_prediction_data: a parent is outside the"
            " condensed cluster ids; refused by name"
        )
    var flag = ctx.enqueue_create_buffer[DType.int32](n_leaves + 1)
    ctx.enqueue_memset(flag, Int32(0))
    ctx.enqueue_function[pd_exemplar_kernel](
        d_par.unsafe_ptr(), d_ch.unsafe_ptr(), d_lam.unsafe_ptr(),
        d_lab.unsafe_ptr(), is_leaf.unsafe_ptr(), deaths.unsafe_ptr(),
        flag.unsafe_ptr(), Int32(n_leaves), Int32(n_edges),
        grid_dim=(td_grid(n_edges), 1, 1), block_dim=(TD_TPB, 1, 1),
    )
    td_exclusive_scan(ctx, flag, n_leaves + 1)
    var n_exemplars = td_read_i32(ctx, flag, n_leaves)

    var exemplar_idx = List[Int32]()
    var exemplar_label_offsets = List[Int32](
        length=n_selected_clusters + 1, fill=Int32(0)
    )
    var selected_clusters = List[Int32](
        length=n_selected_clusters, fill=Int32(0)
    )
    var index_into_children = List[Int32](length=n_edges + 1, fill=Int32(-1))
    if n_exemplars > 0:
        var keys = ctx.enqueue_create_buffer[DType.uint32](n_exemplars)
        var vals = ctx.enqueue_create_buffer[DType.uint32](n_exemplars)
        ctx.enqueue_function[pd_compact_kernel](
            flag.unsafe_ptr(), d_lab.unsafe_ptr(), d_inv.unsafe_ptr(),
            keys.unsafe_ptr(), vals.unsafe_ptr(), Int32(n_leaves),
            grid_dim=(td_grid(n_leaves), 1, 1), block_dim=(TD_TPB, 1, 1),
        )
        # `:208-211` sort_by_key, stable over ascending indices.
        td_sort_pairs(ctx, keys, vals, n_exemplars)
        var counts = ctx.enqueue_create_buffer[DType.int32](
            n_selected_clusters + 1
        )
        ctx.enqueue_memset(counts, Int32(0))
        ctx.enqueue_function[pd_count_kernel](
            vals.unsafe_ptr(), d_lab.unsafe_ptr(), counts.unsafe_ptr(),
            err.unsafe_ptr(), Int32(n_exemplars), Int32(n_selected_clusters),
            grid_dim=(td_grid(n_exemplars), 1, 1), block_dim=(TD_TPB, 1, 1),
        )
        td_exclusive_scan(ctx, counts, n_selected_clusters + 1)
        var sel = ctx.enqueue_create_buffer[DType.int32](
            max(n_selected_clusters, 1)
        )
        ctx.enqueue_memset(sel, Int32(0))
        if n_selected_clusters > 0:
            ctx.enqueue_function[pd_selected_kernel](
                keys.unsafe_ptr(), counts.unsafe_ptr(), sel.unsafe_ptr(),
                err.unsafe_ptr(), Int32(n_leaves), Int32(n_exemplars),
                Int32(n_selected_clusters),
                grid_dim=(td_grid(n_selected_clusters), 1, 1),
                block_dim=(TD_TPB, 1, 1),
            )
        var iic = ctx.enqueue_create_buffer[DType.int32](n_edges + 1)
        ctx.enqueue_memset(iic, Int32(-1))
        ctx.enqueue_function[pd_children_index_kernel](
            d_ch.unsafe_ptr(), iic.unsafe_ptr(), err.unsafe_ptr(),
            Int32(n_edges),
            grid_dim=(td_grid(n_edges), 1, 1), block_dim=(TD_TPB, 1, 1),
        )
        var code = td_read_i32(ctx, err, 0)
        if code == PD_ERR_LABEL:
            raise Error(
                "hdbscan.generate_prediction_data: an exemplar label is"
                " outside [0, " + String(n_selected_clusters) + "]; refused"
                " by name"
            )
        if code == PD_ERR_NO_EXEMPLAR:
            raise Error(
                "hdbscan.generate_prediction_data: a selected cluster has no"
                " exemplar, and their transform at prediction_data.cu:228-234"
                " would read past exemplar_labels; refused by name"
            )
        if code == PD_ERR_CHILD:
            raise Error(
                "hdbscan.generate_prediction_data: a child is outside [0,"
                " n_edges]; refused by name"
            )
        var h_vals = ctx.enqueue_create_host_buffer[DType.uint32](n_exemplars)
        ctx.enqueue_copy(dst_ptr=h_vals.unsafe_ptr(), src_buf=vals)
        ctx.synchronize()
        exemplar_idx = List[Int32](capacity=n_exemplars)
        for j in range(n_exemplars):
            exemplar_idx.append(Int32(Int(h_vals.unsafe_ptr().unsafe_load(j))))
        _ = h_vals^
        exemplar_label_offsets = td_download_i32(
            ctx, counts, n_selected_clusters + 1
        )
        selected_clusters = td_download_i32(ctx, sel, n_selected_clusters)
        index_into_children = td_download_i32(ctx, iic, n_edges + 1)
        _ = keys^
        _ = vals^
        _ = counts^
        _ = sel^
        _ = iic^
    var h_deaths = td_download_f32(ctx, deaths, n_clusters)
    _ = d_par^
    _ = d_ch^
    _ = d_lam^
    _ = d_sz^
    _ = d_lab^
    _ = d_inv^
    _ = err^
    _ = dkey^
    _ = is_leaf^
    _ = deaths^
    _ = flag^
    return PredictionData(
        n_leaves, n_edges, n_clusters, n_selected_clusters, n_exemplars,
        h_deaths^, exemplar_idx^, exemplar_label_offsets^,
        selected_clusters^, index_into_children^,
    )


def prediction_neighborhood(min_samples: Int, n_rows: Int) raises -> Int:
    """`predict.cuh:161`, `neighborhood = (min_samples - 1) * 2`, with the
    two values theirs does not guard refused by name: below 2 the
    neighborhood is empty (their k-NN at k = 0), and above `n_rows` the
    brute-force k-NN has fewer points than neighbors."""
    if min_samples < 2:
        raise Error(
            "hdbscan.approximate_predict: min_samples=" + String(min_samples)
            + " refused by name; the prediction neighborhood (min_samples - 1)"
            " * 2 (predict.cuh:161) is empty below 2"
        )
    var k = (min_samples - 1) * 2
    if k > n_rows:
        raise Error(
            "hdbscan.approximate_predict: the prediction neighborhood "
            + String(k) + " = (min_samples - 1) * 2 exceeds the "
            + String(n_rows) + " training rows; refused by name"
        )
    return k


def refuse_soft_clustering_inputs(
    parents: List[Int32],
    lambdas: List[Float32],
    pd: PredictionData,
    m: Int,
    where: String,
) raises:
    """The fitted state `membership_vector` and
    `all_points_membership_vectors` read, checked before any kernel reads
    it through an index (`hdbscan/impl/detail/soft_clustering.mojo`). The
    arrays arrive from Python, so every index a kernel follows is in range,
    every selected cluster has an exemplar (their `reduction_op` would
    return FLT_MAX for an empty range, `soft_clustering.cuh:111-115`), and
    every cluster's parent is a smaller id, which is what makes the merge
    height walk (`kernels/soft_clustering.cuh:34-44`) reach the root.
    Refused by name."""
    from std.math import isfinite

    var nl = pd.n_leaves
    var ne = pd.n_edges
    var nc = pd.n_clusters
    var ns = pd.n_selected_clusters
    var nx = pd.n_exemplars
    if (
        nl != m
        or nc < 1
        or ne != nl + nc - 1
        or len(parents) < ne
        or len(lambdas) < ne
        or len(pd.index_into_children) < ne + 1
        or len(pd.deaths) < nc
    ):
        raise Error(
            where + ": the condensed tree does not have one edge per point and"
            " per non-root cluster (n_leaves=" + String(nl) + ", n_edges="
            + String(ne) + ", n_clusters=" + String(nc) + ", n_rows="
            + String(m) + "); refused by name"
        )
    if (
        ns < 1
        or len(pd.selected_clusters) < ns
        or len(pd.exemplar_label_offsets) < ns + 1
        or nx < ns
        or len(pd.exemplar_idx) < nx
    ):
        raise Error(
            where + ": the prediction data has " + String(ns)
            + " selected clusters and " + String(nx)
            + " exemplars; refused by name"
        )
    if Int(pd.exemplar_label_offsets[0]) != 0 or Int(
        pd.exemplar_label_offsets[ns]
    ) != nx:
        raise Error(
            where + ": exemplar_label_offsets does not span the exemplars;"
            " refused by name"
        )
    for c in range(ns):
        if pd.exemplar_label_offsets[c + 1] <= pd.exemplar_label_offsets[c]:
            raise Error(
                where + ": selected cluster " + String(c)
                + " has no exemplar; refused by name"
            )
        var s = Int(pd.selected_clusters[c])
        if s < nl or s >= nl + nc:
            raise Error(
                where + ": selected cluster " + String(c) + " is node "
                + String(s) + ", outside the condensed clusters; refused by"
                " name"
            )
    for j in range(nx):
        var r = Int(pd.exemplar_idx[j])
        if r < 0 or r >= m:
            raise Error(
                where + ": exemplar " + String(j) + " is row " + String(r)
                + ", outside the training rows; refused by name"
            )
    for e in range(ne):
        var p = Int(parents[e])
        if p < nl or p >= nl + nc:
            raise Error(
                where + ": parent " + String(p) + " at edge " + String(e)
                + " is outside the condensed clusters; refused by name"
            )
        if not isfinite(lambdas[e]):
            raise Error(
                where + ": condensed lambda " + String(e) + " is not finite;"
                " refused by name (DEVIATION 1607)"
            )
    for c in range(nc):
        if not isfinite(pd.deaths[c]):
            raise Error(
                where + ": death " + String(c) + " is not finite; refused by"
                " name (DEVIATION 1607)"
            )
    for x in range(ne + 1):
        if x == nl:
            continue
        var e = Int(pd.index_into_children[x])
        if e < 0 or e >= ne:
            raise Error(
                where + ": index_into_children[" + String(x) + "] = "
                + String(e) + " is outside the edges; refused by name"
            )
        if x > nl and Int(parents[e]) >= x:
            raise Error(
                where + ": cluster " + String(x) + " has parent "
                + String(Int(parents[e]))
                + ", not a smaller id, so the merge height walk would not"
                " reach the root; refused by name"
            )


def refuse_nonfinite_queries(q: List[Float32], n_q: Int, d: Int) raises:
    """DEVIATION 1607 at the query boundary: a NaN or infinite query cell
    would carry the vendor's NaN payload into a distance. Refused by name."""
    from std.math import isfinite

    for i in range(n_q * d):
        if not isfinite(q[i]):
            raise Error(
                "hdbscan.approximate_predict: points_to_predict has a NaN or"
                " infinite value at row " + String(i // d) + ", column "
                + String(i % d) + "; refused by name (DEVIATION 1607)"
            )
