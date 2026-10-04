# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Cluster extraction: stabilities, selection, and the point labels.

Reference: `cuml-v26.08.00/cpp/src/hdbscan/detail/extract.cuh`
(cuML `265b9da`): `TreeUnionFind` (`:49-86`), `do_labelling_on_host`
(`:88-167`) and `extract_clusters` (`:246-314`). `_compute_inverse_label_
map` (`:172-221`) is the CPU/GPU-interop half of `extract_clusters` and is
not reached by a fit; `hdbscan/NOT_IMPLEMENTED.tsv` has the row.
Steps run in the reference order.

PROBABILITIES. Their `:311` runs `Membership::get_probabilities`
(`detail/membership.cuh`); here it is `get_probabilities_host` below
(DEVIATION 5116), host code the bindings call on the fit's output.

THE LABELLING RUNS ON THE DEVICE (`do_labelling_device`, lane c-cluster,
2026-10-02). Theirs (`:88-167`) is a host union-find over the condensed
edges: `perform_union(parent, child)` for every edge whose child is not a
selected cluster, then `find(i)` per point. The condensed tree is sorted by
(parent, child) and every child id is larger than its parent's, so the
unions run in ascending-parent order: when edge (p, c) is unioned, `c` is
still a fresh singleton of rank 0 and `p`'s component has rank >= 1 or is
`p` alone (rank 0, ties keep `x_root = p`). The representative of every
component is therefore its TOPMOST node: walking up from a point through
parents, the first node that is a selected cluster or the root. That walk
is what the device computes, with no union-find:

    next[v] = v                 v selected, the root, or never a child
    next[v] = parent(v)         otherwise
    next = next[next]           ceil(log2(n_nodes)) rounds, ping-pong

Each round is one grid-wide launch over the nodes reading one buffer and
writing the other, so no thread reads a value another thread of the same
round writes, and the fixed round count reaches every chain's end (a chain
is at most `n_nodes` long). The result is integers, so there is no fold
order; `parent_lambdas[root]` is a max taken on `weight_order_key` by
`Atomic.max`, an order-free integer max. The labels equal the host walk's
(`hdbscan/host/labelling_host.mojo::do_labelling_on_host`, the CPU
column's, which keeps DEVIATION 1609) for every tree.

THE CLUSTER SET IS A SORTED SET ON BOTH SIDES. Theirs is
`std::set<value_idx>` (`:281-284`), so iterating it yields ASCENDING
cluster ids and `label_map_h[cluster - n_leaves] = i` numbers the final
labels by ascending condensed id (`:291-295`). Ours builds the same
ascending list by scanning `is_cluster` from 0 upward, which is the same
order without a container. That numbering IS the labels a caller sees, so
it is part of the answer and not a formatting choice.
"""

from std.atomic import Atomic
from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from hdbscan.checks.hdbscan_sabotage import HDB_SAB_NONE
from hdbscan.impl.condensed_hierarchy import CondensedHierarchy
from hdbscan.impl.detail.select import select_clusters
from hdbscan.impl.detail.stabilities import (
    compute_stabilities,
    get_stability_scores_device,
)
from hdbscan.impl.detail.fast_apple import HDB_ONE_SYNC, HDB_SELECT_DEVICE
from hdbscan.impl.detail.idn_switches import (
    IDN_HDB_ONE_SYNC,
    IDN_HDB_SELECT_ONE_READ,
)
from hdbscan.checks.hdbscan_sabotage import HDB_SAB_EOM_NO_UPDATE
from hdbscan.impl.detail.select import (
    CLUSTER_SELECTION_EOM,
    CLUSTER_SELECTION_LEAF,
    eom_init_kernel,
    eom_level_kernel,
    flag_kernel,
    label_map_kernel,
    leaf_kernel,
    negate_apply_kernel,
    negate_init_kernel,
)
from hdbscan.impl.detail.stabilities import (
    STAB_FOLD,
    STAB_TPB,
    births_init_kernel,
    cluster_stability_kernel,
    cluster_count_kernel,
    max_lambda_key_kernel,
    stability_score_kernel,
)
from hdbscan.impl.detail.tree_device import (
    TD_TPB,
    td_exclusive_scan,
    td_grid,
    td_path_reduce,
)
from hdbscan.impl.detail.tree_device import (
    DeviceTree,
    td_download_f32,
    td_download_i32,
    td_find,
    td_read_i32,
    td_stage_f32,
    td_stage_i32,
    td_take_f32,
    td_take_i32,
)
from hierarchy.checks.edge_order import weight_order_key, weight_order_unkey
from std.math import isinf, isnan

from checks.numerics import ftz, identical_div


@fieldwise_init
struct ExtractOutput(Copyable, Movable):
    """What `extract_clusters` hands back, every array computed on the
    device and downloaded once at the end. Theirs writes through
    out-pointers and returns `clusters.size()` (`:313`)."""

    var n_selected: Int
    var n_outliers: Int
    var labels: List[Int32]
    """`n_leaves`, CONDENSED cluster ids (or -1), before the label_map
    remap their `runner.h:226-233` applies."""
    var final_labels: List[Int32]
    """`n_leaves`, after the remap: `0 .. n_selected-1` or -1."""
    var is_cluster: List[Int32]
    """`n_clusters`, the selection."""
    var tree_stabilities: List[Float32]
    """`n_clusters`, AFTER `excess_of_mass` mutated it (DEVIATION 1605)."""
    var label_map: List[Int32]
    """`n_clusters`, condensed id -> final label, -1 where unselected."""
    var inverse_label_map: List[Int32]
    """`n_selected`, final label -> condensed id."""
    var stability_scores: List[Float32]
    """`n_selected`, `get_stability_scores`' normalized output."""
    var probabilities: List[Float32]
    """`n_leaves`, `Membership::get_probabilities` (DEVIATION 5116)."""


def get_probabilities_host(
    tree: CondensedHierarchy, raw_labels: List[Int32], n_leaves: Int
) raises -> List[Float32]:
    """`membership.cuh:39-98` `Membership::get_probabilities` and
    `kernels/membership.cuh::probabilities_functor`, DEVIATION 5116 (the
    cluster lane): HOST code both routes call on the condensed tree and the
    cluster-space labels `do_labelling_on_host` returned (`extract.cuh:311`).

    `deaths[c]` is the max lambda over cluster `c`'s children (their
    `cub::DeviceSegmentedReduce::Max` over `Utils::parent_csr`): a float
    max compared by `weight_order_key`, the order `parent_lambdas` uses
    above, so no fold order or NaN rule can move it. Per point edge:
    noise stays 0; `death == 0` or a non-finite lambda gives 1 (theirs
    tests `isnan`, scikit-learn's `get_probabilities` tests `isinf`; both
    are taken, since `inf / inf` would be a NaN nobody asked for); else
    `min(lambda, death) / death` by `identical_div`.
    """
    var n_clusters = tree.n_clusters
    var deaths = List[Float32](length=max(n_clusters, 1), fill=Float32(0.0))
    var seen = List[Bool](length=max(n_clusters, 1), fill=False)
    for i in range(tree.n_edges):
        var c = Int(tree.parents[i]) - n_leaves
        if c < 0 or c >= n_clusters:
            raise Error(
                "hdbscan.get_probabilities: parent " + String(tree.parents[i])
                + " at edge " + String(i) + " is outside the cluster range"
            )
        var lam = tree.lambdas[i]
        if not seen[c] or weight_order_key(lam) > weight_order_key(deaths[c]):
            deaths[c] = lam
            seen[c] = True
    var out = List[Float32](length=n_leaves, fill=Float32(0.0))
    for i in range(tree.n_edges):
        var child = Int(tree.children[i])
        if child >= n_leaves:
            continue
        var cluster = Int(raw_labels[child])
        if cluster == -1:
            continue
        var death = deaths[cluster]
        var lam = tree.lambdas[i]
        if death == Float32(0.0) or isnan(lam) or isinf(lam):
            out[child] = Float32(1.0)
        else:
            var lo = lam if lam < death else death
            out[child] = ftz(identical_div(lo, death))
    return out^


def probabilities_from_labels(
    tree: CondensedHierarchy,
    labels: List[Int32],
    inverse_label_map: List[Int32],
    n_leaves: Int,
) raises -> List[Float32]:
    """`get_probabilities_host` from the FINAL labels: each is mapped back
    to its condensed cluster through `inverse_label_map` (the inverse of
    `runner.h:226-233`'s remap), the cluster-space labels theirs reads.
    Both bindings call this on their fit's output."""
    var raw = List[Int32](capacity=n_leaves)
    for i in range(n_leaves):
        var l = Int(labels[i])
        raw.append(Int32(-1) if l < 0 else inverse_label_map[l])
    return get_probabilities_host(tree, raw, n_leaves)


comptime LABEL_TPB = 256
"""Threads per block of the labelling launches. SCHEDULING only: every
launch writes integers whose values no thread order can move."""


def label_init_kernel(
    nxt: MutPointer[Int32, MutAnyOrigin],
    child_lambda: MutPointer[Float32, MutAnyOrigin],
    n_nodes: Int32,
):
    """`next[v] = v`, `child_lambda[v] = 0` for every node."""
    var v = Int(block_dim.x) * Int(block_idx.x) + Int(thread_idx.x)
    if v >= Int(n_nodes):
        return
    nxt.unsafe_store(v, Int32(v))
    child_lambda.unsafe_store(v, Float32(0.0))


def label_edges_kernel(
    parents: MutPointer[Int32, MutAnyOrigin],
    children: MutPointer[Int32, MutAnyOrigin],
    lambdas: MutPointer[Float32, MutAnyOrigin],
    is_cluster: MutPointer[Int32, MutAnyOrigin],
    nxt: MutPointer[Int32, MutAnyOrigin],
    child_lambda: MutPointer[Float32, MutAnyOrigin],
    root_key: MutPointer[Int32, MutAnyOrigin],
    n_leaves_in: Int32,
    n_nodes_in: Int32,
    n_edges_in: Int32,
):
    """`:122-129` per edge. A child that is not a selected cluster points at
    its parent (their `perform_union(parent, child)`); every child records
    its edge's lambda (the `std::find` at `:144`, each child has one edge);
    an edge of the root folds its lambda into `parent_lambdas[root]` by
    `Atomic.max` on `weight_order_key` (`:128`, an integer max, so the
    order threads land in cannot move it). Each child slot is written by
    its one edge only."""
    var e = Int(block_dim.x) * Int(block_idx.x) + Int(thread_idx.x)
    if e >= Int(n_edges_in):
        return
    var n_leaves = Int(n_leaves_in)
    var child = Int(children.unsafe_load(e))
    var parent = Int(parents.unsafe_load(e))
    var lam = lambdas.unsafe_load(e)
    if child < 0 or child >= Int(n_nodes_in):
        return
    var selected = child >= n_leaves and is_cluster.unsafe_load(
        child - n_leaves
    ) != Int32(0)
    if not selected:
        nxt.unsafe_store(child, Int32(parent))
    child_lambda.unsafe_store(child, lam)
    if parent == n_leaves:
        _ = Atomic.max(root_key, weight_order_key(lam))


def label_jump_kernel(
    src: MutPointer[Int32, MutAnyOrigin],
    dst: MutPointer[Int32, MutAnyOrigin],
    n_nodes: Int32,
):
    """One pointer-jumping round, `dst[v] = src[src[v]]`. Reads `src` only
    and writes `dst` only."""
    var v = Int(block_dim.x) * Int(block_idx.x) + Int(thread_idx.x)
    if v >= Int(n_nodes):
        return
    dst.unsafe_store(v, src.unsafe_load(Int(src.unsafe_load(v))))


def label_points_kernel(
    rep: MutPointer[Int32, MutAnyOrigin],
    child_lambda: MutPointer[Float32, MutAnyOrigin],
    root_key: MutPointer[Int32, MutAnyOrigin],
    labels: MutPointer[Int32, MutAnyOrigin],
    n_leaves_in: Int32,
    root_single: Int32,
    use_epsilon: Int32,
    inverse_cluster_selection_epsilon: Float32,
):
    """`:136-164` per point, with `cluster = rep[i]` (their `find(i)`)."""
    var i = Int(block_dim.x) * Int(block_idx.x) + Int(thread_idx.x)
    var n_leaves = Int(n_leaves_in)
    if i >= n_leaves:
        return
    var cluster = Int(rep.unsafe_load(i))
    var out = Int32(-1)
    if cluster > n_leaves:
        out = Int32(cluster - n_leaves)
    elif cluster == n_leaves and root_single != Int32(0):
        # `:141-160`: the root is a label only when it is the one selected
        # cluster and allow_single_cluster is set.
        var lam = child_lambda.unsafe_load(i)
        if use_epsilon != Int32(0):
            if lam >= inverse_cluster_selection_epsilon:
                out = Int32(0)
        elif lam >= weight_order_unkey(root_key.unsafe_load(0)):
            out = Int32(0)
    labels.unsafe_store(i, out)


def do_labelling_device(
    ctx: DeviceContext,
    mut tree: DeviceTree,
    mut is_cluster: DeviceBuffer[DType.int32],
    n_selected: Int,
    allow_single_cluster: Bool,
    cluster_selection_epsilon: Float32,
    tpb: Int = LABEL_TPB,
) raises -> DeviceBuffer[DType.int32]:
    """`extract.cuh:88-167` on the device (this file's header), over the
    device tree. Returns the `n_leaves` condensed cluster ids (or -1), the
    host walk's integers, still on the device."""
    var n_leaves = tree.n_leaves
    var n_edges = tree.n_edges
    var n_clusters = tree.n_clusters
    var n_nodes = n_leaves + n_clusters
    var next_a = ctx.enqueue_create_buffer[DType.int32](n_nodes)
    var next_b = ctx.enqueue_create_buffer[DType.int32](n_nodes)
    var child_lambda = ctx.enqueue_create_buffer[DType.float32](n_nodes)
    var root_key = ctx.enqueue_create_buffer[DType.int32](1)
    var d_labels = ctx.enqueue_create_buffer[DType.int32](max(n_leaves, 1))
    # `:117` parent_lambdas starts at 0.0f.
    ctx.enqueue_memset(root_key, weight_order_key(Float32(0.0)))

    var node_grid = max(1, (n_nodes + tpb - 1) // tpb)
    ctx.enqueue_function[label_init_kernel](
        next_a.unsafe_ptr(),
        child_lambda.unsafe_ptr(),
        Int32(n_nodes),
        grid_dim=(node_grid, 1, 1),
        block_dim=(tpb, 1, 1),
    )
    if n_edges > 0:
        ctx.enqueue_function[label_edges_kernel](
            tree.parents.unsafe_ptr(),
            tree.children.unsafe_ptr(),
            tree.lambdas.unsafe_ptr(),
            is_cluster.unsafe_ptr(),
            next_a.unsafe_ptr(),
            child_lambda.unsafe_ptr(),
            root_key.unsafe_ptr(),
            Int32(n_leaves),
            Int32(n_nodes),
            Int32(n_edges),
            grid_dim=(max(1, (n_edges + tpb - 1) // tpb), 1, 1),
            block_dim=(tpb, 1, 1),
        )
    # A fixed round count: 2^rounds >= n_nodes reaches every chain's end.
    var in_b = td_find(ctx, next_a, next_b, n_nodes)
    var rep_ptr = next_b.unsafe_ptr() if in_b else next_a.unsafe_ptr()

    # `:131-134` identical_div (DEVIATION 5115); unread when epsilon is 0.
    var inverse_cluster_selection_epsilon = Float32(0.0)
    if cluster_selection_epsilon != Float32(0.0):
        inverse_cluster_selection_epsilon = identical_div(
            Float32(1.0), cluster_selection_epsilon
        )
    var root_single = Int32(1) if (
        n_selected == 1 and allow_single_cluster
    ) else Int32(0)
    if n_leaves > 0:
        ctx.enqueue_function[label_points_kernel](
            rep_ptr,
            child_lambda.unsafe_ptr(),
            root_key.unsafe_ptr(),
            d_labels.unsafe_ptr(),
            Int32(n_leaves),
            root_single,
            Int32(1) if cluster_selection_epsilon != Float32(0.0) else Int32(0),
            inverse_cluster_selection_epsilon,
            grid_dim=(max(1, (n_leaves + tpb - 1) // tpb), 1, 1),
            block_dim=(tpb, 1, 1),
        )
    ctx.synchronize()
    _ = next_a^
    _ = next_b^
    _ = child_lambda^
    _ = root_key^
    return d_labels^


def remap_labels_kernel(
    raw: MutPointer[Int32, MutAnyOrigin],
    label_map: MutPointer[Int32, MutAnyOrigin],
    dst: MutPointer[Int32, MutAnyOrigin],
    n_out: MutPointer[Int32, MutAnyOrigin],
    n_leaves: Int32,
):
    """`runner.h:221-233` per point, the outliers counted by an integer
    atomic."""
    var i = Int(block_dim.x) * Int(block_idx.x) + Int(thread_idx.x)
    if i >= Int(n_leaves):
        return
    var l = Int(raw[i])
    var v = Int32(-1)
    if l != -1:
        v = label_map[l]
    dst[i] = v
    if v == Int32(-1):
        _ = Atomic.fetch_add(n_out, Int32(1))


def deaths_key_kernel(
    parents: MutPointer[Int32, MutAnyOrigin],
    lambdas: MutPointer[Float32, MutAnyOrigin],
    dkey: MutPointer[Int32, MutAnyOrigin],
    n_leaves: Int32,
    n_edges: Int32,
):
    """`deaths[c]`, the max child lambda of each cluster, as an integer
    `Atomic.max` on `weight_order_key` (order-free)."""
    var e = Int(block_dim.x) * Int(block_idx.x) + Int(thread_idx.x)
    if e >= Int(n_edges):
        return
    var c = Int(parents[e]) - Int(n_leaves)
    _ = Atomic.max(dkey.unsafe_offset(c), weight_order_key(lambdas[e]))


def probabilities_kernel(
    children: MutPointer[Int32, MutAnyOrigin],
    lambdas: MutPointer[Float32, MutAnyOrigin],
    raw: MutPointer[Int32, MutAnyOrigin],
    dkey: MutPointer[Int32, MutAnyOrigin],
    dst: MutPointer[Float32, MutAnyOrigin],
    n_leaves: Int32,
    n_edges: Int32,
):
    """`get_probabilities_host`'s per-point step, one thread per edge."""
    var e = Int(block_dim.x) * Int(block_idx.x) + Int(thread_idx.x)
    if e >= Int(n_edges):
        return
    var child = Int(children[e])
    if child >= Int(n_leaves):
        return
    var cluster = Int(raw[child])
    if cluster == -1:
        return
    var k = dkey[cluster]
    var death = Float32(0.0)
    if k != Int32(-2147483647 - 1):
        death = weight_order_unkey(k)
    var lam = lambdas[e]
    if death == Float32(0.0) or isnan(lam) or isinf(lam):
        dst[child] = Float32(1.0)
    else:
        var lo = lam if lam < death else death
        dst[child] = ftz(identical_div(lo, death))


# ---------------------------------------- lane af-hdbscan2, FAST Apple ---
# `-D MOJOLEARN_HDB_SELECT_DEVICE` (epsilon == 0 only; the epsilon search
# needs the selected count on the host to decide whether it runs, so it
# keeps main's route). Main's extract already runs every pass on the device
# (stabilities by a fixed fold, EOM level by level, negation by pointer
# jumping, labels by pointer jumping, scores, probabilities); what is left
# is waits and reads: a synchronize after the stabilities, after the
# negation, after the selection and after the labelling, the selected count
# read back (1 wait) because the labelling's root rule and two output sizes
# use it, the stability scores downloaded (1 wait), then the eight output
# downloads (8 waits, or 1 under ONE_SYNC). Here the same kernels run in the
# same order on the same values with NO wait in between; the selected
# count stays on the device (`off[n_clusters]`, read by
# `label_points_sel_kernel` for the root rule), the scores land in an
# n_clusters-long buffer (the kernel writes slots < n_selected only), and
# every output, the count included, comes back in ONE readback. The EOM
# stays one launch per cluster-tree level: it is a tree recurrence whose
# float sums must keep their order (0 + stab[k0] + stab[k1]), so neither a
# scan nor a pointer-jumping form gives the same bits; the levels carry no
# wait. Labels, probabilities, stabilities and scores are main's bits.


def label_points_sel_kernel(
    rep: MutPointer[Int32, MutAnyOrigin],
    child_lambda: MutPointer[Float32, MutAnyOrigin],
    root_key: MutPointer[Int32, MutAnyOrigin],
    labels: MutPointer[Int32, MutAnyOrigin],
    off: MutPointer[Int32, MutAnyOrigin],
    n_clusters: Int32,
    n_leaves_in: Int32,
    allow: Int32,
):
    """`label_points_kernel` (epsilon 0) with `root_single = n_selected ==
    1 and allow_single_cluster`, n_selected read from the device scan."""
    var i = Int(block_dim.x) * Int(block_idx.x) + Int(thread_idx.x)
    var n_leaves = Int(n_leaves_in)
    if i >= n_leaves:
        return
    var root_single = allow != Int32(0) and off[Int(n_clusters)] == Int32(1)
    var cluster = Int(rep.unsafe_load(i))
    var out = Int32(-1)
    if cluster > n_leaves:
        out = Int32(cluster - n_leaves)
    elif cluster == n_leaves and root_single:
        var lam = child_lambda.unsafe_load(i)
        if lam >= weight_order_unkey(root_key.unsafe_load(0)):
            out = Int32(0)
    labels.unsafe_store(i, out)


def _extract_one_read(
    ctx: DeviceContext,
    mut tree: DeviceTree,
    cluster_selection_method: Int,
    allow_single_cluster: Bool,
    max_cluster_size_in: Int,
    sabotage: Int32,
) raises -> ExtractOutput:
    """`extract_clusters` at epsilon 0 with one wait (block comment above)."""
    var n_clusters = tree.n_clusters
    var n_leaves = tree.n_leaves
    var n_edges = tree.n_edges
    var n = n_clusters
    if n_clusters < 1:
        raise Error(
            "hdbscan.compute_stabilities: n_clusters=" + String(n_clusters)
            + " < 1; the condensed tree has no cluster to score"
        )
    if (
        cluster_selection_method != CLUSTER_SELECTION_EOM
        and cluster_selection_method != CLUSTER_SELECTION_LEAF
    ):
        raise Error(
            "hdbscan.select_clusters: cluster_selection_method="
            + String(cluster_selection_method)
            + " refused by name; their enum has exactly two values, EOM=0"
            " and LEAF=1 (hdbscan.hpp:126)"
        )
    if n_edges < 1:
        raise Error(
            "hdbscan.max_lambda_of: the condensed tree has no edges; their"
            " thrust::max_element at runner.h:209 dereferences an empty"
            " range here"
        )
    var stabilities = ctx.enqueue_create_buffer[DType.float32](n_clusters)
    var is_cluster = ctx.enqueue_create_buffer[DType.int32](n_clusters)
    var label_map = ctx.enqueue_create_buffer[DType.int32](n_clusters)
    var inverse = ctx.enqueue_create_buffer[DType.int32](n_clusters)

    # compute_stabilities, no wait.
    var births = ctx.enqueue_create_buffer[DType.float32](n_clusters)
    ctx.enqueue_memset(births, Float32(0.0))
    ctx.enqueue_function[births_init_kernel](
        births.unsafe_ptr(), tree.children.unsafe_ptr(),
        tree.lambdas.unsafe_ptr(), Int32(n_leaves), Int32(n_edges),
        grid_dim=((n_edges + STAB_TPB - 1) // STAB_TPB, 1, 1),
        block_dim=(STAB_TPB, 1, 1),
    )
    ctx.enqueue_memset(stabilities, Float32(0.0))
    ctx.enqueue_function[cluster_stability_kernel](
        stabilities.unsafe_ptr(), births.unsafe_ptr(),
        tree.indptr.unsafe_ptr(), tree.lambdas.unsafe_ptr(),
        tree.sizes.unsafe_ptr(), Int32(n_clusters), sabotage,
        grid_dim=(n_clusters, 1, 1), block_dim=(STAB_FOLD, 1, 1),
    )

    var max_cluster_size = max_cluster_size_in
    if max_cluster_size <= 0:
        max_cluster_size = n_leaves

    # select_clusters, no wait.
    var fr = ctx.enqueue_create_buffer[DType.int32](n)
    var pa = ctx.enqueue_create_buffer[DType.int32](n)
    var va = ctx.enqueue_create_buffer[DType.int32](n)
    var pb = ctx.enqueue_create_buffer[DType.int32](n)
    var vb = ctx.enqueue_create_buffer[DType.int32](n)
    if cluster_selection_method == CLUSTER_SELECTION_EOM:
        ctx.enqueue_function[eom_init_kernel](
            is_cluster.unsafe_ptr(), fr.unsafe_ptr(), tree.csize.unsafe_ptr(),
            tree.kids.unsafe_ptr(),
            Int32(1) if allow_single_cluster else Int32(0), Int32(n),
            grid_dim=(td_grid(n), 1, 1), block_dim=(TD_TPB, 1, 1),
        )
        var tree_top = 0 if allow_single_cluster else 1
        var level = tree.max_cdepth
        while level >= 0:
            ctx.enqueue_function[eom_level_kernel](
                stabilities.unsafe_ptr(), is_cluster.unsafe_ptr(),
                fr.unsafe_ptr(), tree.kids.unsafe_ptr(),
                tree.csize.unsafe_ptr(), tree.cdepth.unsafe_ptr(),
                Int32(level), Int32(tree_top), Int32(max_cluster_size),
                Int32(1) if sabotage == HDB_SAB_EOM_NO_UPDATE else Int32(0),
                Int32(n),
                grid_dim=(td_grid(n), 1, 1), block_dim=(TD_TPB, 1, 1),
            )
            level -= 1
        ctx.enqueue_function[negate_init_kernel](
            tree.cpar.unsafe_ptr(), fr.unsafe_ptr(), pa.unsafe_ptr(),
            va.unsafe_ptr(), Int32(n),
            grid_dim=(td_grid(n), 1, 1), block_dim=(TD_TPB, 1, 1),
        )
        var nb = td_path_reduce[False](ctx, pa, va, pb, vb, n)
        var vptr = vb.unsafe_ptr() if nb else va.unsafe_ptr()
        ctx.enqueue_function[negate_apply_kernel](
            vptr, is_cluster.unsafe_ptr(), Int32(n),
            grid_dim=(td_grid(n), 1, 1), block_dim=(TD_TPB, 1, 1),
        )
    else:
        ctx.enqueue_function[leaf_kernel](
            tree.kids.unsafe_ptr(), is_cluster.unsafe_ptr(), Int32(n),
            grid_dim=(td_grid(n), 1, 1), block_dim=(TD_TPB, 1, 1),
        )
    var off = ctx.enqueue_create_buffer[DType.int32](n + 1)
    ctx.enqueue_function[flag_kernel](
        is_cluster.unsafe_ptr(), off.unsafe_ptr(), Int32(n),
        grid_dim=(td_grid(n + 1), 1, 1), block_dim=(TD_TPB, 1, 1),
    )
    td_exclusive_scan(ctx, off, n + 1)
    ctx.enqueue_function[label_map_kernel](
        is_cluster.unsafe_ptr(), off.unsafe_ptr(), label_map.unsafe_ptr(),
        inverse.unsafe_ptr(), Int32(n),
        grid_dim=(td_grid(n), 1, 1), block_dim=(TD_TPB, 1, 1),
    )

    # do_labelling_device, no wait, the root rule read on the device.
    var n_nodes = n_leaves + n_clusters
    var next_a = ctx.enqueue_create_buffer[DType.int32](n_nodes)
    var next_b = ctx.enqueue_create_buffer[DType.int32](n_nodes)
    var child_lambda = ctx.enqueue_create_buffer[DType.float32](n_nodes)
    var root_key = ctx.enqueue_create_buffer[DType.int32](1)
    var raw = ctx.enqueue_create_buffer[DType.int32](max(n_leaves, 1))
    ctx.enqueue_memset(root_key, weight_order_key(Float32(0.0)))
    var node_grid = max(1, (n_nodes + LABEL_TPB - 1) // LABEL_TPB)
    ctx.enqueue_function[label_init_kernel](
        next_a.unsafe_ptr(), child_lambda.unsafe_ptr(), Int32(n_nodes),
        grid_dim=(node_grid, 1, 1), block_dim=(LABEL_TPB, 1, 1),
    )
    ctx.enqueue_function[label_edges_kernel](
        tree.parents.unsafe_ptr(), tree.children.unsafe_ptr(),
        tree.lambdas.unsafe_ptr(), is_cluster.unsafe_ptr(),
        next_a.unsafe_ptr(), child_lambda.unsafe_ptr(),
        root_key.unsafe_ptr(), Int32(n_leaves), Int32(n_nodes),
        Int32(n_edges),
        grid_dim=(max(1, (n_edges + LABEL_TPB - 1) // LABEL_TPB), 1, 1),
        block_dim=(LABEL_TPB, 1, 1),
    )
    var in_b = td_find(ctx, next_a, next_b, n_nodes)
    var rep_ptr = next_b.unsafe_ptr() if in_b else next_a.unsafe_ptr()
    var g_pts = max(1, (n_leaves + LABEL_TPB - 1) // LABEL_TPB)
    if n_leaves > 0:
        ctx.enqueue_function[label_points_sel_kernel](
            rep_ptr, child_lambda.unsafe_ptr(), root_key.unsafe_ptr(),
            raw.unsafe_ptr(), off.unsafe_ptr(), Int32(n), Int32(n_leaves),
            Int32(1) if allow_single_cluster else Int32(0),
            grid_dim=(g_pts, 1, 1), block_dim=(LABEL_TPB, 1, 1),
        )

    # the remap.
    var final = ctx.enqueue_create_buffer[DType.int32](max(n_leaves, 1))
    var n_out = ctx.enqueue_create_buffer[DType.int32](1)
    ctx.enqueue_memset(n_out, Int32(0))
    ctx.enqueue_function[remap_labels_kernel](
        raw.unsafe_ptr(), label_map.unsafe_ptr(), final.unsafe_ptr(),
        n_out.unsafe_ptr(), Int32(n_leaves),
        grid_dim=(g_pts, 1, 1), block_dim=(LABEL_TPB, 1, 1),
    )

    # get_stability_scores_device, no download (n_clusters slots).
    var max_key = ctx.enqueue_create_buffer[DType.int32](1)
    ctx.enqueue_memset(max_key, Int32(-2147483647 - 1))
    ctx.enqueue_function[max_lambda_key_kernel](
        tree.lambdas.unsafe_ptr(), max_key.unsafe_ptr(), Int32(n_edges),
        grid_dim=((n_edges + STAB_TPB - 1) // STAB_TPB, 1, 1),
        block_dim=(STAB_TPB, 1, 1),
    )
    var counts = ctx.enqueue_create_buffer[DType.int32](n_clusters)
    ctx.enqueue_memset(counts, Int32(0))
    ctx.enqueue_function[cluster_count_kernel](
        raw.unsafe_ptr(), counts.unsafe_ptr(), Int32(n_leaves),
        grid_dim=(max(1, (n_leaves + STAB_TPB - 1) // STAB_TPB), 1, 1),
        block_dim=(STAB_TPB, 1, 1),
    )
    var scores = ctx.enqueue_create_buffer[DType.float32](n_clusters)
    ctx.enqueue_memset(scores, Float32(0.0))
    ctx.enqueue_function[stability_score_kernel](
        stabilities.unsafe_ptr(), counts.unsafe_ptr(),
        label_map.unsafe_ptr(), max_key.unsafe_ptr(), scores.unsafe_ptr(),
        Int32(n_clusters),
        grid_dim=((n_clusters + STAB_TPB - 1) // STAB_TPB, 1, 1),
        block_dim=(STAB_TPB, 1, 1),
    )

    # the probabilities.
    var dkey = ctx.enqueue_create_buffer[DType.int32](n_clusters)
    ctx.enqueue_memset(dkey, Int32(-2147483647 - 1))
    var probs = ctx.enqueue_create_buffer[DType.float32](max(n_leaves, 1))
    ctx.enqueue_memset(probs, Float32(0.0))
    var g_edges = max(1, (n_edges + LABEL_TPB - 1) // LABEL_TPB)
    ctx.enqueue_function[deaths_key_kernel](
        tree.parents.unsafe_ptr(), tree.lambdas.unsafe_ptr(),
        dkey.unsafe_ptr(), Int32(n_leaves), Int32(n_edges),
        grid_dim=(g_edges, 1, 1), block_dim=(LABEL_TPB, 1, 1),
    )
    ctx.enqueue_function[probabilities_kernel](
        tree.children.unsafe_ptr(), tree.lambdas.unsafe_ptr(),
        raw.unsafe_ptr(), dkey.unsafe_ptr(), probs.unsafe_ptr(),
        Int32(n_leaves), Int32(n_edges),
        grid_dim=(g_edges, 1, 1), block_dim=(LABEL_TPB, 1, 1),
    )

    # ONE readback: every output and the selected count.
    var s_raw = td_stage_i32(ctx, raw)
    var s_final = td_stage_i32(ctx, final)
    var s_isc = td_stage_i32(ctx, is_cluster)
    var s_stab = td_stage_f32(ctx, stabilities)
    var s_map = td_stage_i32(ctx, label_map)
    var s_inv = td_stage_i32(ctx, inverse)
    var s_scores = td_stage_f32(ctx, scores)
    var s_probs = td_stage_f32(ctx, probs)
    var s_out = td_stage_i32(ctx, n_out)
    var s_off = td_stage_i32(ctx, off)
    ctx.synchronize()
    var n_selected = Int(s_off.unsafe_ptr().unsafe_load(n))
    var h_raw = td_take_i32(s_raw, n_leaves)
    var h_final = td_take_i32(s_final, n_leaves)
    var h_isc = td_take_i32(s_isc, n_clusters)
    var h_stab = td_take_f32(s_stab, n_clusters)
    var h_map = td_take_i32(s_map, n_clusters)
    var h_inv = td_take_i32(s_inv, n_selected)
    var h_scores = td_take_f32(s_scores, n_selected)
    var h_probs = td_take_f32(s_probs, n_leaves)
    var n_outliers = Int(s_out.unsafe_ptr().unsafe_load(0))
    _ = s_raw^
    _ = s_final^
    _ = s_isc^
    _ = s_stab^
    _ = s_map^
    _ = s_inv^
    _ = s_scores^
    _ = s_probs^
    _ = s_out^
    _ = s_off^
    _ = births^
    _ = fr^
    _ = pa^
    _ = va^
    _ = pb^
    _ = vb^
    _ = next_a^
    _ = next_b^
    _ = child_lambda^
    _ = root_key^
    _ = max_key^
    _ = counts^
    _ = dkey^
    _ = stabilities^
    _ = is_cluster^
    _ = label_map^
    _ = inverse^
    _ = raw^
    _ = final^
    _ = n_out^
    _ = scores^
    _ = probs^
    _ = off^
    return ExtractOutput(
        n_selected, n_outliers, h_raw^, h_final^, h_isc^, h_stab^, h_map^,
        h_inv^, h_scores^, h_probs^,
    )


def extract_clusters(
    ctx: DeviceContext,
    mut tree: DeviceTree,
    cluster_selection_method: Int,
    allow_single_cluster: Bool,
    max_cluster_size_in: Int,
    cluster_selection_epsilon: Float32,
    sabotage: Int32 = HDB_SAB_NONE,
) raises -> ExtractOutput:
    """`extract.cuh:246-314`, then the runner's tail (`runner.h:208-233`:
    max_lambda, the stability scores, the label remap) and the
    probabilities, all on the device over `tree`."""
    # lane af-hdbscan2 (-D MOJOLEARN_HDB_SELECT_DEVICE): the same passes
    # with no wait between them and one readback (epsilon 0 only).
    # fam2-cluster: the same route under IDENTICAL on every vendor
    # (IDN_HDB_SELECT_ONE_READ).
    comptime if HDB_SELECT_DEVICE or IDN_HDB_SELECT_ONE_READ:
        if cluster_selection_epsilon == Float32(0.0):
            return _extract_one_read(
                ctx, tree, cluster_selection_method, allow_single_cluster,
                max_cluster_size_in, sabotage,
            )
    var n_clusters = tree.n_clusters
    var n_leaves = tree.n_leaves

    # `:263`
    var stabilities = ctx.enqueue_create_buffer[DType.float32](n_clusters)
    var is_cluster = ctx.enqueue_create_buffer[DType.int32](n_clusters)
    var label_map = ctx.enqueue_create_buffer[DType.int32](n_clusters)
    var inverse = ctx.enqueue_create_buffer[DType.int32](n_clusters)
    compute_stabilities(ctx, tree, stabilities, sabotage)

    # `:266` if (max_cluster_size <= 0) max_cluster_size = n_leaves
    var max_cluster_size = max_cluster_size_in
    if max_cluster_size <= 0:
        max_cluster_size = n_leaves

    # `:268-295` the selection and the ascending label maps.
    var n_selected = select_clusters(
        ctx, tree, stabilities, is_cluster, label_map, inverse,
        cluster_selection_method, allow_single_cluster, max_cluster_size,
        cluster_selection_epsilon, sabotage,
    )

    # `:303-309` the labelling, on the device.
    var raw = do_labelling_device(
        ctx, tree, is_cluster, n_selected, allow_single_cluster,
        cluster_selection_epsilon,
    )

    # `runner.h:221-233` the remap.
    var final = ctx.enqueue_create_buffer[DType.int32](max(n_leaves, 1))
    var n_out = ctx.enqueue_create_buffer[DType.int32](1)
    ctx.enqueue_memset(n_out, Int32(0))
    var g_pts = max(1, (n_leaves + LABEL_TPB - 1) // LABEL_TPB)
    ctx.enqueue_function[remap_labels_kernel](
        raw.unsafe_ptr(), label_map.unsafe_ptr(), final.unsafe_ptr(),
        n_out.unsafe_ptr(), Int32(n_leaves),
        grid_dim=(g_pts, 1, 1), block_dim=(LABEL_TPB, 1, 1),
    )

    # `runner.h:208-219` max_lambda and get_stability_scores.
    var scores = get_stability_scores_device(
        ctx, tree, raw, stabilities, label_map, n_selected
    )

    # `:311` Membership::get_probabilities (DEVIATION 5116).
    var dkey = ctx.enqueue_create_buffer[DType.int32](n_clusters)
    ctx.enqueue_memset(dkey, Int32(-2147483647 - 1))
    var probs = ctx.enqueue_create_buffer[DType.float32](max(n_leaves, 1))
    ctx.enqueue_memset(probs, Float32(0.0))
    var g_edges = max(1, (tree.n_edges + LABEL_TPB - 1) // LABEL_TPB)
    ctx.enqueue_function[deaths_key_kernel](
        tree.parents.unsafe_ptr(), tree.lambdas.unsafe_ptr(),
        dkey.unsafe_ptr(), Int32(n_leaves), Int32(tree.n_edges),
        grid_dim=(g_edges, 1, 1), block_dim=(LABEL_TPB, 1, 1),
    )
    ctx.enqueue_function[probabilities_kernel](
        tree.children.unsafe_ptr(), tree.lambdas.unsafe_ptr(),
        raw.unsafe_ptr(), dkey.unsafe_ptr(), probs.unsafe_ptr(),
        Int32(n_leaves), Int32(tree.n_edges),
        grid_dim=(g_edges, 1, 1), block_dim=(LABEL_TPB, 1, 1),
    )

    var h_raw: List[Int32]
    var h_final: List[Int32]
    var h_isc: List[Int32]
    var h_stab: List[Float32]
    var h_map: List[Int32]
    var h_inv: List[Int32]
    var h_probs: List[Float32]
    var n_outliers: Int
    # lane af-hdbscan2 (FAST on Apple, -D MOJOLEARN_HDB_ONE_SYNC): the seven
    # downloads and the scalar staged, ONE wait, then taken; main's route
    # below waits eight times.
    comptime if HDB_ONE_SYNC or IDN_HDB_ONE_SYNC:
        var s_raw = td_stage_i32(ctx, raw)
        var s_final = td_stage_i32(ctx, final)
        var s_isc = td_stage_i32(ctx, is_cluster)
        var s_stab = td_stage_f32(ctx, stabilities)
        var s_map = td_stage_i32(ctx, label_map)
        var s_inv = td_stage_i32(ctx, inverse)
        var s_probs = td_stage_f32(ctx, probs)
        var s_out = td_stage_i32(ctx, n_out)
        ctx.synchronize()
        h_raw = td_take_i32(s_raw, n_leaves)
        h_final = td_take_i32(s_final, n_leaves)
        h_isc = td_take_i32(s_isc, n_clusters)
        h_stab = td_take_f32(s_stab, n_clusters)
        h_map = td_take_i32(s_map, n_clusters)
        h_inv = td_take_i32(s_inv, n_selected)
        h_probs = td_take_f32(s_probs, n_leaves)
        n_outliers = Int(s_out.unsafe_ptr().unsafe_load(0))
        _ = s_raw^
        _ = s_final^
        _ = s_isc^
        _ = s_stab^
        _ = s_map^
        _ = s_inv^
        _ = s_probs^
        _ = s_out^
    else:
        h_raw = td_download_i32(ctx, raw, n_leaves)
        h_final = td_download_i32(ctx, final, n_leaves)
        h_isc = td_download_i32(ctx, is_cluster, n_clusters)
        h_stab = td_download_f32(ctx, stabilities, n_clusters)
        h_map = td_download_i32(ctx, label_map, n_clusters)
        h_inv = td_download_i32(ctx, inverse, n_selected)
        h_probs = td_download_f32(ctx, probs, n_leaves)
        n_outliers = td_read_i32(ctx, n_out, 0)
    _ = stabilities^
    _ = is_cluster^
    _ = label_map^
    _ = inverse^
    _ = raw^
    _ = final^
    _ = n_out^
    _ = dkey^
    _ = probs^
    return ExtractOutput(
        n_selected, n_outliers, h_raw^, h_final^, h_isc^, h_stab^, h_map^,
        h_inv^, scores^, h_probs^,
    )
