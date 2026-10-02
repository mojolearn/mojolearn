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
from hdbscan.impl.detail.select import (
    SELECT_TPB,
    select_clusters,
)
from hdbscan.impl.detail.stabilities import (
    STAB_TPB,
    compute_stabilities,
)
from hierarchy.checks.edge_order import weight_order_key, weight_order_unkey
from std.math import isinf, isnan

from checks.numerics import ftz, identical_div


@fieldwise_init
struct ExtractOutput(Copyable, Movable):
    """What `extract_clusters` hands back. Theirs writes through
    out-pointers and returns `clusters.size()` (`:313`); ours returns the
    same values in one struct so the driver can record each as a stage
    without a second call."""

    var n_selected: Int
    var labels: List[Int32]
    """`n_leaves`, CONDENSED cluster ids (or -1), before the label_map
    remap their `runner.h:226-233` applies."""
    var is_cluster: List[Int32]
    """`n_clusters`, the selection."""
    var tree_stabilities: List[Float32]
    """`n_clusters`, AFTER `excess_of_mass` mutated it (DEVIATION 1605)."""
    var label_map: List[Int32]
    """`n_clusters`, condensed id -> final label, -1 where unselected."""
    var inverse_label_map: List[Int32]
    """`n_selected`, final label -> condensed id."""


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
    tree: CondensedHierarchy,
    mut is_cluster: DeviceBuffer[DType.int32],
    n_selected: Int,
    n_leaves: Int,
    allow_single_cluster: Bool,
    cluster_selection_epsilon: Float32,
    tpb: Int = LABEL_TPB,
) raises -> List[Int32]:
    """`extract.cuh:88-167` on the device (this file's header). Returns the
    `n_leaves` condensed cluster ids (or -1), the host walk's integers."""
    var n_edges = tree.n_edges
    var n_clusters = tree.n_clusters
    var n_nodes = n_leaves + n_clusters
    var d_parents = ctx.enqueue_create_buffer[DType.int32](max(n_edges, 1))
    var d_children = ctx.enqueue_create_buffer[DType.int32](max(n_edges, 1))
    var d_lambdas = ctx.enqueue_create_buffer[DType.float32](max(n_edges, 1))
    var next_a = ctx.enqueue_create_buffer[DType.int32](n_nodes)
    var next_b = ctx.enqueue_create_buffer[DType.int32](n_nodes)
    var child_lambda = ctx.enqueue_create_buffer[DType.float32](n_nodes)
    var root_key = ctx.enqueue_create_buffer[DType.int32](1)
    var d_labels = ctx.enqueue_create_buffer[DType.int32](max(n_leaves, 1))
    var h_parents = tree.parents.copy()
    var h_children = tree.children.copy()
    var h_lambdas = tree.lambdas.copy()
    if n_edges > 0:
        ctx.enqueue_copy(
            dst_buf=d_parents.create_sub_buffer[DType.int32](0, n_edges),
            src_ptr=h_parents.unsafe_ptr(),
        )
        ctx.enqueue_copy(
            dst_buf=d_children.create_sub_buffer[DType.int32](0, n_edges),
            src_ptr=h_children.unsafe_ptr(),
        )
        ctx.enqueue_copy(
            dst_buf=d_lambdas.create_sub_buffer[DType.float32](0, n_edges),
            src_ptr=h_lambdas.unsafe_ptr(),
        )
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
            d_parents.unsafe_ptr(),
            d_children.unsafe_ptr(),
            d_lambdas.unsafe_ptr(),
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
    var rounds = 1
    while (1 << rounds) < n_nodes:
        rounds += 1
    for r in range(rounds):
        if r % 2 == 0:
            ctx.enqueue_function[label_jump_kernel](
                next_a.unsafe_ptr(),
                next_b.unsafe_ptr(),
                Int32(n_nodes),
                grid_dim=(node_grid, 1, 1),
                block_dim=(tpb, 1, 1),
            )
        else:
            ctx.enqueue_function[label_jump_kernel](
                next_b.unsafe_ptr(),
                next_a.unsafe_ptr(),
                Int32(n_nodes),
                grid_dim=(node_grid, 1, 1),
                block_dim=(tpb, 1, 1),
            )
    var rep_ptr = next_b.unsafe_ptr() if rounds % 2 == 1 else next_a.unsafe_ptr()

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
    var labels = _download_i32(ctx, d_labels, n_leaves)
    _ = h_parents^
    _ = h_children^
    _ = h_lambdas^
    _ = d_parents^
    _ = d_children^
    _ = d_lambdas^
    _ = next_a^
    _ = next_b^
    _ = child_lambda^
    _ = root_key^
    _ = d_labels^
    return labels^


def extract_clusters(
    ctx: DeviceContext,
    tree: CondensedHierarchy,
    n_leaves: Int,
    cluster_selection_method: Int,
    allow_single_cluster: Bool,
    max_cluster_size_in: Int,
    cluster_selection_epsilon: Float32,
    stab_tpb: Int = STAB_TPB,
    select_tpb: Int = SELECT_TPB,
    sabotage: Int32 = HDB_SAB_NONE,
) raises -> ExtractOutput:
    """`extract.cuh:246-314`."""
    var n_clusters = tree.n_clusters

    # `:263`
    var stabilities = ctx.enqueue_create_buffer[DType.float32](n_clusters)
    var is_cluster = ctx.enqueue_create_buffer[DType.int32](n_clusters)
    ctx.synchronize()
    compute_stabilities(ctx, tree, stabilities, stab_tpb, sabotage)

    # `:266` if (max_cluster_size <= 0) max_cluster_size = n_leaves
    var max_cluster_size = max_cluster_size_in
    if max_cluster_size <= 0:
        max_cluster_size = n_leaves

    # `:268-275`
    _ = select_clusters(
        ctx, tree, stabilities, is_cluster, cluster_selection_method,
        allow_single_cluster, max_cluster_size, cluster_selection_epsilon,
        select_tpb, sabotage,
    )

    # `:277-284` is_cluster back to the host; clusters = the ascending set
    # of `i + n_leaves` for every selected i.
    var h_isc = _download_i32(ctx, is_cluster, n_clusters)
    var h_stab = _download_f32(ctx, stabilities, n_clusters)

    # `:286-295` the forward and inverse maps, in ascending cluster order.
    var label_map = List[Int32](capacity=n_clusters)
    for _ in range(n_clusters):
        label_map.append(Int32(-1))
    var inverse_label_map = List[Int32]()
    var n_selected = 0
    for i in range(n_clusters):
        if h_isc[i] != Int32(0):
            label_map[i] = Int32(n_selected)
            inverse_label_map.append(Int32(i))
            n_selected += 1

    # `:303-309`, on the device: `is_cluster` stays where the selection
    # left it, and the membership test reads it by condensed id.
    var labels = do_labelling_device(
        ctx, tree, is_cluster, n_selected, n_leaves, allow_single_cluster,
        cluster_selection_epsilon,
    )

    # `:311` Membership::get_probabilities: `get_probabilities_host`
    # (DEVIATION 5116), called by the runner on these raw labels.

    _ = stabilities^
    _ = is_cluster^
    return ExtractOutput(
        n_selected, labels^, h_isc^, h_stab^, label_map^, inverse_label_map^
    )


def _download_i32(
    ctx: DeviceContext, buf: DeviceBuffer[DType.int32], n: Int
) raises -> List[Int32]:
    var h = ctx.enqueue_create_host_buffer[DType.int32](n)
    ctx.synchronize()
    var v = buf.create_sub_buffer[DType.int32](0, n)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=v)
    ctx.synchronize()
    var out = List[Int32](capacity=n)
    for i in range(n):
        out.append(h.unsafe_ptr().unsafe_load(i))
    _ = h^
    _ = v^
    return out^


def _download_f32(
    ctx: DeviceContext, buf: DeviceBuffer[DType.float32], n: Int
) raises -> List[Float32]:
    var h = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.synchronize()
    var v = buf.create_sub_buffer[DType.float32](0, n)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=v)
    ctx.synchronize()
    var out = List[Float32](capacity=n)
    for i in range(n):
        out.append(h.unsafe_ptr().unsafe_load(i))
    _ = h^
    _ = v^
    return out^
