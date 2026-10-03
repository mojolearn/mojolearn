# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The flat labels on the device. The dendrogram is `dendrogram_device.mojo`'s;
its host reference walk lives in `hierarchy/checks/dendrogram_host_ref.mojo`.

Reference: `UnionFind` (`:41-80`), `build_dendrogram_host` (`:104-155`),
`write_levels_kernel` (`:157-166`), `inherit_labels` (`:179-210`),
`init_label_roots` (`:212-224`) and `extract_flattened_clusters`
(`:238-326`), `cuvs/cpp/src/cluster/detail/agglomerative.cuh` (cuVS `94c2819`).
(`UnionFind` and its DEVIATION 622 moved to `hierarchy/checks/dendrogram_host_ref.mojo`
with `build_dendrogram_host`.)

WHY THE DENDROGRAM IS DETERMINISTIC GIVEN THE SORTED MST. `build_dendrogram
_host` walks the sorted edge list in order and, per edge, merges the two
ROOTS of its endpoints and records them; no float is compared, no order is
chosen. So `children`, `out_delta`, `out_size` are a pure function of the
sorted edge list (order AND orientation: `children[2i] = find(src)`,
`children[2i+1] = find(dst)`). DEVIATION 621 makes that list a pure function
of the edge set, and DEVIATION 620 makes the edge set a pure function of the
weights; the orientation is Boruvka's (`temp_src[tid] = tid`, the vertex
that added the edge, `mst_kernels.cuh:146`), which is itself a function of
the colors and therefore of the input.

WHY THE LABELS ARE TOO. `extract_flattened_clusters` cuts at
`cut_level = (n_leaves - 1) - (n_clusters - 1)` and labels the `n_clusters`
cluster roots 0.. in DESCENDING root-index order (`:293-310`: the last
`2(n_clusters-1)` children sorted descending, the `n_clusters` smallest of
them are the roots, label `j` to the `j`-th from the top of that tail).
`inherit_labels` then walks every node at or below the cut up to its root.
Reads of `labels[...]` race with writes of the SAME value (a node is
labelled by whichever thread reaches it, with the root's label either
way), so the output does not depend on thread order. Nothing here reads
`children`'s orientation: `write_levels_kernel` maps each CHILD to its
merge row, and the label roots are a SET.
"""

from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from hierarchy.impl.cluster.detail.dendrogram_device import _roots_kernel


def write_levels_kernel(
    children: MutPointer[Int32, MutAnyOrigin],
    parents: MutPointer[Int32, MutAnyOrigin],
    n_vertices_in: Int32,
):
    """`agglomerative.cuh:157-166`. `parents[child] = merge row`."""
    var tid = Int(block_dim.x) * Int(block_idx.x) + Int(thread_idx.x)
    if tid < Int(n_vertices_in):
        var level = tid // 2
        var child = children.unsafe_load(tid)
        parents.unsafe_store(Int(child), Int32(level))


def inherit_labels(
    children: MutPointer[Int32, MutAnyOrigin],
    levels: MutPointer[Int32, MutAnyOrigin],
    n_leaves_in: Int32,
    labels: MutPointer[Int32, MutAnyOrigin],
    cut_level: Int32,
    n_vertices_in: Int32,
):
    """`agglomerative.cuh:179-210`."""
    var tid = Int(block_dim.x) * Int(block_idx.x) + Int(thread_idx.x)
    if tid < Int(n_vertices_in):
        var node = children.unsafe_load(tid)
        var cur_level = Int32(tid // 2)
        # Any roots above the cut level should be ignored.
        # Any leaves at the cut level should already be labeled
        if cur_level > cut_level:
            return
        var cur_parent = node
        var label = labels.unsafe_load(Int(cur_parent))
        while label == Int32(-1):
            cur_parent = cur_level + n_leaves_in
            cur_level = levels.unsafe_load(Int(cur_parent))
            label = labels.unsafe_load(Int(cur_parent))
        labels.unsafe_store(Int(node), label)


def fill_labels_kernel(
    labels: MutPointer[Int32, MutAnyOrigin], value: Int32, n_in: Int32
):
    """`thrust::fill` over the labels (`:250`, `:301`)."""
    var tid = Int(block_dim.x) * Int(block_idx.x) + Int(thread_idx.x)
    if tid < Int(n_in):
        labels.unsafe_store(tid, value)


def init_label_roots_kernel(
    labels: MutPointer[Int32, MutAnyOrigin],
    roots: MutPointer[Int32, MutAnyOrigin],
    n_clusters_in: Int32,
):
    """`init_label_roots` under `thrust::for_each` (`:212-224`, `:304-310`):
    `labels[roots[j]] = j`."""
    var tid = Int(block_dim.x) * Int(block_idx.x) + Int(thread_idx.x)
    if tid < Int(n_clusters_in):
        labels.unsafe_store(Int(roots.unsafe_load(tid)), Int32(tid))


comptime EXTRACT_TPB = 256
"""Their `int tpb = 256` template default (`:238`)."""


def extract_flattened_clusters(
    ctx: DeviceContext,
    mut labels: DeviceBuffer[DType.int32],
    mut children: DeviceBuffer[DType.int32],
    n_clusters: Int,
    n_leaves: Int,
    tpb: Int = EXTRACT_TPB,
) raises:
    """`agglomerative.cuh:238-326`. `tpb` is their template parameter,
    exposed so the check can launch at two block sizes."""
    if n_clusters == 1:
        ctx.enqueue_function[fill_labels_kernel](
            labels.unsafe_ptr(),
            Int32(0),
            Int32(n_leaves),
            grid_dim=((n_leaves + tpb - 1) // tpb, 1, 1),
            block_dim=(tpb, 1, 1),
        )
        ctx.synchronize()
        return

    var n_edges = (n_leaves - 1) * 2
    if n_leaves <= 0:
        raise Error("hierarchy.extract_flattened_clusters: n_leaves must be positive")

    # `:263-272` n_vertices = max(children) + 1, checked against
    # (n_leaves - 1) * 2: every caller hands a dendrogram of a spanning tree
    # (`build_sorted_mst` refuses any other edge count and any second
    # component), whose children are exactly the 2 (n_leaves - 1) node ids
    # below the root, so the count is that, with no read of the children
    # (lane hr2-mds-agglo: it was a download and a host scan).
    var n_vertices = n_edges

    var levels = ctx.enqueue_create_buffer[DType.int32](n_vertices)
    var n_blocks = (n_vertices + tpb - 1) // tpb
    ctx.enqueue_function[write_levels_kernel](
        children.unsafe_ptr(),
        levels.unsafe_ptr(),
        Int32(n_vertices),
        grid_dim=(n_blocks, 1, 1),
        block_dim=(tpb, 1, 1),
    )

    # `:279-296` Step 1: label roots = the last (n_clusters - 1) * 2 children,
    # sorted DESCENDING (thrust::sort with thrust::greater); the n_clusters
    # at the TAIL of that order are the cluster roots: on the device, each
    # candidate placed by its count of larger candidates (`_roots_kernel`).
    var child_size = (n_clusters - 1) * 2
    var d_roots = ctx.enqueue_create_buffer[DType.int32](n_clusters)
    ctx.enqueue_function[_roots_kernel](
        children.unsafe_ptr(),
        d_roots.unsafe_ptr(),
        Int32(n_edges),
        Int32(n_clusters),
        grid_dim=((child_size + tpb - 1) // tpb, 1, 1),
        block_dim=(tpb, 1, 1),
    )

    # `:298-310` tmp_labels = -1; labels for the roots
    var tmp_labels = ctx.enqueue_create_buffer[DType.int32](n_vertices)
    ctx.enqueue_function[fill_labels_kernel](
        tmp_labels.unsafe_ptr(),
        Int32(-1),
        Int32(n_vertices),
        grid_dim=(n_blocks, 1, 1),
        block_dim=(tpb, 1, 1),
    )
    ctx.enqueue_function[init_label_roots_kernel](
        tmp_labels.unsafe_ptr(),
        d_roots.unsafe_ptr(),
        Int32(n_clusters),
        grid_dim=((n_clusters + tpb - 1) // tpb, 1, 1),
        block_dim=(tpb, 1, 1),
    )

    # `:312-321` Step 2: propagate
    var cut_level = (n_edges // 2) - (n_clusters - 1)
    ctx.enqueue_function[inherit_labels](
        children.unsafe_ptr(),
        levels.unsafe_ptr(),
        Int32(n_leaves),
        tmp_labels.unsafe_ptr(),
        Int32(cut_level),
        Int32(n_vertices),
        grid_dim=(n_blocks, 1, 1),
        block_dim=(tpb, 1, 1),
    )

    # `:323-324` copy tmp labels to actual labels
    var v_tmp = tmp_labels.create_sub_buffer[DType.int32](0, n_leaves)
    var v_labels = labels.create_sub_buffer[DType.int32](0, n_leaves)
    ctx.enqueue_copy(dst_buf=v_labels, src_buf=v_tmp)
    ctx.synchronize()
    _ = levels^
    _ = d_roots^
    _ = tmp_labels^
    _ = v_tmp^
    _ = v_labels^
