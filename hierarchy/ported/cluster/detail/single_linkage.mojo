# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Derivative work: the upstream file and its pinned commit are recorded in this lane's PORTED_MAP.tsv and in this file's own docstring. See NOTICE.
"""Single linkage: connectivities -> sorted MST -> dendrogram -> labels.

PORT OF `cuvs/cpp/src/cluster/detail/single_linkage.cuh`, cuVS `94c2819`:
`build_dist_linkage` (`:139-205`) and `single_linkage` (`:227-269`).
`build_mr_linkage` (`:50-118`, the mutual-reachability linkage HDBSCAN
uses) is NOT ported here and is listed in `hierarchy/UNPORTED.tsv`.
Transliterated, their order. Do not improve.

`single_linkage_output` (`cuvs/cluster/agglomerative.hpp`) is the struct of
out-pointers plus `m`, `n_clusters`, `n_leaves`, `n_connected_components`;
ours carries the same fields over two caller-owned device buffers.

THE KNOBS THAT ARE NOT THEIRS. `tile_tpb`, `mst_tpb`, `extract_tpb` are
block sizes (their launches take them from device properties or template
defaults) and `sabotage` selects a check arm; all four default to the
production values and exist so `linkage_check.mojo` can prove the output
bytes do not depend on the first three and DO depend on the pins the
fourth breaks.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from hierarchy.mojo_only.edge_order import LINK_SAB_NONE
from hierarchy.ported.cluster.detail.agglomerative import (
    EXTRACT_TPB,
    build_dendrogram_host,
    extract_flattened_clusters,
)
from hierarchy.ported.cluster.detail.connectivities import (
    LINKAGE_PAIRWISE,
    get_distance_graph,
)
from hierarchy.ported.cluster.detail.mst import build_sorted_mst
from neighbors.mojo_only.pinned_distance_tile import PINNED_TILE_TPB


@fieldwise_init
struct SingleLinkageOutput(Copyable, Movable):
    """`single_linkage_output<value_idx>` minus the two out-pointers, which
    are the `children` / `labels` buffers the caller passed in."""

    var m: Int
    var n_clusters: Int
    var n_leaves: Int
    var n_connected_components: Int
    var n_boruvka_rounds: Int
    """NOT THEIRS. The Boruvka round count, an integer stage for the card."""


def build_dist_linkage(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    c: Int,
    metric: Int,
    dist_type: Int,
    mut mst_rows: DeviceBuffer[DType.int32],
    mut mst_cols: DeviceBuffer[DType.int32],
    mut mst_weights: DeviceBuffer[DType.float32],
    mut out_dendrogram: DeviceBuffer[DType.int32],
    mut out_distances: DeviceBuffer[DType.float32],
    mut out_sizes: DeviceBuffer[DType.int32],
    tile_tpb: Int = PINNED_TILE_TPB,
    mst_tpb: Int = 256,
    sabotage: Int32 = LINK_SAB_NONE,
) raises -> Int:
    """`single_linkage.cuh:139-205`. Returns the Boruvka round count."""
    # `:153-168` 1. Construct distance graph. PAIRWISE needs indptr m+1,
    # indices/data m*m (their `resize`s inside the impl, `:199-200`).
    var nnz = m * m
    var indptr = ctx.enqueue_create_buffer[DType.int32](m + 1)
    var indices = ctx.enqueue_create_buffer[DType.int32](nnz)
    var pw_dists = ctx.enqueue_create_buffer[DType.float32](nnz)
    var norms = ctx.enqueue_create_buffer[DType.float32](m)
    get_distance_graph(
        ctx, x, m, n, metric, dist_type, c, indptr, indices, pw_dists, norms,
        tile_tpb, sabotage,
    )

    # `:170-191` 2. Construct MST, sorted by weights
    var color = ctx.enqueue_create_buffer[DType.int32](m)
    var n_edges = m - 1
    var rounds = build_sorted_mst(
        ctx, indptr, indices, pw_dists, m, n,
        mst_rows, mst_cols, mst_weights, color, nnz,
        max_iter=10, mst_tpb=mst_tpb, sabotage=sabotage,
    )
    # `:192` pw_dists.release()
    _ = pw_dists^
    _ = indices^
    _ = indptr^
    _ = norms^
    _ = color^

    # `:194-204` Perform hierarchical labeling
    build_dendrogram_host(
        ctx, mst_rows, mst_cols, mst_weights, n_edges,
        out_dendrogram, out_distances, out_sizes,
    )
    return rounds


def single_linkage(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    metric: Int,
    mut children: DeviceBuffer[DType.int32],
    mut labels: DeviceBuffer[DType.int32],
    c: Int,
    n_clusters: Int,
    dist_type: Int = LINKAGE_PAIRWISE,
    tile_tpb: Int = PINNED_TILE_TPB,
    mst_tpb: Int = 256,
    extract_tpb: Int = EXTRACT_TPB,
    sabotage: Int32 = LINK_SAB_NONE,
) raises -> SingleLinkageOutput:
    """`single_linkage.cuh:227-269`. `children` holds `(m - 1) * 2`,
    `labels` holds `m`."""
    if n_clusters > m:
        raise Error(
            "hierarchy.single_linkage: n_clusters must be less than or equal"
            " to the number of data points (n_clusters=" + String(n_clusters)
            + ", n_rows=" + String(m) + ")"
        )
    if n_clusters < 1:
        raise Error(
            "hierarchy.single_linkage: n_clusters=" + String(n_clusters)
            + " < 1 refused by name (their extract_flattened_clusters would"
            " index children at a negative offset)"
        )
    var n_edges = m - 1
    var mst_rows = ctx.enqueue_create_buffer[DType.int32](n_edges if n_edges > 0 else 1)
    var mst_cols = ctx.enqueue_create_buffer[DType.int32](n_edges if n_edges > 0 else 1)
    var mst_weights = ctx.enqueue_create_buffer[DType.float32](n_edges if n_edges > 0 else 1)
    var out_delta = ctx.enqueue_create_buffer[DType.float32](n_edges if n_edges > 0 else 1)
    var out_sizes = ctx.enqueue_create_buffer[DType.int32](n_edges if n_edges > 0 else 1)

    var rounds = build_dist_linkage(
        ctx, x, m, n, c, metric, dist_type,
        mst_rows, mst_cols, mst_weights, children, out_delta, out_sizes,
        tile_tpb, mst_tpb, sabotage,
    )

    # `:263`
    extract_flattened_clusters(ctx, labels, children, n_clusters, m, extract_tpb)

    _ = mst_rows^
    _ = mst_cols^
    _ = mst_weights^
    _ = out_delta^
    _ = out_sizes^
    # `:265-268`
    return SingleLinkageOutput(m, n_clusters, m, 1, rounds)
