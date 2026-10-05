# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Single linkage: connectivities -> sorted MST -> dendrogram -> labels.

Reference: `build_dist_linkage` (`:139-205`) and `single_linkage` (`:227-269`),
`cuvs/cpp/src/cluster/detail/single_linkage.cuh` (cuVS `94c2819`).
`build_mr_linkage` (`:50-118`, the mutual-reachability linkage HDBSCAN
uses) is NOT implemented here and is listed in `hierarchy/NOT_IMPLEMENTED.tsv`.
Steps run in the reference's order.

`single_linkage_output` (`cuvs/cluster/agglomerative.hpp`) is the struct of
out-pointers plus `m`, `n_clusters`, `n_leaves`, `n_connected_components`;
this implementation carries the same fields over two caller-owned device buffers.

THE KNOBS THE REFERENCE DOES NOT HAVE. `tile_tpb`, `mst_tpb`, `extract_tpb` are
block sizes (their launches take them from device properties or template
defaults) and `sabotage` selects a check arm; all four default to the
production values and exist so `linkage_check.mojo` can prove the output
bytes do not depend on the first three and DO depend on the pins the
fourth breaks.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from hierarchy.checks.edge_order import LINK_SAB_NONE
from hierarchy.impl.cluster.detail.agglomerative import (
    EXTRACT_TPB,
    extract_flattened_clusters,
)
from hierarchy.impl.cluster.detail.dendrogram_device import (
    IDN_DENDRO_UNION,
    build_dendrogram_device,
)
from hdbscan.impl.cluster.detail.dendrogram_union import build_dendrogram_union
from hierarchy.impl.cluster.detail.connectivities import (
    DISTANCE_L2_EXPANDED,
    DISTANCE_L2_SQRT_EXPANDED,
    LINKAGE_PAIRWISE,
    get_distance_graph,
)
from hierarchy.impl.cluster.detail.mst import build_sorted_mst
from hierarchy.impl.cluster.detail.fast_boruvka import fast_euclidean_mst
from checks.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_FAST,
    NUMERIC_IDENTICAL,
)
from std.sys.compile import is_defined
from std.sys.defines import get_defined_int
from hdbscan.impl.cluster.detail.sparse_mr_mst import sparse_mr_mst_device
from std.sys.info import has_apple_gpu_accelerator

comptime SL_FAST_BORUVKA = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and not is_defined["MOJOLEARN_SL_FAST_BORUVKA_OFF"]()
)
"""FAST on Apple: the PAIRWISE Euclidean MST from Boruvka rounds with the
distances computed on the fly (`fast_boruvka.mojo`) instead of the dense
`m * m` graph, which the M4 cannot allocate from ~38k rows up."""

#: fam2-cluster (2026-10-04), IDENTICAL, CANDIDATE ARM (default OFF; turn on
#: with `-D MOJOLEARN_IDN_SL_SPARSE_MST=1`, off again under
#: `-D MOJOLEARN_IDN_ALL_OFF=1`). Single linkage on the PAIRWISE
#: L2SqrtExpanded graph from HDBSCAN's matrix-free Boruvka
#: (`hdbscan/impl/cluster/detail/sparse_mr_mst.mojo`, DEVIATION 1620) with
#: every core distance +0.0 and `1 / alpha = 1.0`: the mutual reachability
#: `max(0, 0, 1.0 * d)` is the distance `d` bit for bit, so the search
#: returns the Euclidean MST with no m * m matrix (17 GB at 46,340 rows) and
#: with its bound pruning, and the 46,340-row refusal does not apply.
#: The MST under the total order is the dense arm's edge set and weights.
#: TWO DIFFERENCES TO CHECK before adopting it: (1) the edges come oriented
#: (lo, hi), which is the host column's order (`hierarchy/host/
#: linkage_host.mojo`) but not Boruvka's, so the two columns of a `children`
#: row may swap against the dense device arm (the gate compares them as an
#: unordered pair); (2) the sparse search refuses an infinite weight where
#: the dense arm keeps +inf. Rows at or above
#: `-D MOJOLEARN_IDN_SL_SPARSE_MIN_ROWS=<n>` (default 4096) take it.
comptime IDN_SL_SPARSE_MST = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and is_defined["MOJOLEARN_IDN_SL_SPARSE_MST"]()
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
comptime IDN_SL_SPARSE_MIN_ROWS = get_defined_int[
    "MOJOLEARN_IDN_SL_SPARSE_MIN_ROWS", 4096
]()

comptime SL_FAST_BORUVKA_MIN_ROWS = 4096 if is_defined[
    "MOJOLEARN_LEGACY_NARROW_SL_BORUVKA"
]() else 1
"""Rows the FAST Boruvka route needs (it takes m > this). Was 4096, just
below the board's 5,000-row tiny shape ("dense as fast at 5,000 rows");
removed as benchmark-tuned on 2026-10-04, replacement UNMEASURED: every
m >= 2 takes the matrix-free route (its buffers are all m-sized; no tile
needs a minimum m). `-D MOJOLEARN_LEGACY_NARROW_SL_BORUVKA` restores 4096.
The `n <= 64` guard at the call sites is the kernel's: the query point is a
register list `InlineArray[Float32, PPT * DMAX]` with DMAX tiers 8..64
(`fast_boruvka.mojo`), so it stays."""
from neighbors.checks.pinned_distance_tile import PINNED_TILE_TPB


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
    comptime if SL_FAST_BORUVKA:
        if (
            dist_type == LINKAGE_PAIRWISE
            and sabotage == LINK_SAB_NONE
            and (
                metric == DISTANCE_L2_SQRT_EXPANDED
                or metric == DISTANCE_L2_EXPANDED
            )
            and n <= 64
            and m > SL_FAST_BORUVKA_MIN_ROWS
        ):
            # `core_ptr` is read only on the mutual-reachability arm; a
            # Pointer cannot be null, so it gets `x`'s address.
            var r = fast_euclidean_mst(
                ctx, x, m, n, metric == DISTANCE_L2_SQRT_EXPANDED,
                mst_rows, mst_cols, mst_weights,
                False, x.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
            )
            build_dendrogram_device(
                ctx, mst_rows, mst_cols, mst_weights, m - 1,
                out_dendrogram, out_distances, out_sizes,
            )
            return r
    comptime if IDN_SL_SPARSE_MST:
        if (
            dist_type == LINKAGE_PAIRWISE
            and sabotage == LINK_SAB_NONE
            and metric == DISTANCE_L2_SQRT_EXPANDED
            and m >= IDN_SL_SPARSE_MIN_ROWS
        ):
            var zero_core = ctx.enqueue_create_buffer[DType.float32](m)
            ctx.enqueue_memset(zero_core, Float32(0.0))
            var sr = sparse_mr_mst_device(
                ctx, x, zero_core, m, n, Float32(1.0),
                mst_rows, mst_cols, mst_weights,
            )
            _ = zero_core^
            comptime if IDN_DENDRO_UNION:
                build_dendrogram_union(
                    ctx, mst_rows, mst_cols, mst_weights, m - 1,
                    out_dendrogram, out_distances, out_sizes, drain=True,
                )
            else:
                build_dendrogram_device(
                    ctx, mst_rows, mst_cols, mst_weights, m - 1,
                    out_dendrogram, out_distances, out_sizes,
                )
            return sr
    # `:153-168` 1. Construct distance graph. PAIRWISE needs indptr m+1,
    # indices/data m*m (their `resize`s inside the impl, `:199-200`).
    var nnz = m * m
    var indptr = ctx.enqueue_create_buffer[DType.int32](m + 1)
    # the column of cell e is e % m: the DENSE solver computes it, so the
    # m * m index array is neither written nor read (lane/cluster-apple)
    var indices = ctx.enqueue_create_buffer[DType.int32](1)
    var pw_dists = ctx.enqueue_create_buffer[DType.float32](nnz)
    var norms = ctx.enqueue_create_buffer[DType.float32](m)
    get_distance_graph(
        ctx, x, m, n, metric, dist_type, c, indptr, indices, pw_dists, norms,
        tile_tpb, sabotage, fill_indices=False,
    )

    # `:170-191` 2. Construct MST, sorted by weights
    var color = ctx.enqueue_create_buffer[DType.int32](m)
    var n_edges = m - 1
    var rounds = build_sorted_mst[DENSE=True](
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
    # fam2-cluster, IDN_DENDRO_UNION: the same three outputs from one
    # lock-free union launch per level, no flag readback.
    comptime if IDN_DENDRO_UNION:
        build_dendrogram_union(
            ctx, mst_rows, mst_cols, mst_weights, n_edges,
            out_dendrogram, out_distances, out_sizes, drain=True,
        )
    else:
        build_dendrogram_device(
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
