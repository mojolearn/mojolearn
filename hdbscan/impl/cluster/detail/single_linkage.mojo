# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`build_mr_linkage`: the linkage in mutual reachability space.

Reference: `cuvs/cpp/src/cluster/detail/single_linkage.cuh::build_mr_linkage`
(`:50-118`, cuVS `94c2819`), the function `hierarchy/NOT_IMPLEMENTED.tsv` line
8 names as "HDBSCAN's linkage (core distances, mutual reachability), not
single linkage's ... it is the ROADMAP's Phase 1 and would reuse this
lane's mst_solver.mojo and agglomerative.mojo unchanged."

THAT CLAIM HELD. Nothing in `hierarchy/impl/sparse/solver/`,
`hierarchy/impl/cluster/detail/agglomerative.mojo`,
`hierarchy/impl/sparse/op/sort.mojo` or
`hierarchy/checks/edge_order.mojo` was changed, copied or re-derived
for this lane; they are IMPORTED. The one thing the claim did not say,
and that this lane found, is that the GRAPH is not carried over: the reference's
is a sparse k-NN COO whose MST is a forest, and the fix-up it needs is
the part `hierarchy` records as NOT IMPLEMENTED. See DEVIATION 1600 in
`hdbscan/checks/mutual_reachability_dense.mojo`.

WHAT THE REFERENCE FUNCTION DOES, STEP FOR STEP (`:62-117`), AND WHAT THIS ONE DOES
  `:64-79`   `mutual_reachability_graph(...)` -> indptr, core_dists, COO
             OURS: `compute_core_dists` (their `compute_knn` +
             `core_distances`, unchanged) then the DENSE transform,
             DEVIATION 1600. Their `mutual_reachability_indptr` is
             `hierarchy`'s dense CSR `indptr[i] = i * m`.
  `:81-84`   `color`, `MutualReachabilityFixConnectivitiesRedOp`
             OURS: `color` unchanged; the reduction op is an argument of
             the FIX-UP LOOP ONLY (`build_sorted_mst`'s `connect_knn_
             graph`), which a complete graph never enters, so it is not
             implemented. `hdbscan/NOT_IMPLEMENTED.tsv` has the row.
  `:88-102`  `build_sorted_mst(...)` with `nnz = mr_coo.nnz`
             OURS: `hierarchy/impl/cluster/detail/mst.mojo::
             build_sorted_mst`, unchanged, with `nnz = m * m`.
  `:107-117` `build_dendrogram_host(...)`
             OURS: `hierarchy/impl/cluster/detail/dendrogram_device.mojo::
             build_dendrogram_device`, the same three outputs bit for bit
             on the device (integer work only), so no host step sits
             between the MST and the condense.

THE SABOTAGE ARGUMENT IS NOT FORWARDED INTO `hierarchy/`. This lane's
`HDB_SAB_*` constants and that lane's `LINK_SAB_*` constants are two
independent numberings that share small integers, so every call into
`hierarchy` below passes `LINK_SAB_NONE` EXPLICITLY. Forwarding would
silently select a distance-tile sabotage whenever this lane asked for a
condense sabotage, which is the kind of defect a green suite does not
show.
"""

from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import IdentityTrace
from hdbscan.checks.hdbscan_sabotage import HDB_SAB_MST_ORIENT_RAW, HDB_SAB_NONE
from hdbscan.checks.mutual_reachability_dense import (
    MR_TPB,
    mutual_reachability_dense,
    mutual_reachability_dense_guarded,
    refuse_nonfinite_device,
)
from hdbscan.impl.cluster.detail.sparse_mr_mst import IDN_HDB_MST_SEED_KNN, sparse_mr_mst_device
from hdbscan.impl.detail.reachability import (
    CORE_TPB,
    compute_core_dists,
)
from hierarchy.checks.edge_order import LINK_SAB_NONE, edge_hi, edge_lo
from hierarchy.impl.cluster.detail.dendrogram_device import (
    IDN_DENDRO_UNION,
    build_dendrogram_device,
)
from hdbscan.impl.detail.idn_switches import (
    IDN_HDB_MR_FUSED_GUARD,
    IDN_HDB_SPARSE_MIN_ROWS,
)
from hierarchy.impl.cluster.detail.connectivities import (
    DISTANCE_L2_SQRT_EXPANDED,
    PAIRWISE_MAX_ROWS,
    pairwise_distances,
)
from hierarchy.impl.cluster.detail.mst import build_sorted_mst
from hierarchy.impl.cluster.detail.fast_boruvka import fast_euclidean_mst
from hierarchy.impl.cluster.detail.single_linkage import (
    SL_FAST_BORUVKA,
    SL_FAST_BORUVKA_MIN_ROWS,
)
from checks.numerics import identical_div
from hdbscan.impl.detail.fast_apple import (
    HDB_DEV_BORUVKA,
    HDB_LINKAGE_DEVICE,
)
from hdbscan.impl.cluster.detail.dendrogram_union import build_dendrogram_union
from hdbscan.impl.cluster.detail.fast_mr_mst_device import fast_mr_mst_device
from neighbors.checks.pinned_distance_tile import PINNED_TILE_TPB
from std.os import getenv
from std.time import perf_counter_ns


comptime MR_GRAPH_AUTO = 0
"""The dense graph up to `PAIRWISE_MAX_ROWS`, the sparse arm past it
(DEVIATION 1620). What every caller outside the checks passes."""
comptime MR_GRAPH_DENSE = 1
"""The dense m x m graph (DEVIATION 1600) at any size; refused past
`PAIRWISE_MAX_ROWS`."""
comptime MR_GRAPH_SPARSE = 2
"""The on-the-fly Boruvka (DEVIATION 1620) at any size; the seam check
runs it where the dense arm can run too."""

comptime ORIENT_TPB = 256


def _orient_edges_kernel(
    src: MutPointer[Int32, MutAnyOrigin],
    dst: MutPointer[Int32, MutAnyOrigin],
    n_edges: Int32,
):
    """DEVIATION 1614 per edge: `(min(u, v), max(u, v))`. One thread per
    edge, each writing only its own slots, so there is no order to pin."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_edges):
        return
    var cu = src.unsafe_load(i)
    var cv = dst.unsafe_load(i)
    src.unsafe_store(i, edge_lo(cu, cv))
    dst.unsafe_store(i, edge_hi(cu, cv))


def _interleave_edges_kernel(
    src: MutPointer[Int32, MutAnyOrigin],
    dst: MutPointer[Int32, MutAnyOrigin],
    pairs: MutPointer[Int32, MutAnyOrigin],
    n_edges: Int32,
):
    """`(src[i], dst[i])` at `pairs[2i], pairs[2i + 1]`: the trace's edge list."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_edges):
        return
    pairs.unsafe_store(2 * i, src.unsafe_load(i))
    pairs.unsafe_store(2 * i + 1, dst.unsafe_load(i))


def build_mr_linkage(
    ctx: DeviceContext,
    mut trace: IdentityTrace,
    mut x_host: List[Float32],
    mut x: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    min_samples: Int,
    alpha: Float32,
    metric: Int,
    mut core_dists: DeviceBuffer[DType.float32],
    mut mst_rows: DeviceBuffer[DType.int32],
    mut mst_cols: DeviceBuffer[DType.int32],
    mut mst_weights: DeviceBuffer[DType.float32],
    mut out_dendrogram: DeviceBuffer[DType.int32],
    mut out_distances: DeviceBuffer[DType.float32],
    mut out_sizes: DeviceBuffer[DType.int32],
    tile_tpb: Int = PINNED_TILE_TPB,
    mst_tpb: Int = 256,
    mr_tpb: Int = MR_TPB,
    core_tpb: Int = CORE_TPB,
    sabotage: Int32 = HDB_SAB_NONE,
    graph: Int = MR_GRAPH_AUTO,
) raises -> Int:
    """`single_linkage.cuh:50-118`. Returns the Boruvka round count.

    `graph` picks the arm (`MR_GRAPH_*`); both return the same tree, bit
    for bit (DEVIATION 1620, `hdbscan/checks/sparse_mr_check.mojo`).

    `x_host` is the SAME data as `x`, on the host. No step here reads it
    any more (the k-NN and the sparse MST take the device buffer, lane
    cgr3-hdbscan-mst); it stays in the signature for the callers.
    """
    if m < 2:
        raise Error(
            "hdbscan.build_mr_linkage: n_rows=" + String(m)
            + " < 2; a dendrogram needs at least two points"
        )
    # FAST on Apple: the mutual-reachability MST from Boruvka rounds with
    # the reachabilities computed on the fly (hierarchy's
    # `fast_euclidean_mst`, mutual_reach=True) instead of the dense graph.
    var use_fast = False
    comptime if SL_FAST_BORUVKA:
        use_fast = (
            m > SL_FAST_BORUVKA_MIN_ROWS
            and n <= 64
            and sabotage == HDB_SAB_NONE
            and not trace.enabled
            and graph == MR_GRAPH_AUTO
        )
    # DEVIATION 1620: past the dense bound (or when asked) the mutual
    # reachability MST is built without the m * m graph.
    # fam2-cluster: the auto switch point is IDN_HDB_SPARSE_MIN_ROWS (a
    # candidate arm; its default is PAIRWISE_MAX_ROWS, the old test).
    comptime assert (
        IDN_HDB_SPARSE_MIN_ROWS <= PAIRWISE_MAX_ROWS
        and IDN_HDB_SPARSE_MIN_ROWS >= 2
    ), "MOJOLEARN_IDN_HDB_SPARSE_MIN_ROWS must be in 2..46340"
    var use_sparse = graph == MR_GRAPH_SPARSE or (
        graph == MR_GRAPH_AUTO
        and m > IDN_HDB_SPARSE_MIN_ROWS
        and not use_fast
        and (m > PAIRWISE_MAX_ROWS or not trace.enabled)
    )
    if graph != MR_GRAPH_AUTO and graph != MR_GRAPH_DENSE and graph != MR_GRAPH_SPARSE:
        raise Error(
            "hdbscan.build_mr_linkage: graph=" + String(graph)
            + " refused by name; 0 (auto), 1 (dense) or 2 (sparse)"
        )
    if m > PAIRWISE_MAX_ROWS and not use_fast and not use_sparse:
        raise Error(
            "hdbscan.build_mr_linkage: n_rows=" + String(m) + " > "
            + String(PAIRWISE_MAX_ROWS)
            + "; the dense mutual reachability graph is m * m cells and"
            " hierarchy's PAIRWISE connectivity refuses past that bound"
            " (their value_idx overflows). The dense arm was asked for by"
            " name; the default (graph=0) takes the sparse arm here"
            " (DEVIATION 1620)"
        )
    if metric != DISTANCE_L2_SQRT_EXPANDED:
        raise Error(
            "hdbscan.build_mr_linkage: metric=" + String(metric)
            + " refused by name; Currently only L2 expanded distance is"
            " supported (their RAFT_EXPECTS, reachability.cuh:109)"
        )
    # `:53-55` alpha is "weight applied when internal distance is chosen
    # for mutual reachability (value of 1.0 disables the weighting)".
    # Their `:222` passes `(value_t)1.0 / alpha` down to the functor, so
    # alpha == 0 is a division by zero they never guard. Refused by name:
    # an infinite multiplier would make every mutual reachability infinite
    # and DEVIATION 1607 would then refuse the whole matrix with a message
    # about the matrix rather than about the parameter.
    if not (alpha > Float32(0.0)) or alpha > Float32(3.4028234663852886e38):
        raise Error(
            "hdbscan.build_mr_linkage: alpha=" + String(alpha)
            + " refused by name; alpha must be finite and strictly"
            " positive (1.0 disables the weighting, their default). Their"
            " reachability.cuh:222 forms 1.0 / alpha with no guard"
        )

    # `:64-79` mutual_reachability_graph, DEVIATION 1600's two halves.
    #
    # Half one: the k-NN and the core distances, which ARE theirs.
    # MOJOLEARN_STAGE_TIMES=1 (a diagnostic, lane cluster-apple3): drain and
    # print the wall time of each part. Off, nothing changes.
    var st_on = getenv("MOJOLEARN_STAGE_TIMES") == "1"
    var st_t = Int(perf_counter_ns())
    # fg-tsne-dbscan H2 (IDN_HDB_MST_SEED_KNN): the k-NN one neighbour wider
    # on the sparse arm, so round 1 can resolve points from their lists
    var knn_w = min_samples
    comptime if IDN_HDB_MST_SEED_KNN:
        if (
            use_sparse
            and m > min_samples + 1
            and alpha <= Float32(1.0)
            and sabotage == HDB_SAB_NONE
            and not trace.enabled
        ):
            knn_w = min_samples + 1
    var knn_cells = m * knn_w
    # TOMBSTONE: MOJOLEARN_HDB_CORE_TILE (DROP) deleted 2026-10-09 by lane/owed-deletions-D3; code recoverable at b639a2bd2.
    # Tried: one-cell k-NN outputs when the tiled core kernel ran.
    # Restore: git apply experiments/removed/MOJOLEARN_HDB_CORE_TILE.patch; record in docs/TOMBSTONES.md.
    var knn_dists = ctx.enqueue_create_buffer[DType.float32](knn_cells)
    var knn_inds = ctx.enqueue_create_buffer[DType.int32](knn_cells)
    compute_core_dists(
        ctx, trace, x, core_dists, m, n, metric, min_samples,
        knn_dists, knn_inds, core_tpb, sabotage, knn_w,
    )
    trace.record_device[DType.float32](ctx, "hdbscan.core_dists", core_dists, m)
    if st_on:
        ctx.synchronize()
        var now = Int(perf_counter_ns())
        print("HDB_STAGE core_dists_ms=" + String(Float64(now - st_t) / 1.0e6))
        st_t = now

    # Half two: the DENSE graph in place of their sparse COO. `indptr`,
    # `indices` and `pw_dists` are the PAIRWISE connectivity
    # (`connectivities.cuh:110-204`) that `hierarchy` already gates, with
    # `indptr[i] = i * m`, `indices[i*m + j] = j` and the diagonal at
    # FLT_MAX; `pairwise_distances` also runs DEVIATION 623's NaN refusal
    # on the matrix before it returns.
    var nnz = 1 if (use_fast or use_sparse) else m * m
    var indptr = ctx.enqueue_create_buffer[DType.int32](m + 1)
    # the column of cell e is e % m: the solver computes it (DENSE), so the
    # m * m index array is neither written nor read (lane/cluster-apple)
    var indices = ctx.enqueue_create_buffer[DType.int32](1)
    var pw_dists = ctx.enqueue_create_buffer[DType.float32](nnz)
    var norms = ctx.enqueue_create_buffer[DType.float32](m)
    if not use_fast and not use_sparse:
        # fam2-cluster, IDN_HDB_MR_FUSED_GUARD: the NaN count of this matrix
        # is taken inside the mutual-reachability pass below.
        comptime if IDN_HDB_MR_FUSED_GUARD:
            pairwise_distances(
                ctx, x, m, n, metric, indptr, indices, pw_dists, norms,
                tile_tpb, LINK_SAB_NONE, fill_indices=False, nan_guard=False,
            )
        else:
            pairwise_distances(
                ctx, x, m, n, metric, indptr, indices, pw_dists, norms,
                tile_tpb, LINK_SAB_NONE, fill_indices=False,
            )

    # `reachability.cuh:222` `(value_t)1.0 / alpha`, on the host as
    # theirs is, through `identical_div` (row 49's seam). At the shipped
    # alpha = 1.0 the quotient is exactly 1.0 in both modes.
    var inv_alpha = identical_div(Float32(1.0), alpha)
    # IN PLACE over the distances (a view: one thread reads cell idx of
    # `pw_dists` and writes cell idx of `mr`, nothing else reads it after),
    # one m * m allocation instead of two (lane/cluster-apple)
    var mr = pw_dists.create_sub_buffer[DType.float32](0, nnz)
    if not use_fast and not use_sparse:
        comptime if IDN_HDB_MR_FUSED_GUARD:
            mutual_reachability_dense_guarded(
                ctx, mr, pw_dists, core_dists, m, inv_alpha,
                "hdbscan.build_mr_linkage", mr_tpb, sabotage,
            )
        else:
            mutual_reachability_dense(
                ctx, mr, pw_dists, core_dists, m, inv_alpha, mr_tpb, sabotage
            )
            refuse_nonfinite_device(
                ctx, mr, nnz, "hdbscan.build_mr_linkage",
                "mutual reachability cells", sabotage,
            )
        trace.record_device[DType.float32](ctx, "hdbscan.mr.dists", mr, nnz)

    # `:81-102` color, then build_sorted_mst. The reduction op and the
    # metric are arguments of the FIX-UP LOOP only; the graph here is
    # complete, so Boruvka returns one component on the first call and the
    # loop body is never entered. If it ever were,
    # `hierarchy/impl/cluster/detail/mst.mojo::connect_knn_graph` raises
    # BY NAME rather than pretending, which is what this lane wants.
    var color = ctx.enqueue_create_buffer[DType.int32](m)
    var rounds: Int
    if use_sparse:
        # DEVIATION 1620: the same tree, sorted and oriented, with no graph.
        if knn_w > min_samples:
            # H2: round 1 seeded from the wider k-NN rows
            rounds = sparse_mr_mst_device(
                ctx, x, core_dists, m, n, inv_alpha, mst_rows, mst_cols,
                mst_weights, sabotage,
                knn_d_addr=Int(knn_dists.unsafe_ptr()),
                knn_i_addr=Int(knn_inds.unsafe_ptr()),
                knn_k=knn_w,
            )
        else:
            rounds = sparse_mr_mst_device(
                ctx, x, core_dists, m, n, inv_alpha, mst_rows, mst_cols,
                mst_weights, sabotage,
            )
    elif use_fast:
        # lane af-hdbscan2 (-D MOJOLEARN_HDB_DEV_BORUVKA): the same search
        # kernels, the rounds driven on the device (fast_mr_mst_device.mojo).
        comptime if HDB_DEV_BORUVKA:
            rounds = fast_mr_mst_device(
                ctx, x, core_dists, m, n, inv_alpha, mst_rows, mst_cols,
                mst_weights,
            )
        else:
            rounds = fast_euclidean_mst(
                ctx, x, m, n, True, mst_rows, mst_cols, mst_weights,
                mutual_reach=True,
                core_ptr=core_dists.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
                inv_alpha=inv_alpha,
            )
    else:
        rounds = build_sorted_mst[DENSE=True](
            ctx, indptr, indices, mr, m, n,
            mst_rows, mst_cols, mst_weights, color, nnz,
            max_iter=10, mst_tpb=mst_tpb, sabotage=LINK_SAB_NONE,
        )

    if st_on:
        ctx.synchronize()
        var now = Int(perf_counter_ns())
        print("HDB_STAGE mst_ms=" + String(Float64(now - st_t) / 1.0e6) + " rounds=" + String(rounds))
        st_t = now
    var n_edges = m - 1
    var orient_grid = max(1, (n_edges + ORIENT_TPB - 1) // ORIENT_TPB)
    # ==================================================================
    # DEVIATION 1614. THE MST EDGE ORIENTATION IS CANONICALIZED TO
    # (min(u, v), max(u, v)). THEIRS IS BORUVKA'S, AND IS NOT A RULE.
    # ==================================================================
    # THEIRS. agglomerative.cuh:134-150 puts find(src) in the left slot and
    # find(dst) in the right. The only stage between the solver and that loop
    # is coo_sort_by_weight (mst.cuh:337-338, sort.h:94-102, a
    # thrust::sort_by_key on the weights), which reorders rows and reorients
    # nothing. So `src` is whatever min_edge_per_supervertex stored, and that
    # kernel writes temp_src[tid] = tid with the mutual-add tie broken on a
    # COLOR comparison, not a vertex one.
    #
    # THERE IS NOTHING TO MATCH HERE. The reference colors come from a round whose
    # min-edge tie is a cuRAND draw (DEVIATION 620) feeding a sort documented
    # unstable (DEVIATION 621), so their condensed-tree numbering varies run
    # to run on one GPU. An artifact is not a rule, and we cannot reproduce
    # one. We therefore CHOOSE, and record the choice here.
    #
    # OURS. Left is the lower vertex index. condense.cuh:156-160 numbers with
    # next_label++ in left-then-right order, so this makes the condensed
    # tree's numbering, and therefore the cluster NUMBERS in labels_, a pure
    # function of the MST edge SET, exactly as DEVIATION 621 made the edge
    # LIST a pure function of the graph.
    #
    # THE PARTITION IS UNCHANGED EITHER WAY. condense.cuh:137-197 case 1 only
    # swaps which sibling takes the smaller next_label; case 2 emits the same
    # leaf edges with the same parent, lambda and size, reordered, which
    # DEVIATION 1611's (parent, child) sort restores; cases 3 and 4 select on
    # which count is small, never on a slot. Downstream, select.mojo's
    # reverse loop needs only child id > parent id, true under either order,
    # and its subtree fold has two terms, so commuting them is bit-exact.
    # What changes is that the numbering stops depending on Boruvka's colors.
    #
    # SCOPE. hierarchy/ is NOT touched. Its dendrogram keeps Boruvka's
    # orientation, which it documents as inert for its own labels and which
    # its gate compares as an UNORDERED pair
    # (linkage_check.mojo::_children_pairs_equal). HDBSCAN's condense READS
    # the orientation, so it is inert there and load bearing here.
    #
    # MEASUREMENT. HDB_SAB_MST_ORIENT_RAW skips this loop and MUST FAIL
    # check_condensed_tree_vs_oracle on blobs96.
    # ==================================================================
    if sabotage != HDB_SAB_MST_ORIENT_RAW:
        ctx.enqueue_function[_orient_edges_kernel](
            mst_rows.unsafe_ptr(),
            mst_cols.unsafe_ptr(),
            Int32(n_edges),
            grid_dim=(orient_grid, 1, 1),
            block_dim=(ORIENT_TPB, 1, 1),
        )

    var rounds_list = List[Int32]()
    rounds_list.append(Int32(rounds))
    trace.record_list_i32("hdbscan.mst.rounds", rounds_list)
    if trace.enabled:
        var edges = ctx.enqueue_create_buffer[DType.int32](max(1, n_edges * 2))
        ctx.enqueue_function[_interleave_edges_kernel](
            mst_rows.unsafe_ptr(),
            mst_cols.unsafe_ptr(),
            edges.unsafe_ptr(),
            Int32(n_edges),
            grid_dim=(orient_grid, 1, 1),
            block_dim=(ORIENT_TPB, 1, 1),
        )
        trace.record_device[DType.int32](
            ctx, "hdbscan.mst.edges", edges, n_edges * 2
        )
        _ = edges^
    trace.record_device[DType.float32](
        ctx, "hdbscan.mst.weights", mst_weights, n_edges
    )

    if st_on:
        ctx.synchronize()
        var now = Int(perf_counter_ns())
        print("HDB_STAGE edges_ms=" + String(Float64(now - st_t) / 1.0e6))
        st_t = now
    # `:107-117` Perform hierarchical labeling, on the device.
    # lane af-hdbscan2 (-D MOJOLEARN_HDB_LINKAGE_DEVICE): the same three
    # outputs with one lock-free union launch per level and no flag
    # readback (dendrogram_union.mojo).
    comptime if HDB_LINKAGE_DEVICE:
        build_dendrogram_union(
            ctx, mst_rows, mst_cols, mst_weights, n_edges,
            out_dendrogram, out_distances, out_sizes,
        )
    elif IDN_DENDRO_UNION:
        # fam2-cluster: the same union under IDENTICAL on every vendor, with
        # one wait before its scratch is released.
        build_dendrogram_union(
            ctx, mst_rows, mst_cols, mst_weights, n_edges,
            out_dendrogram, out_distances, out_sizes, drain=True,
        )
    else:
        build_dendrogram_device(
            ctx, mst_rows, mst_cols, mst_weights, n_edges,
            out_dendrogram, out_distances, out_sizes,
        )
    trace.record_device[DType.int32](
        ctx, "hdbscan.dendrogram.children", out_dendrogram, n_edges * 2
    )
    trace.record_device[DType.float32](
        ctx, "hdbscan.dendrogram.deltas", out_distances, n_edges
    )
    trace.record_device[DType.int32](
        ctx, "hdbscan.dendrogram.sizes", out_sizes, n_edges
    )

    if st_on:
        ctx.synchronize()
        var now = Int(perf_counter_ns())
        print("HDB_STAGE dendrogram_ms=" + String(Float64(now - st_t) / 1.0e6))
    _ = knn_dists^
    _ = knn_inds^
    _ = indptr^
    _ = indices^
    _ = mr^
    _ = pw_dists^
    _ = norms^
    _ = color^
    return rounds
