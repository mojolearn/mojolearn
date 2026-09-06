# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`build_mr_linkage`: the linkage in mutual reachability space.

PORT OF `cuvs/cpp/src/cluster/detail/single_linkage.cuh::build_mr_linkage`
(`:50-118`) at cuVS `94c2819`, the function `hierarchy/NOT_IMPLEMENTED.tsv` line
8 names as "HDBSCAN's linkage (core distances, mutual reachability), not
single linkage's ... it is the ROADMAP's Phase 1 and would reuse this
lane's mst_solver.mojo and agglomerative.mojo unchanged."

THAT CLAIM HELD. Nothing in `hierarchy/impl/sparse/solver/`,
`hierarchy/impl/cluster/detail/agglomerative.mojo`,
`hierarchy/impl/sparse/op/sort.mojo` or
`hierarchy/checks/edge_order.mojo` was changed, copied or re-derived
for this lane; they are IMPORTED. The one thing the claim did not say,
and that this lane found, is that the GRAPH is not carried over: theirs
is a sparse k-NN COO whose MST is a forest, and the fix-up it needs is
the part `hierarchy` records as NOT PORTED. See DEVIATION 1600 in
`hdbscan/checks/mutual_reachability_dense.mojo`.

WHAT THEIR FUNCTION DOES, STEP FOR STEP (`:62-117`), AND WHAT OURS DOES
  `:64-79`   `mutual_reachability_graph(...)` -> indptr, core_dists, COO
             OURS: `compute_core_dists` (their `compute_knn` +
             `core_distances`, unchanged) then the DENSE transform,
             DEVIATION 1600. Their `mutual_reachability_indptr` is
             `hierarchy`'s dense CSR `indptr[i] = i * m`.
  `:81-84`   `color`, `MutualReachabilityFixConnectivitiesRedOp`
             OURS: `color` unchanged; the reduction op is an argument of
             the FIX-UP LOOP ONLY (`build_sorted_mst`'s `connect_knn_
             graph`), which a complete graph never enters, so it is not
             ported. `hdbscan/NOT_IMPLEMENTED.tsv` has the row.
  `:88-102`  `build_sorted_mst(...)` with `nnz = mr_coo.nnz`
             OURS: `hierarchy/impl/cluster/detail/mst.mojo::
             build_sorted_mst`, unchanged, with `nnz = m * m`.
  `:107-117` `build_dendrogram_host(...)`
             OURS: `hierarchy/impl/cluster/detail/agglomerative.mojo`,
             unchanged.

THE SABOTAGE ARGUMENT IS NOT FORWARDED INTO `hierarchy/`. This lane's
`HDB_SAB_*` constants and that lane's `LINK_SAB_*` constants are two
independent numberings that share small integers, so every call into
`hierarchy` below passes `LINK_SAB_NONE` EXPLICITLY. Forwarding would
silently select a distance-tile sabotage whenever this lane asked for a
condense sabotage, which is the kind of defect a green suite does not
show.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import IdentityTrace
from hdbscan.checks.hdbscan_sabotage import HDB_SAB_MST_ORIENT_RAW, HDB_SAB_NONE
from hdbscan.checks.mutual_reachability_dense import (
    MR_TPB,
    mutual_reachability_dense,
    refuse_nonfinite_device,
)
from hdbscan.impl.hdbscan.detail.reachability import (
    CORE_TPB,
    compute_core_dists,
)
from hierarchy.checks.edge_order import LINK_SAB_NONE, edge_hi, edge_lo
from hierarchy.impl.cluster.detail.agglomerative import build_dendrogram_host
from hierarchy.impl.cluster.detail.connectivities import (
    DISTANCE_L2_SQRT_EXPANDED,
    PAIRWISE_MAX_ROWS,
    pairwise_distances,
)
from hierarchy.impl.cluster.detail.mst import build_sorted_mst
from checks.numerics import identical_div
from neighbors.checks.pinned_distance_tile import PINNED_TILE_TPB


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
) raises -> Int:
    """`single_linkage.cuh:50-118`. Returns the Boruvka round count.

    `x_host` is the SAME data as `x`, on the host, because this tree's
    k-NN entry takes host pointers (see `reachability.mojo`'s header) and
    the dense distance step takes a device buffer. The caller owns both
    and this function copies neither.
    """
    if m < 2:
        raise Error(
            "hdbscan.build_mr_linkage: n_rows=" + String(m)
            + " < 2; a dendrogram needs at least two points"
        )
    if m > PAIRWISE_MAX_ROWS:
        raise Error(
            "hdbscan.build_mr_linkage: n_rows=" + String(m) + " > "
            + String(PAIRWISE_MAX_ROWS)
            + "; the dense mutual reachability graph is m * m cells and"
            " hierarchy's PAIRWISE connectivity refuses past that bound"
            " (their value_idx overflows). To close this refusal, port the"
            " SPARSE arm (DEVIATION 1600), which needs the cross-component"
            " fix-up in the hierarchy lane and a k-NN distance epilogue in"
            " the neighbors lane"
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
    var knn_dists = ctx.enqueue_create_buffer[DType.float32](m * min_samples)
    var knn_inds = ctx.enqueue_create_buffer[DType.int32](m * min_samples)
    compute_core_dists(
        ctx, trace, x_host, core_dists, m, n, metric, min_samples,
        knn_dists, knn_inds, core_tpb, sabotage,
    )
    trace.record_device[DType.float32](ctx, "hdbscan.core_dists", core_dists, m)

    # Half two: the DENSE graph in place of their sparse COO. `indptr`,
    # `indices` and `pw_dists` are the PAIRWISE connectivity
    # (`connectivities.cuh:110-204`) that `hierarchy` already gates, with
    # `indptr[i] = i * m`, `indices[i*m + j] = j` and the diagonal at
    # FLT_MAX; `pairwise_distances` also runs DEVIATION 623's NaN refusal
    # on the matrix before it returns.
    var nnz = m * m
    var indptr = ctx.enqueue_create_buffer[DType.int32](m + 1)
    var indices = ctx.enqueue_create_buffer[DType.int32](nnz)
    var pw_dists = ctx.enqueue_create_buffer[DType.float32](nnz)
    var norms = ctx.enqueue_create_buffer[DType.float32](m)
    pairwise_distances(
        ctx, x, m, n, metric, indptr, indices, pw_dists, norms,
        tile_tpb, LINK_SAB_NONE,
    )

    # `reachability.cuh:222` `(value_t)1.0 / alpha`, on the host as
    # theirs is, through `identical_div` (row 49's seam). At the shipped
    # alpha = 1.0 the quotient is exactly 1.0 in both modes.
    var inv_alpha = identical_div(Float32(1.0), alpha)
    var mr = ctx.enqueue_create_buffer[DType.float32](nnz)
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
    var rounds = build_sorted_mst(
        ctx, indptr, indices, mr, m, n,
        mst_rows, mst_cols, mst_weights, color, nnz,
        max_iter=10, mst_tpb=mst_tpb, sabotage=LINK_SAB_NONE,
    )

    var n_edges = m - 1
    var h_src = ctx.enqueue_create_host_buffer[DType.int32](n_edges)
    var h_dst = ctx.enqueue_create_host_buffer[DType.int32](n_edges)
    ctx.synchronize()
    var v_src = mst_rows.create_sub_buffer[DType.int32](0, n_edges)
    var v_dst = mst_cols.create_sub_buffer[DType.int32](0, n_edges)
    ctx.enqueue_copy(dst_ptr=h_src.unsafe_ptr(), src_buf=v_src)
    ctx.enqueue_copy(dst_ptr=h_dst.unsafe_ptr(), src_buf=v_dst)
    ctx.synchronize()
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
    # THERE IS NOTHING TO PORT HERE. Their colors come from a round whose
    # min-edge tie is a cuRAND draw (DEVIATION 620) feeding a sort documented
    # unstable (DEVIATION 621), so their condensed-tree numbering varies run
    # to run on one GPU. An artifact is not a rule, and we cannot transcribe
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
        for i in range(n_edges):
            var cu = h_src.unsafe_ptr().unsafe_load(i)
            var cv = h_dst.unsafe_ptr().unsafe_load(i)
            h_src.unsafe_ptr().unsafe_store(i, edge_lo(cu, cv))
            h_dst.unsafe_ptr().unsafe_store(i, edge_hi(cu, cv))
        ctx.enqueue_copy(dst_buf=v_src, src_ptr=h_src.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=v_dst, src_ptr=h_dst.unsafe_ptr())
        ctx.synchronize()

    var edges = List[Int32](capacity=n_edges * 2)
    for i in range(n_edges):
        edges.append(h_src.unsafe_ptr().unsafe_load(i))
        edges.append(h_dst.unsafe_ptr().unsafe_load(i))
    var rounds_list = List[Int32]()
    rounds_list.append(Int32(rounds))
    trace.record_list_i32("hdbscan.mst.rounds", rounds_list)
    trace.record_list_i32("hdbscan.mst.edges", edges)
    trace.record_device[DType.float32](
        ctx, "hdbscan.mst.weights", mst_weights, n_edges
    )

    # `:107-117` Perform hierarchical labeling.
    build_dendrogram_host(
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

    _ = knn_dists^
    _ = knn_inds^
    _ = indptr^
    _ = indices^
    _ = pw_dists^
    _ = norms^
    _ = mr^
    _ = color^
    _ = h_src^
    _ = h_dst^
    _ = v_src^
    _ = v_dst^
    return rounds
