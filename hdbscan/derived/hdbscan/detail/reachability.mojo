# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Core distances from the k-NN graph, and the mutual reachability space.

PORT OF `cuml-v26.08.00/cpp/src/hdbscan/detail/reachability.cuh`
(cuML `265b9da`) and, for the parts cuML 26.08 has already delegated to
cuVS, `cuvs/cpp/src/neighbors/detail/reachability.cuh` (cuVS `94c2819`).
Both files carry the SAME `core_distances` and `compute_knn`, which is
why one Mojo file stands for both; `hdbscan/DERIVATION_MAP.tsv` records the
pair. Transliterated, their order. Do not improve.

WHAT IS HERE
  `core_distances`      `reachability.cuh:49-63` (cuVS) / `:42-56` (cuML)
  `compute_knn`         `:80-94` (cuVS) / `:73-117` (cuML)
  `_compute_core_dists` `:100-122` (cuVS) / `:123-145` (cuML)

WHAT IS REFUSED BY NAME
  `mutual_reachability_knn_l2`  `:151-188`
  `mutual_reachability_graph`   `:190-256`
both for DEVIATION 1600's two reasons, which are written out in full in
`hdbscan/original/mutual_reachability_dense.mojo`'s block: the sparse
graph's MST is a forest and the fix-up (`connect_knn_graph`,
`cross_component_nn`, `merge_msts`) is NOT PORTED in `hierarchy/`; and the
`DistanceEpilogue` template `mutual_reachability_knn_l2` needs does not
exist on `neighbors/derived/neighbors/detail/knn_brute_force.mojo`.

WHICH k-NN, AND WHY IT MATTERS MORE HERE THAN ANYWHERE ELSE
------------------------------------------------------------
Theirs calls `tiled_brute_force_knn(..., dists, inds, metric)` whose
`cuvs::selection::select_k(..., select_min, true)` -- the trailing `true`
is `sorted` (`knn_brute_force.cuh:265-276` and `:307-318`) -- so the k
neighbours arrive ASCENDING BY DISTANCE and `core_distances` reads the
last column of a sorted row.

Ours calls `neighbors/estimator.mojo::knn_search_traced`, this tree's
k-NN entry, which under IDENTICAL runs the TILED arm (DEVIATION 509),
takes its distances from `neighbors/original/pinned_distance_tile.mojo`
(DEVIATION 505), ranks on the composite `(distance, index)` radix key
(DEVIATION 500) and then host-sorts the k slots on `(distance, index)`.
That is theirs plus an index tie-break, and the tie-break is DEVIATION
1602 below.

WHERE IT RUNS. Theirs takes device pointers; `knn_search_traced`'s
boundary is host pointers (`MutUntrackedOrigin`), so `compute_knn` below
copies through host buffers. This moves the boundary, not the algorithm:
`metrics/derived/stats/detail/trustworthiness_score.mojo:193-214` reaches
the same entry the same way and says so for the same reason.

======================================================================
DEVIATION BLOCK -- DEVIATION 1602. THE CORE DISTANCE IS A k-TH ORDER
STATISTIC, AND THE NEIGHBOUR IT COMES FROM IS NAMED.
======================================================================

WHAT THE HAZARD IS. `core_distances` returns
`knn_dists[row * n_neighbors + (min_samples - 1)]`: the k-th smallest
distance from row `row`. When two neighbours of `row` are EQUIDISTANT at
the k-th position -- which is not a corner case in a clustering library,
it is duplicate points and grid data -- the VALUE of that k-th distance
is well defined but WHICH NEIGHBOUR SUPPLIED IT is not. Upstream's answer
comes from `select_k`'s comparator, which compares the distance only
(IDENTITY_PATHS row 11), so the INDEX is arrival order.

WHAT THAT DOES AND DOES NOT COST, PRECISELY, BECAUSE THE TWO ARE
DIFFERENT QUESTIONS.

  (a) THE VALUE. `core_dists[row]` is the k-th smallest element of a
      multiset of `m` distances. A multiset's k-th smallest is a pure
      function of the multiset, so it does not depend on which of several
      equal elements was selected, on the order they were visited, or on
      the selector. **Under IDENTICAL the core distance VALUE is
      bit-identical whenever the distance MATRIX is**, and the distance
      matrix is pinned by DEVIATION 505 and gated in `neighbors/`. That
      is the claim `check_core_distances_vs_oracle` asserts, per cell,
      bit for bit, against a host oracle that computes the same k-th
      order statistic by a DIFFERENT route (a full serial sort of the
      row, not a top-k).
  (b) THE INDEX. `core_distances` does not read the index and this lane
      never does either, so the tie has no path to an output through this
      function. It is recorded as a stage (`knn.sorted_idx`, written by
      `knn_search_traced` itself) so that a card diff can SEE a tie
      resolving differently and dismiss it, rather than mistaking it for
      a numeric divergence.
  (c) WHAT MAKES (a) TRUE AND NOT MERELY PLAUSIBLE. It needs the selector
      to return the k SMALLEST as a SET, which is what a top-k is, and
      the k-th slot to be the k-th smallest, which needs the row SORTED.
      `knn_search_traced` sorts on `(distance, index)`, a total order, so
      slot `k-1` holds the k-th smallest distance on every vendor.
      Without that sort the last SLOT would be an arbitrary member of the
      set and (a) would be false. `neighbors/`'s own measurement is the
      warning: the tiled arm returns the right k neighbours in an
      UNSPECIFIED order (157 of 320 slots "wrong" ordered, 0 wrong as a
      set), and `knn_search`'s host sort is what repairs it.

WHERE THE GUARANTEE STOPS, and this lane may not assume it away.
`UNSUPERVISED_IDENTITY.md` records that under **FAST** three consecutive
runs of one binary on one device returned three different sorted INDEX
sets, and that two COLUMNS diverge at `knn.out_dist` with every sorted
distance equal. So under FAST (b) is not reproducible run to run, and (a)
is only as reproducible as the vendor matmul underneath it. Every
distance-derived gate in this lane is therefore an ASSERTION under
IDENTICAL and a REPORT under FAST, exactly as `hierarchy`'s are.
======================================================================
"""

from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from core.identity_trace import IdentityTrace
from hdbscan.original.hdbscan_sabotage import (
    HDB_SAB_CORE_KTH_PLUS_ONE,
    HDB_SAB_NONE,
)
from hdbscan.original.mutual_reachability_dense import refuse_nonfinite_device
from hierarchy.derived.cluster.detail.connectivities import (
    DISTANCE_L2_SQRT_EXPANDED,
)
from neighbors.estimator import knn_search_traced


comptime CORE_TPB = 256
"""Their `int tpb = 256` template default (`reachability.cuh:49`).
SCHEDULING: one thread per row, one load, one store."""


def core_distances_kernel(
    out_core: MutPointer[Float32, MutAnyOrigin],
    knn_dists: MutPointer[Float32, MutAnyOrigin],
    min_samples_in: Int32,
    n_neighbors_in: Int32,
    n_in: Int32,
    sabotage: Int32,
):
    """`reachability.cuh:60-62`, the `thrust::transform` body:

        return knn_dists[row * n_neighbors + (min_samples - 1)];

    One thread per row, one load, one store. No fold, no atomic, no lane
    primitive, so the answer cannot see the launch shape.
    """
    var row = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if row >= Int(n_in):
        return
    var n_neighbors = Int(n_neighbors_in)
    var slot = Int(min_samples_in) - 1
    if sabotage == HDB_SAB_CORE_KTH_PLUS_ONE:
        slot = Int(min_samples_in)
    out_core.unsafe_store(row, knn_dists.unsafe_load(row * n_neighbors + slot))


def core_distances(
    ctx: DeviceContext,
    mut knn_dists: DeviceBuffer[DType.float32],
    min_samples: Int,
    n_neighbors: Int,
    n: Int,
    mut out: DeviceBuffer[DType.float32],
    core_tpb: Int = CORE_TPB,
    sabotage: Int32 = HDB_SAB_NONE,
) raises:
    """`reachability.cuh:49-63`. `knn_dists` is `n x n_neighbors`, SORTED
    ascending by `(distance, index)`; `out` is `n`."""
    # `:53-54` ASSERT(n_neighbors >= min_samples, ...), their wording.
    if n_neighbors < min_samples:
        raise Error(
            "hdbscan.core_distances: the size of the neighborhood should be"
            " greater than or equal to min_samples (n_neighbors="
            + String(n_neighbors) + ", min_samples=" + String(min_samples) + ")"
        )
    if min_samples < 1:
        raise Error(
            "hdbscan.core_distances: min_samples=" + String(min_samples)
            + " < 1 refused by name; the k-th order statistic is undefined"
            " below k = 1 and their slice would index at -1"
        )
    var blocks = (n + core_tpb - 1) // core_tpb if n > 0 else 1
    ctx.enqueue_function[core_distances_kernel](
        out.unsafe_ptr(),
        knn_dists.unsafe_ptr(),
        Int32(min_samples),
        Int32(n_neighbors),
        Int32(n),
        sabotage,
        grid_dim=(blocks, 1, 1),
        block_dim=(core_tpb, 1, 1),
    )
    ctx.synchronize()


def compute_knn(
    ctx: DeviceContext,
    mut trace: IdentityTrace,
    mut x_host: List[Float32],
    m: Int,
    n: Int,
    k: Int,
    mut out_dists: DeviceBuffer[DType.float32],
    mut out_inds: DeviceBuffer[DType.int32],
) raises:
    """`reachability.cuh:80-94`: "perform knn", index and queries both X.

    Theirs converts FAISS's `int64_t` indices down to `value_idx` in a
    `thrust::transform` (`cuml .../reachability.cuh:111-116`); ours
    converts `knn_search`'s `UInt32` to `Int32` in the same host pass that
    moves the result to the device, which is the same narrowing at the
    same point in the pipeline.

    `trace` is threaded in rather than constructed here so the k-NN's own
    `knn.*` stages land in the SAME card as this lane's, with one
    increasing `seq` (`core/identity_trace.mojo`'s uniqueness invariant;
    the DEVIATION 518 lesson, restated at `neighbors/estimator.mojo:190`).
    """
    var h_dist = ctx.enqueue_create_host_buffer[DType.float32](m * k)
    var h_idx = ctx.enqueue_create_host_buffer[DType.uint32](m * k)
    var h_x = ctx.enqueue_create_host_buffer[DType.float32](m * n)
    ctx.synchronize()
    for i in range(m * n):
        h_x.unsafe_ptr().unsafe_store(i, x_host[i])
    _ = knn_search_traced(
        ctx,
        trace,
        h_x.unsafe_ptr(),
        m,
        h_x.unsafe_ptr(),
        m,
        n,
        k,
        h_dist.unsafe_ptr(),
        h_idx.unsafe_ptr(),
        True,
    )
    var i32 = List[Int32](capacity=m * k)
    for i in range(m * k):
        i32.append(Int32(Int(h_idx.unsafe_ptr().unsafe_load(i))))
    ctx.enqueue_copy(dst_buf=out_dists, src_ptr=h_dist.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=out_inds, src_ptr=i32.unsafe_ptr())
    ctx.synchronize()
    _ = h_dist^
    _ = h_idx^
    _ = h_x^
    _ = i32^


def compute_core_dists(
    ctx: DeviceContext,
    mut trace: IdentityTrace,
    mut x_host: List[Float32],
    mut core_dists: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    metric: Int,
    min_samples: Int,
    mut knn_dists: DeviceBuffer[DType.float32],
    mut knn_inds: DeviceBuffer[DType.int32],
    core_tpb: Int = CORE_TPB,
    sabotage: Int32 = HDB_SAB_NONE,
) raises:
    """`_compute_core_dists`, `reachability.cuh:100-122`. Their name has a
    leading underscore because it is their CPU/GPU interop entry; the
    underscore is dropped here because Mojo has no such convention and
    `hdbscan/DERIVATION_MAP.tsv` records the rename.

    `knn_dists` (`m x min_samples`) and `knn_inds` are the caller's, so
    the driver can record them as stages; theirs are function-local
    `rmm::device_uvector`s (`:114-115`).
    """
    # `:109-110` RAFT_EXPECTS(metric == L2SqrtExpanded, "Currently only L2
    # expanded distance is supported"), their wording and their only
    # accepted metric.
    if metric != DISTANCE_L2_SQRT_EXPANDED:
        raise Error(
            "hdbscan.compute_core_dists: metric=" + String(metric)
            + " refused by name; Currently only L2 expanded distance is"
            " supported (their RAFT_EXPECTS at reachability.cuh:109). To"
            " close this refusal, port the metric into"
            " hierarchy/derived/cluster/detail/connectivities.mojo's"
            " distance step first, because the dense mutual reachability"
            " graph reads that matrix"
        )
    compute_knn(ctx, trace, x_host, m, n, min_samples, knn_dists, knn_inds)
    # `:121` Slice core distances (distances to kth nearest neighbor)
    core_distances(
        ctx, knn_dists, min_samples, min_samples, m, core_dists,
        core_tpb, sabotage,
    )
    # NOT THEIRS. DEVIATION 1607: the core distances come off a different
    # code path (`neighbors/`) from the dense matrix, so `hierarchy`'s
    # DEVIATION 623 guard does not cover them.
    refuse_nonfinite_device(
        ctx, core_dists, m, "hdbscan.compute_core_dists", "core distances",
        sabotage,
    )


def mutual_reachability_knn_l2() raises:
    """`reachability.cuh:151-188`. NOT PORTED; raises by name."""
    raise Error(
        "hdbscan.mutual_reachability_knn_l2: NOT PORTED (DEVIATION 1600)."
        " It runs cuvs::neighbors::detail::tiled_brute_force_knn with the"
        " ReachabilityPostProcess epilogue so the k-selection happens IN"
        " mutual reachability space, and"
        " neighbors/derived/neighbors/detail/knn_brute_force.mojo carries no"
        " DistanceEpilogue template parameter. The epilogue is not"
        " optional: max(core[col], max(core[row], alpha*d)) is not monotone"
        " in d alone, so the mutual-reachability neighbours are not the"
        " plain neighbours reordered. To close this refusal, the neighbors"
        " lane threads an epilogue through the tiled arm; until then the"
        " dense arm (DEVIATION 1600) is the graph"
    )


def mutual_reachability_graph(n_components_hint: Int) raises:
    """`reachability.cuh:190-256`. NOT PORTED; raises by name."""
    raise Error(
        "hdbscan.mutual_reachability_graph: NOT PORTED (DEVIATION 1600)."
        " Their sparse graph is a symmetrized min_samples-nearest-neighbour"
        " COO, which is DISCONNECTED whenever the data has more than one"
        " well-separated cluster -- the case HDBSCAN exists for -- so"
        " build_sorted_mst returns a forest and enters its fix-up loop."
        " That loop needs connect_knn_graph / cross_component_nn /"
        " merge_msts, which hierarchy/DERIVATION_MAP.tsv records as NOT PORTED"
        " and hierarchy/derived/cluster/detail/mst.mojo raises by name. It"
        " also needs mutual_reachability_knn_l2, refused above. To close"
        " this refusal, port the cross-component fix-up in the hierarchy"
        " lane (pinning the host overload's std::mt19937 random vertex"
        " choice, mst.cuh:167-190) and the k-NN epilogue in the neighbors"
        " lane; until then use the dense arm, which is the complete graph"
        " and cannot be disconnected (hint: "
        + String(n_components_hint) + " components)"
    )
