# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Derivative work: the upstream file and its pinned commit are recorded in this lane's DERIVATION_MAP.tsv and in this file's own docstring. See NOTICE.
"""IVF-FLAT's search: coarse select, probe, select again.

PORT OF `cuvs/src/neighbors/ivf_flat/ivf_flat_search.cuh` at cuVS
`6ba2ce2`: `search_impl` (`:40-306`) and `search_with_filtering`
(`:311-374`). Partial, and one of their two kernels is REFUSED rather than
ported.

THEIR FIVE STEPS
-----------------

| # | theirs | line | ours |
|---|---|---|---|
| 1 | query norms + `outer_add` + `gemm` = query-to-centroid distances | `:109-162` | the tiled k-NN arm's two kernels, mode-dispatched (below) |
| 2 | `cuvs::selection::select_k` picks the `n_probes` nearest lists | `:180-188` | `select_radix_identical` / `select_radix`, key `(distance, list id)` |
| 3 | `calc_chunk_indices` = the segmented scan of probed list sizes | `:219-223` | `ivf/derived/neighbors/ivf_common.mojo::calc_chunk_indices` |
| 4 | `ivfflat_interleaved_scan` scores the candidates and keeps a local top-k | `:248-266` | **REFUSED, DEVIATION 1785**, replaced (below) |
| 5 | `select_k` again over the per-block results, then `postprocess_neighbors` | `:275-303` | one selection over the candidate row, then the carry lookup |

STEP 4, AND WHY IT IS A REFUSAL
--------------------------------
`ivfflat_interleaved_scan` cannot be ported into the identical column, and
the reasons are three separate rows of `IDENTITY_PATHS.md` at once.

  - **It is a warp-sort queue on a numeric path.** Its local top-k is
    `raft::matrix::detail::select::warpsort` over `kSubwarpSize =
    min(Capacity, WarpSize)` lanes
    (`ivf_flat_interleaved_scan_jit.cuh:189-192`). That is row 23's
    refusal verbatim -- a bitonic network whose width is the hardware's
    lane count, so it is 32 lanes on Apple and NVIDIA and 64 on AMD's
    wavefront, and its comparator resolves an equidistant tie by the
    queue's feed order.
  - **Its grid width is occupancy-derived and it is a summation
    membership.** `search_impl` calls the scan ONCE with null pointers
    purely to read `grid_dim_x` back (`:191-215`), then splits the probes
    across that many blocks and merges with a second `select_k`. A block
    count that comes from the device is row 3's and row 7's class, and here
    it decides which candidates are compared against which.
  - **Its data layout is the interleaved group** this port does not build
    (DEVIATION 1782), and `veclen` is chosen from `dim` by
    `calculate_veclen` (`ivf_flat_index.cpp:36`).

WHAT REPLACES IT
-----------------
The candidates of one query are gathered into a contiguous row (ascending
by carried original index, `ivf/original/list_layout.mojo`), and that row
goes through THE SAME TWO KERNELS the tiled brute-force k-NN arm uses:

  - distances: `neighbors/original/pinned_distance_tile.mojo` under
    `IDENTICAL` (DEVIATION 505 -- one thread per cell, the feature axis
    walked ascending through `identical_mul_add`, no vendor matmul and no
    k-split), `core/gemm.mojo::gemm_nt` + `core/expand_distances.mojo`
    under `FAST`. This file WRITES NEITHER; the dispatch below is a copy of
    `knn_brute_force.mojo`'s at `:170-205` and is cited as such.
  - selection: `neighbors/original/select_radix_identical.mojo` under
    `IDENTICAL` (DEVIATIONS 500/501 -- the composite `(distance, index)`
    key and the ranked placement), the ported `select_radix` under `FAST`.

**This is DEVIATION 509's choice of arm, inherited.** Under `IDENTICAL` the
k-NN lane pins AUTO to the TILED arm on every column, because that is the
arm whose tie set is NAMED and which contains no warp primitive at all.
An IVF search whose inner loop used a different arm than the brute force it
must reduce to could not have the reduction gate, so there was never a
second option here.

THE CANDIDATE ROW'S POSITION ORDER IS THE TIE-BREAK
----------------------------------------------------
The identical selector keys on `(twiddle_in(distance) << 32) | POSITION IN
THE ROW` and takes no index array. So the row's position order decides
every equidistant tie. `merge_probed_lists` orders it by CARRIED ORIGINAL
INDEX (DEVIATION 1786), which makes the key `(distance, original index)`
restricted to the candidates -- the same total order brute force uses -- and
which is why `n_probe == n_lists` reduces to brute force exactly rather
than approximately. Read `ivf/original/list_layout.mojo`'s header before
changing anything about that order.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from cluster.derived.cluster.detail.kmeans_common import metric_is_sqrt
from core.expand_distances import expand_distances_kernel
from core.gemm import gemm_nt
from core.identity_trace import IdentityTrace
from ivf.original.list_layout import (
    ListLayout,
    gather_candidate_indices,
    gather_candidate_norms,
    gather_candidate_vectors,
    merge_probed_lists,
)
from ivf.derived.neighbors.ivf_common import (
    calc_chunk_indices,
    n_samples_from_chunks,
    postprocess_neighbors,
    postprocess_distances_is_identity,
)
from ivf.derived.neighbors.ivf_flat.ivf_flat_build import (
    compute_row_norms,
    download_f32,
    download_u32,
    upload_f32,
)
from ivf.derived.neighbors.ivf_flat.ivf_flat_index import (
    IvfFlatIndex,
    IvfFlatSearchParams,
    ivf_metric_name,
    ivf_search_params_validate,
    ivf_validate_data,
)
from original.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    PIN_DETERMINISM,
    numeric_mode_name,
)
from neighbors.original.pinned_distance_tile import (
    PINNED_TILE_TPB,
    pinned_distance_tile_kernel,
)
from neighbors.original.select_radix_identical import (
    radix_topk_identical_kernel,
)
from neighbors.derived.matrix.detail.select_radix import (
    SELECT_BLOCK,
    radix_topk_one_block_kernel,
)


comptime IVF_EXPAND_TPB = 256
"""SCHEDULING. The elementwise epilogue's threads per block on the FAST
arm, matching the 256 `knn_brute_force.mojo:200` launches
`expand_distances_kernel` with. One thread per output cell, so the block
width reaches no fold and no accumulator; `check_launch_invariance` moves
it anyway, because "reaches no fold" is an argument and the check is a
measurement."""


@fieldwise_init
struct IvfSearchResult(Movable):
    """`n_queries x k` distances and ORIGINAL row ids, plus the candidate
    count per query.

    `n_candidates` is returned rather than kept private for the same reason
    `knn_search` returns the query tile it used: it is the number that says
    HOW MUCH of the index this search actually looked at, and a recall
    report that cannot state it is a recall report about nothing.
    """

    var distances: List[Float32]
    var indices: List[UInt32]
    var n_candidates: List[Int32]


def _expanded_distances(
    ctx: DeviceContext,
    mut z: DeviceBuffer[DType.float32],
    mut q: DeviceBuffer[DType.float32],
    q_row_offset: Int,
    mut y: DeviceBuffer[DType.float32],
    mut q_norm: DeviceBuffer[DType.float32],
    mut y_norm: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    d: Int,
    is_sqrt: Bool,
    tile_tpb: Int,
    expand_tpb: Int,
) raises:
    """`z[m x n] = ||q_i||^2 + ||y_j||^2 - 2 q_i . y_j`, mode-dispatched.

    A COPY OF `tiled_brute_force_knn`'s dispatch
    (`neighbors/derived/neighbors/detail/knn_brute_force.mojo:170-205`),
    with the query row offset threaded through because this file calls it
    once for the whole query set (against the centroids) and once per query
    (against that query's candidates). NEITHER KERNEL IS WRITTEN HERE.

    Under `IDENTICAL` the pinned tile is one thread per cell and the
    summation order is a pure function of `d`, so `m` and `n` do not enter
    the arithmetic at all -- which is what lets this be called at `m =
    n_queries` for the coarse step and at `m = 1` for the candidate step
    and still be the same function. Under `FAST` it is a vendor matmul,
    whose tile shape and k-split are per-shape, so those two calls are NOT
    the same function and `ivf/README.md` says so where it matters.
    """
    var cells = m * n
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        ctx.enqueue_function[pinned_distance_tile_kernel](
            z.unsafe_ptr(),
            q.unsafe_ptr().unsafe_offset(q_row_offset * d),
            y.unsafe_ptr(),
            q_norm.unsafe_ptr().unsafe_offset(q_row_offset),
            y_norm.unsafe_ptr(),
            Int32(m),
            Int32(n),
            Int32(d),
            Int32(1 if is_sqrt else 0),
            grid_dim=((cells + tile_tpb - 1) // tile_tpb, 1, 1),
            block_dim=(tile_tpb, 1, 1),
        )
    else:
        # EXACT SUB-BUFFERS ON ALL THREE OPERANDS. The workspaces here are
        # allocated once at the worst-case candidate count and used short,
        # and MAX's matmul takes a `TileTensor` over a whole buffer.
        var qv = q.create_sub_buffer[DType.float32](q_row_offset * d, m * d)
        var yv = y.create_sub_buffer[DType.float32](0, n * d)
        var zv = z.create_sub_buffer[DType.float32](0, cells)
        gemm_nt(ctx, zv, qv, yv, m, n, d)
        ctx.enqueue_function[expand_distances_kernel](
            z.unsafe_ptr(),
            q_norm.unsafe_ptr().unsafe_offset(q_row_offset),
            y_norm.unsafe_ptr(),
            Int32(m),
            Int32(n),
            Int32(1 if is_sqrt else 0),
            grid_dim=((cells + expand_tpb - 1) // expand_tpb, 1, 1),
            block_dim=(expand_tpb, 1, 1),
        )


def _select_top_k(
    ctx: DeviceContext,
    mut in_val: DeviceBuffer[DType.float32],
    mut out_val: DeviceBuffer[DType.float32],
    mut out_idx: DeviceBuffer[DType.uint32],
    mut buf_val: DeviceBuffer[DType.float32],
    mut buf_idx: DeviceBuffer[DType.uint32],
    n_rows: Int,
    length: Int,
    k: Int,
    buf_len: Int,
) raises:
    """The smallest `k` of each row, mode-dispatched. NOT WRITTEN HERE.

    `radix_topk_identical_kernel` under `IDENTICAL`: eight passes over the
    64-bit `(twiddle_in(distance) << 32) | position` key, then the rank pass
    that writes each winner to its rank rather than to an atomic slot
    (DEVIATIONS 500/501). `radix_topk_one_block_kernel` under `FAST`: RAFT's
    own selector, tie back-fill and atomic placement included, because
    fixing a thing upstream does not do is an improvement and improvements
    do not live in `derived/`.

    `k > SELECT_BLOCK` is REFUSED IN BOTH MODES, which is
    `UNSUPERVISED_IDENTITY.md`'s owed item 5 reaching this lane unchanged:
    the identical selector's rank pass gives one thread to each output slot
    and a larger k needs a loop nobody has written. Refused at the launcher,
    not silent.
    """
    if k > SELECT_BLOCK:
        raise Error(
            "ivf_flat: a selection of k = "
            + String(k)
            + " exceeds SELECT_BLOCK ("
            + String(SELECT_BLOCK)
            + "). The identical selector's rank pass gives one thread to"
            " each output slot; a larger k needs a loop, which is"
            " UNSUPERVISED_IDENTITY.md's owed item 5 and is not written"
            " until something asks for it."
        )
    if k > length:
        raise Error(
            "ivf_flat: a selection of k = "
            + String(k)
            + " over a row of "
            + String(length)
            + " elements. The ported radix selector cannot take k > len --"
            " no bucket satisfies `prev_count < k <= cur_count`, every"
            " later pass drops every element, and last_filter reads a"
            " buffer nothing wrote (knn_brute_force.mojo's own note)."
        )
    comptime if PIN_DETERMINISM:
        # Ledger row 11 (DEVIATIONS 500/501), and `PIN_DETERMINISM`
        # since 2026-08-29. The FAST arm is RAFT's own selector with its
        # tie back-fill and ATOMIC PLACEMENT; this one writes each winner
        # to its rank rather than to an atomic slot. Atomic placement is
        # a run-to-run property, so the middle tier needs this pin, and
        # this is the same cause as `knn_brute_force.mojo`'s row-11
        # branch. The `k > SELECT_BLOCK` and `k > length` raises above
        # sit OUTSIDE this block and already bound both modes, so
        # re-keying adds no new refusal -- only the eight radix passes.
        #
        # `:180` in this file is the row-24 DISTANCE dispatch and stays
        # keyed to identical: a vendor matmul's k-split is per-vendor,
        # not per-run.
        ctx.enqueue_function[radix_topk_identical_kernel](
            in_val.unsafe_ptr(),
            out_val.unsafe_ptr(),
            out_idx.unsafe_ptr(),
            buf_val.unsafe_ptr(),
            buf_idx.unsafe_ptr(),
            Int32(length),
            Int32(k),
            Int32(buf_len),
            Int32(1),
            grid_dim=(n_rows, 1, 1),
            block_dim=(SELECT_BLOCK, 1, 1),
        )
    else:
        ctx.enqueue_function[radix_topk_one_block_kernel](
            in_val.unsafe_ptr(),
            out_val.unsafe_ptr(),
            out_idx.unsafe_ptr(),
            buf_val.unsafe_ptr(),
            buf_idx.unsafe_ptr(),
            Int32(length),
            Int32(k),
            Int32(buf_len),
            Int32(1),
            grid_dim=(n_rows, 1, 1),
            block_dim=(SELECT_BLOCK, 1, 1),
        )


def sort_slots_by_distance_then_index(
    mut dist: List[Float32], mut idx: List[UInt32], base: Int, k: Int
):
    """Insertion sort of `k` slots on the TOTAL order `(distance, index)`.

    `neighbors/estimator.mojo`'s host sort, spelled again because that one
    is not exported and sorts a runtime host buffer rather than a `List`.
    An exported helper there would delete this function; that file is
    another lane's, so `ivf/README.md`'s WHAT IS OWED names it instead of
    this lane editing it.

    IT IS A CORRECTNESS REQUIREMENT AND NOT A COURTESY, twice over.
    (1) Under `FAST` the ported selector does not sort at all -- RAFT's
    radix select returns the right `k` in an unspecified order -- so
    without this the card's `ivf.out_*` stages and the returned arrays
    would carry an atomic arrival order. (2) scikit-learn's `kneighbors`
    returns neighbours ascending, so a drop-in has to.

    Under `IDENTICAL` the selector already emits this exact order
    (DEVIATION 501's rank pass), so this pass is a no-op there -- and it is
    still run, because a sort that is a no-op is cheaper than a mode-
    dependent output contract.
    """
    for a in range(1, k):
        var dv = dist[base + a]
        var iv = idx[base + a]
        var b = a - 1
        while b >= 0:
            var db = dist[base + b]
            var ib = idx[base + b]
            if db < dv or (db == dv and ib <= iv):
                break
            dist[base + b + 1] = db
            idx[base + b + 1] = ib
            b -= 1
        dist[base + b + 1] = dv
        idx[base + b + 1] = iv


def ivf_flat_search_traced(
    ctx: DeviceContext,
    mut trace: IdentityTrace,
    index: IvfFlatIndex,
    sp: IvfFlatSearchParams,
    queries: List[Float32],
    n_queries: Int,
    k: Int,
    tile_tpb: Int = PINNED_TILE_TPB,
    expand_tpb: Int = IVF_EXPAND_TPB,
) raises -> IvfSearchResult:
    """`ivf_flat::search`, `ivf_flat_search.cuh:311-374` then `:40-306`.

    Row-major `queries` of `n_queries x dim`. Returns `n_queries x k`
    distances and ORIGINAL row ids, ascending by `(distance, index)`.

    THEIR BATCHING HEURISTIC IS NOT PORTED (`:343-353`): `max_queries` comes
    from `get_workspace_free_bytes`, which is a device memory number, and a
    number that decides how the query set is cut is a number this lane must
    not take from the hardware. Every query is served in one pass here.
    `check_launch_invariance` gates that a query answered alone and the
    same query answered inside a batch agree bit for bit, which is the
    property their heuristic would have had to preserve and never stated.

    STAGES RECORDED (the tags, in order):

        ivf.query_norm      [n_queries], the squared query norms
        ivf.coarse_dist     [n_queries, n_lists], query-to-centroid
        ivf.probe_dist      [n_queries, n_probes], sorted ascending
        ivf.probe_lists     [n_queries, n_probes], the chosen LIST IDS
        ivf.cand_counts     [n_queries], the candidate count per query
        ivf.cand_idx        the candidates' ORIGINAL ids, all queries
                            concatenated in query order
        ivf.cand_dist       the candidate distances, same order
        ivf.out_dist        [n_queries, k]
        ivf.out_idx         [n_queries, k]

    Every tag names a position in the algorithm and none carries a block
    count, a grid width or a device property (`core/identity_trace.mojo`
    rule 2). `cand_counts` is a function of the data and the parameters
    only, which is why the two ragged stages after it can be compared as
    flat arrays at all.
    """
    ivf_search_params_validate(sp, index.n_lists, n_queries, k)
    ivf_validate_data(queries, n_queries, index.dim, "queries")
    _ = postprocess_distances_is_identity(index.metric)

    var dim = index.dim
    var n_lists = index.n_lists
    var n_probes = sp.n_probes
    var is_sqrt = metric_is_sqrt(index.metric)

    if n_probes > SELECT_BLOCK:
        raise Error(
            "ivf_flat search: n_probes ("
            + String(n_probes)
            + ") exceeds SELECT_BLOCK ("
            + String(SELECT_BLOCK)
            + "); the coarse selection runs through the same selector as"
            " the final one and inherits its refusal."
        )

    if trace.enabled:
        trace.header(
            String("ivf_flat search: n_queries=")
            + String(n_queries)
            + " dim="
            + String(dim)
            + " n_lists="
            + String(n_lists)
            + " n_probes="
            + String(n_probes)
            + " k="
            + String(k)
            + " n_rows="
            + String(index.n_rows)
            + " metric="
            + ivf_metric_name(index.metric)
        )

    # ---- upload -------------------------------------------------------
    var dq = upload_f32(ctx, queries)
    var dcenters = upload_f32(ctx, index.centers)
    var dcenter_norm = upload_f32(ctx, index.center_norms)
    var dlist_data = upload_f32(ctx, index.list_data)
    var dq_norm = ctx.enqueue_create_buffer[DType.float32](n_queries)
    var dlist_norm = ctx.enqueue_create_buffer[DType.float32](index.n_rows)
    ctx.synchronize()

    compute_row_norms(ctx, dq, dq_norm, n_queries, dim, is_sqrt)
    # THE CANDIDATE NORMS ARE COMPUTED OVER `list_data`, NOT OVER THE
    # ORIGINAL ROWS, AND THAT IS BIT-EXACT RATHER THAN CLOSE.
    # `row_norm_kernel` is one block per row reading only that row, so
    # permuting the rows permutes the outputs and changes no float. This is
    # what lets `check_nprobe_equals_nlists_is_brute_force` compare against
    # a `knn_search` whose norms were taken over the unpermuted matrix.
    compute_row_norms(ctx, dlist_data, dlist_norm, index.n_rows, dim, is_sqrt)
    ctx.synchronize()
    if trace.enabled:
        trace.record_device(ctx, "ivf.query_norm", dq_norm, n_queries)

    # ---- step 1: query-to-centroid distances ---------------------------
    var dcoarse = ctx.enqueue_create_buffer[DType.float32](n_queries * n_lists)
    ctx.synchronize()
    _expanded_distances(
        ctx, dcoarse, dq, 0, dcenters, dq_norm, dcenter_norm,
        n_queries, n_lists, dim, is_sqrt, tile_tpb, expand_tpb,
    )
    ctx.synchronize()
    if trace.enabled:
        trace.record_device(
            ctx, "ivf.coarse_dist", dcoarse, n_queries * n_lists
        )

    # ---- step 2: the n_probes nearest lists ----------------------------
    #
    # THE TIE RULE IS THE SELECTOR'S KEY AND IS NOT DECIDED HERE
    # (DEVIATION 1788). The row being selected over is indexed BY LIST ID,
    # so the composite key's low half IS the list id and a query
    # equidistant from two centroids probes the LOWER-NUMBERED list first.
    # `check_assignment_ties` gates the same rule on the build side, where
    # it comes from a different mechanism (`raft::argmin_op`) and has to
    # agree.
    var probe_buf_len = n_lists // 8
    if probe_buf_len < n_probes:
        probe_buf_len = n_probes
    var dprobe_dist = ctx.enqueue_create_buffer[DType.float32](
        n_queries * n_probes
    )
    var dprobe_idx = ctx.enqueue_create_buffer[DType.uint32](
        n_queries * n_probes
    )
    var dpbuf_val = ctx.enqueue_create_buffer[DType.float32](
        n_queries * 2 * probe_buf_len
    )
    var dpbuf_idx = ctx.enqueue_create_buffer[DType.uint32](
        n_queries * 2 * probe_buf_len
    )
    ctx.synchronize()
    _select_top_k(
        ctx, dcoarse, dprobe_dist, dprobe_idx, dpbuf_val, dpbuf_idx,
        n_queries, n_lists, n_probes, probe_buf_len,
    )
    ctx.synchronize()

    var probe_dist = download_f32(ctx, dprobe_dist, n_queries * n_probes)
    var probe_ids = download_u32(ctx, dprobe_idx, n_queries * n_probes)
    for q in range(n_queries):
        sort_slots_by_distance_then_index(
            probe_dist, probe_ids, q * n_probes, n_probes
        )
    if trace.enabled:
        trace.record_list_f32("ivf.probe_dist", probe_dist)
        var probe_i32 = List[Int32]()
        for i in range(n_queries * n_probes):
            probe_i32.append(Int32(probe_ids[i]))
        trace.record_list_i32("ivf.probe_lists", probe_i32)

    # ---- steps 3-5: the candidates of each query -----------------------
    var layout = ListLayout(
        n_lists,
        index.n_rows,
        dim,
        index.list_offsets.copy(),
        index.list_indices.copy(),
        index.list_data.copy(),
    )
    var list_sizes = List[Int32]()
    for l in range(n_lists):
        list_sizes.append(Int32(index.list_size(l)))
    var list_norm = download_f32(ctx, dlist_norm, index.n_rows)

    # ONE ALLOCATION AT THE WORST CASE, REUSED. The worst case is
    # `n_probes == n_lists`, where every vector is a candidate; anything
    # smaller uses the head of the same buffers. The alternative is an
    # allocation per query, which would make the workspace a function of
    # the query and give `check_launch_invariance` a second thing to move.
    var max_cand = index.n_rows
    var cand_buf_len = max_cand // 8
    if cand_buf_len < k:
        cand_buf_len = k
    var dcand_vec = ctx.enqueue_create_buffer[DType.float32](max_cand * dim)
    var dcand_norm = ctx.enqueue_create_buffer[DType.float32](max_cand)
    var dcand_dist = ctx.enqueue_create_buffer[DType.float32](max_cand)
    var dcbuf_val = ctx.enqueue_create_buffer[DType.float32](2 * cand_buf_len)
    var dcbuf_idx = ctx.enqueue_create_buffer[DType.uint32](2 * cand_buf_len)
    var dsel_val = ctx.enqueue_create_buffer[DType.float32](k)
    var dsel_idx = ctx.enqueue_create_buffer[DType.uint32](k)
    var hcand_vec = ctx.enqueue_create_host_buffer[DType.float32](
        max_cand * dim
    )
    var hcand_norm = ctx.enqueue_create_host_buffer[DType.float32](max_cand)
    ctx.synchronize()

    var out_dist = List[Float32]()
    var out_idx = List[UInt32]()
    var cand_counts = List[Int32]()
    var all_cand_idx = List[Int32]()
    var all_cand_dist = List[Float32]()

    for q in range(n_queries):
        var this_probe = List[UInt32]()
        for p in range(n_probes):
            this_probe.append(probe_ids[q * n_probes + p])

        var chunks = calc_chunk_indices(list_sizes, this_probe, n_probes)
        var n_cand = n_samples_from_chunks(chunks, n_probes)
        cand_counts.append(Int32(n_cand))
        if n_cand < k:
            # DEVIATION 1794. Their `postprocess_neighbors_kernel` fills
            # the short slots with `kOutOfBoundsRecord`
            # (`ivf_common.cuh:106-108`); that fill is not ported and the
            # ported selector cannot take `k > len` at all. Refusing names
            # the two numbers a caller can act on.
            raise Error(
                "ivf_flat search: query "
                + String(q)
                + " probes "
                + String(n_probes)
                + " lists holding "
                + String(n_cand)
                + " vectors between them, fewer than k = "
                + String(k)
                + ". Their kOutOfBoundsRecord short-fill"
                " (ivf_common.cuh:106-108) is not ported. Raise n_probes,"
                " or lower k, or rebuild with fewer lists."
            )

        var slots = merge_probed_lists(layout, this_probe, n_probes)
        var cand_vec = gather_candidate_vectors(layout, slots)
        var cand_orig = gather_candidate_indices(layout, slots)
        var cand_norm = gather_candidate_norms(slots, list_norm)

        for i in range(n_cand * dim):
            hcand_vec.unsafe_ptr().unsafe_store(i, cand_vec[i])
        for i in range(n_cand):
            hcand_norm.unsafe_ptr().unsafe_store(i, cand_norm[i])
        ctx.enqueue_copy(dst_buf=dcand_vec, src_ptr=hcand_vec.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=dcand_norm, src_ptr=hcand_norm.unsafe_ptr())
        ctx.synchronize()

        _expanded_distances(
            ctx, dcand_dist, dq, q, dcand_vec, dq_norm, dcand_norm,
            1, n_cand, dim, is_sqrt, tile_tpb, expand_tpb,
        )
        ctx.synchronize()

        _select_top_k(
            ctx, dcand_dist, dsel_val, dsel_idx, dcbuf_val, dcbuf_idx,
            1, n_cand, k, cand_buf_len,
        )
        ctx.synchronize()

        var sel_dist = download_f32(ctx, dsel_val, k)
        var sel_pos = download_u32(ctx, dsel_idx, k)
        var sel_orig = postprocess_neighbors(sel_pos, cand_orig, k)
        sort_slots_by_distance_then_index(sel_dist, sel_orig, 0, k)

        for i in range(k):
            out_dist.append(sel_dist[i])
            out_idx.append(sel_orig[i])
        if trace.enabled:
            var row = download_f32(ctx, dcand_dist, n_cand)
            for i in range(n_cand):
                all_cand_idx.append(Int32(cand_orig[i]))
                all_cand_dist.append(row[i])

    if trace.enabled:
        trace.record_list_i32("ivf.cand_counts", cand_counts)
        trace.record_list_i32("ivf.cand_idx", all_cand_idx)
        trace.record_list_f32("ivf.cand_dist", all_cand_dist)
        trace.record_list_f32("ivf.out_dist", out_dist)
        var out_i32 = List[Int32]()
        for i in range(n_queries * k):
            out_i32.append(Int32(out_idx[i]))
        trace.record_list_i32("ivf.out_idx", out_i32)

    _ = dq^
    _ = dcenters^
    _ = dcenter_norm^
    _ = dlist_data^
    _ = dq_norm^
    _ = dlist_norm^
    _ = dcoarse^
    _ = dprobe_dist^
    _ = dprobe_idx^
    _ = dpbuf_val^
    _ = dpbuf_idx^
    _ = dcand_vec^
    _ = dcand_norm^
    _ = dcand_dist^
    _ = dcbuf_val^
    _ = dcbuf_idx^
    _ = dsel_val^
    _ = dsel_idx^
    _ = hcand_vec^
    _ = hcand_norm^

    return IvfSearchResult(out_dist^, out_idx^, cand_counts^)
