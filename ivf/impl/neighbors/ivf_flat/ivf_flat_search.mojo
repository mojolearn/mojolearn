# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""IVF-FLAT's search: coarse select, probe, select again.

Reference: `search_impl` (`:40-306`) and `search_with_filtering`
(`:311-374`), `cuvs/src/neighbors/ivf_flat/ivf_flat_search.cuh` (cuVS
`6ba2ce2`). Partial, and one of the reference's two kernels is REFUSED rather than
implemented.

THE REFERENCE'S FIVE STEPS
--------------------------

| # | reference | line | here |
|---|---|---|---|
| 1 | query norms + `outer_add` + `gemm` = query-to-centroid distances | `:109-162` | the tiled k-NN arm's two kernels, mode-dispatched (below) |
| 2 | `cuvs::selection::select_k` picks the `n_probes` nearest lists | `:180-188` | `select_radix_identical` / `select_radix`, key `(distance, list id)` |
| 3 | `calc_chunk_indices` = the segmented scan of probed list sizes | `:219-223` | `ivf/impl/neighbors/ivf_common.mojo::calc_chunk_indices` |
| 4 | `ivfflat_interleaved_scan` scores the candidates and keeps a local top-k | `:248-266` | **REFUSED, DEVIATION 1785**, replaced (below) |
| 5 | `select_k` again over the per-block results, then `postprocess_neighbors` | `:275-303` | one selection over the candidate row, then the carry lookup |

STEP 4, AND WHY IT IS A REFUSAL
--------------------------------
`ivfflat_interleaved_scan` cannot be implemented into the identical column, and
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
  - **Its data layout is the interleaved group** this implementation does not build
    (DEVIATION 1782), and `veclen` is chosen from `dim` by
    `calculate_veclen` (`ivf_flat_index.cpp:36`).

WHAT REPLACES IT
-----------------
The candidates of one query are gathered into a contiguous row (ascending
by carried original index, `ivf/checks/list_layout.mojo`), and that row
goes through THE SAME TWO KERNELS the tiled brute-force k-NN arm uses:

  - distances: `neighbors/checks/pinned_distance_tile.mojo` under
    `IDENTICAL` (DEVIATION 505 -- one thread per cell, the feature axis
    walked ascending through `identical_mul_add`, no vendor matmul and no
    k-split), `core/gemm.mojo::gemm_nt` + `core/expand_distances.mojo`
    under `FAST`. This file WRITES NEITHER; the dispatch below is a copy of
    `knn_brute_force.mojo`'s at `:170-205` and is cited as such.
  - selection: `neighbors/checks/select_radix_identical.mojo` under
    `IDENTICAL` (DEVIATIONS 500/501 -- the composite `(distance, index)`
    key and the ranked placement), the implemented `select_radix` under `FAST`.

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
than approximately. Read `ivf/checks/list_layout.mojo`'s header before
changing anything about that order.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from core.expand_distances import expand_distances_kernel
from core.gemm import gemm_nt
from core.identity_trace import IdentityTrace
from ivf.impl.neighbors.ivf_flat.identical_ivf_scan import (
    IIVF_MAX_DIM,
    IIVF_QPB,
    identical_ivf_scan_kernel,
    identical_ivf_scan_staged_kernel,
    identical_ivf_scan_grouped_kernel,
    identical_ivf_merge_kernel,
    GQPB,
)
from ivf.impl.neighbors.ivf_flat.fast_ivf_scan import (
    FIVF_MAX_DIM,
    FIVF_QPB,
    fast_ivf_scan_kernel,
)
from std.sys.compile import is_defined
from std.os import getenv
from x_ann.io import upload_i32
from std.gpu import WARP_SIZE
from std.sys.info import has_apple_gpu_accelerator
from x_ann.switches import ANN3_PREPARE
from ivf.checks.list_layout import (
    ListLayout,
    filter_candidate_slots,
    gather_candidate_indices,
    gather_candidate_norms,
    gather_candidate_vectors,
    merge_probed_lists,
)
from ivf.impl.neighbors.ivf_common import (
    calc_chunk_indices,
    n_samples_from_chunks,
    postprocess_neighbors,
    postprocess_distances,
    postprocess_distances_is_identity,
)
from ivf.impl.neighbors.ivf_flat.ivf_flat_build import (
    compute_row_norms,
    download_f32,
    download_u32,
    upload_f32,
)
from ivf.impl.neighbors.ivf_flat.ivf_flat_index import (
    IvfFlatIndex,
    IvfFlatSearchParams,
    ivf_metric_name,
    ivf_search_params_validate,
    ivf_validate_data,
)
from checks.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_FAST,
    NUMERIC_IDENTICAL,
    PIN_DETERMINISM,
    numeric_mode_name,
)
from neighbors.checks.pinned_distance_tile import (
    PINNED_TILE_TPB,
    pinned_distance_tile_kernel,
)
from neighbors.checks.select_radix_identical import (
    radix_topk_identical_kernel,
    IDENTICAL_MAX_K,
)
from neighbors.impl.matrix.detail.select_radix import (
    SELECT_BLOCK,
    radix_topk_one_block_kernel,
)


comptime IVF_SELECT_LIMIT = IDENTICAL_MAX_K if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL else SELECT_BLOCK

comptime IVF_EXPAND_TPB = 256
"""SCHEDULING. The elementwise epilogue's threads per block on the FAST
arm, matching the 256 `knn_brute_force.mojo:200` launches
`expand_distances_kernel` with. One thread per output cell, so the block
width reaches no fold and no accumulator; `check_launch_invariance` moves
it anyway, because "reaches no fold" is an argument and the check is a
measurement."""

comptime IVF_IDENTICAL_SCAN = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and not is_defined["MOJOLEARN_IVF_IDENTICAL_SCAN_OFF"]()
)
"""lane/neural-net-experiment (2026-09-30, the classical pass): EVERY
vendor, not Apple alone. The Apple gate left NVIDIA and AMD on the host
round trip per query: 526 s on an L40S and 265 s on an MI325X for
`classical2/ivf` on istella (4,000 queries), against 4.9 s on an M3 Ultra
running this kernel (bench_board 0.8.25). The kernel is written on
WARP_SIZE (its launch below and its lane merge follow it), and its
arithmetic is the pinned path's term for term; the identity gate on each
vendor is the check, as it was on Apple. `MOJOLEARN_IVF_IDENTICAL_SCAN_OFF`
restores the per-query path.

IDENTICAL on Apple (lane/apple-identical-neural, 2026-09-26): steps 3-5
for every query in one launch (`identical_ivf_scan.mojo`), the pinned
distance arithmetic and the `(distance, original index)` key, instead of a
host round trip per query. Same neighbours, same order, same bits."""

comptime IVF_FAST_SCAN = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and not is_defined["MOJOLEARN_IVF_FAST_SCAN_OFF"]()
)
"""FAST on Apple: steps 3-5 for every query in one launch
(`fast_ivf_scan.mojo`) instead of a host round trip per query."""



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




def _ivf_scan_grouped() -> Bool:
    """The identical scan grouped by list (identical_ivf_scan_grouped_kernel
    + identical_ivf_merge_kernel, lane neural-pass42): default on;
    MOJOLEARN_IVF_SCAN_GROUPED=0 keeps the per-query kernels. Same bits."""
    return String(getenv("MOJOLEARN_IVF_SCAN_GROUPED")) != "0"


def _ivf_scan_staged() -> Bool:
    """The identical scan with its candidate rows staged through threadgroup
    memory (identical_ivf_scan_staged_kernel, lane neural-pass35): the
    default on NVIDIA and AMD, where the plain kernel's lanes each walk
    their own row (WARP_SIZE scattered rows per load); the plain kernel on
    Apple, where the staged kernel's barriers cost more than they save
    (M4, 200,000 x 220, 256 lists, 2,000 queries: 3.4 -> 12.2 s). The same
    bits either way; MOJOLEARN_IVF_SCAN_STAGED=0/1 forces either."""
    var v = String(getenv("MOJOLEARN_IVF_SCAN_STAGED"))
    comptime if has_apple_gpu_accelerator():
        return v == "1"
    return v != "0"


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
    tile_tpb: Int,
    expand_tpb: Int,
) raises:
    """`z[m x n] = ||q_i||^2 + ||y_j||^2 - 2 q_i . y_j`, mode-dispatched.

    ALWAYS THE SQUARED DISTANCE: both kernels get `is_sqrt = 0`. Their
    search scores the coarse step and the candidates squared under
    `L2Expanded` and `L2SqrtExpanded` alike (`ivf_flat_search.cuh:110-162`
    has no root, and the interleaved scan roots a value only as its local
    top-k STORES it, `interleaved_scan_impl.cuh:204` with
    `tag_post_process_sqrt`, `ivf_flat_interleaved_scan_jit.cuh:279-290`).
    So both selections run on squared keys and the root is applied to the
    `k` selected distances afterwards (`postprocess_distances`).

    A COPY OF `tiled_brute_force_knn`'s dispatch
    (`neighbors/impl/detail/knn_brute_force.mojo:170-205`),
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
            Int32(0),
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
            Int32(0),
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
    fixing a thing the reference does not do is an improvement and improvements
    do not live in `impl/`.

    IDENTICAL admits k up to IDENTICAL_MAX_K using the shared selector's
    strided rank pass. Other modes retain the existing SELECT_BLOCK bound.
    See neighbors/checks/select_radix_identical.mojo and IDENTITY_PATHS.md.
    """
    if k > IVF_SELECT_LIMIT:
        raise Error("ivf_flat: selection k exceeds the mode's bounded rank capacity " + String(IVF_SELECT_LIMIT))
    if k > length:
        raise Error(
            "ivf_flat: a selection of k = "
            + String(k)
            + " over a row of "
            + String(length)
            + " elements. The implemented radix selector cannot take k > len --"
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
        # branch. The mode-specific rank capacity and k<=length guards
        # run before launch; old k retains its original shared footprint.
        #
        # `:180` in this file is the row-24 DISTANCE dispatch and stays
        # keyed to identical: a vendor matmul's k-split is per-vendor,
        # not per-run.
        if k <= SELECT_BLOCK:
            ctx.enqueue_function[radix_topk_identical_kernel[SELECT_BLOCK]](
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
            ctx.enqueue_function[radix_topk_identical_kernel[IDENTICAL_MAX_K]](
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
    (1) Under `FAST` the implemented selector does not sort at all -- RAFT's
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


struct IvfFlatDevice(Movable):
    """The search's index side, prepared ONCE (lane/py-dn-ann, 2026-09-28,
    retires DEVIATION 1804): the centroids, their norms and the list data on
    the device, the list norms `compute_row_norms` takes over `list_data`
    (downloaded once for the per-query path), the host CSR layout the
    per-query path gathers from, the list sizes and the device CSR pair the
    Apple one-launch scan reads. Every one of these was built at the top of
    every search before, from the same bytes by the same kernels, so a
    search through a prepared index returns the bits a fresh one does.
    `ivf_flat_search_traced` prepares one per call (the one-shot and traced
    doors); `ivf/resident.mojo` keeps one per handle."""

    var dcenters: DeviceBuffer[DType.float32]
    var dcenter_norm: DeviceBuffer[DType.float32]
    var dlist_data: DeviceBuffer[DType.float32]
    var dlist_norm: DeviceBuffer[DType.float32]
    var d_off: DeviceBuffer[DType.int32]
    var d_ind: DeviceBuffer[DType.uint32]
    var list_norm: List[Float32]
    var layout: ListLayout
    var list_sizes: List[Int32]

    def __init__(out self, ctx: DeviceContext, index: IvfFlatIndex) raises:
        self.dcenters = upload_f32(ctx, index.centers)
        self.dcenter_norm = upload_f32(ctx, index.center_norms)
        self.dlist_data = upload_f32(ctx, index.list_data)
        self.dlist_norm = ctx.enqueue_create_buffer[DType.float32](index.n_rows)
        # THE CANDIDATE NORMS ARE COMPUTED OVER `list_data`, NOT OVER THE
        # ORIGINAL ROWS, AND THAT IS BIT-EXACT RATHER THAN CLOSE.
        # `row_norm_kernel` is one block per row reading only that row, so
        # permuting the rows permutes the outputs and changes no float. This is
        # what lets `check_nprobe_equals_nlists_is_brute_force` compare against
        # a `knn_search` whose norms were taken over the unpermuted matrix.
        compute_row_norms(ctx, self.dlist_data, self.dlist_norm, index.n_rows, index.dim)
        self.list_norm = download_f32(ctx, self.dlist_norm, index.n_rows)
        var n_lists = index.n_lists
        var h_off = ctx.enqueue_create_host_buffer[DType.int32](n_lists + 1)
        for i in range(n_lists + 1):
            h_off.unsafe_ptr().unsafe_store(i, index.list_offsets[i])
        var h_ind = ctx.enqueue_create_host_buffer[DType.uint32](index.n_rows)
        for i in range(index.n_rows):
            h_ind.unsafe_ptr().unsafe_store(i, index.list_indices[i])
        self.d_off = ctx.enqueue_create_buffer[DType.int32](n_lists + 1)
        self.d_ind = ctx.enqueue_create_buffer[DType.uint32](index.n_rows)
        ctx.enqueue_copy(dst_buf=self.d_off, src_ptr=h_off.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=self.d_ind, src_ptr=h_ind.unsafe_ptr())
        ctx.synchronize()
        _ = h_off^
        _ = h_ind^
        # lane ann-apple3, behind `ANN3_PREPARE`: the host layout (a copy of
        # the index's three lists, the n_rows x dim vectors among them) is
        # read by the per-query path only, so it is made by `ensure_layout`
        # when a search first takes that path. The one-launch scans never
        # read it. Without the switch it is copied here, as before.
        comptime if ANN3_PREPARE:
            self.layout = ListLayout(
                n_lists, index.n_rows, index.dim, List[Int32](), List[UInt32](), List[Float32]()
            )
        else:
            self.layout = ListLayout(
                n_lists,
                index.n_rows,
                index.dim,
                index.list_offsets.copy(),
                index.list_indices.copy(),
                index.list_data.copy(),
            )
        self.list_sizes = List[Int32]()
        for l in range(n_lists):
            self.list_sizes.append(Int32(index.list_size(l)))

    def ensure_layout(mut self, index: IvfFlatIndex):
        """The host CSR layout the per-query path gathers from, copied from
        the index on first use (the same three lists `__init__` copied)."""
        if len(self.layout.offsets) == index.n_lists + 1:
            return
        self.layout = ListLayout(
            index.n_lists,
            index.n_rows,
            index.dim,
            index.list_offsets.copy(),
            index.list_indices.copy(),
            index.list_data.copy(),
        )


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
    partial_storage: Bool = False,
    keep: List[Int32] = List[Int32](),
) raises -> IvfSearchResult:
    """`ivf_flat::search` over an index prepared for this call alone
    (`IvfFlatDevice`, then `ivf_flat_search_prepared`). See the latter."""
    ivf_search_params_validate(sp, index.n_lists, n_queries, k)
    ivf_validate_data(queries, n_queries, index.dim, "queries")
    var dev = IvfFlatDevice(ctx, index)
    var r = ivf_flat_search_prepared(
        ctx, trace, index, dev, sp, queries, n_queries, k, tile_tpb, expand_tpb, partial_storage, keep
    )
    _ = dev^
    return r^


def ivf_flat_search_prepared(
    ctx: DeviceContext,
    mut trace: IdentityTrace,
    index: IvfFlatIndex,
    mut dev: IvfFlatDevice,
    sp: IvfFlatSearchParams,
    queries: List[Float32],
    n_queries: Int,
    k: Int,
    tile_tpb: Int = PINNED_TILE_TPB,
    expand_tpb: Int = IVF_EXPAND_TPB,
    partial_storage: Bool = False,
    keep: List[Int32] = List[Int32](),
) raises -> IvfSearchResult:
    """`ivf_flat::search`, `ivf_flat_search.cuh:311-374` then `:40-306`.

    `keep` is the sample filter (DEVIATION 5863, `filter_candidate_slots`):
    one int32 per ORIGINAL row id, 0 removing the row before it is scored or
    counted. Empty is no filter. A filtered search takes the per-query path
    below, never the one-launch scan, so an unfiltered search is untouched.

    Row-major `queries` of `n_queries x dim`. Returns `n_queries x k`
    distances and ORIGINAL row ids, ascending by `(distance, index)`.

    THEIR BATCHING HEURISTIC IS NOT IMPLEMENTED (`:343-353`): `max_queries` comes
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
        ivf.cand_dist       the candidate distances, same order, SQUARED
                            under both metrics (the selection key)
        ivf.out_dist        [n_queries, k], rooted under L2SqrtExpanded
        ivf.out_idx         [n_queries, k]

    Every tag names a position in the algorithm and none carries a block
    count, a grid width or a device property (`core/identity_trace.mojo`
    rule 2). `cand_counts` is a function of the data and the parameters
    only, which is why the two ragged stages after it can be compared as
    flat arrays at all.
    """
    ivf_search_params_validate(sp, index.n_lists, n_queries, k)
    ivf_validate_data(queries, n_queries, index.dim, "queries")
    var dist_is_identity = postprocess_distances_is_identity(index.metric)
    var filtered = len(keep) > 0
    if filtered and len(keep) != index.n_rows:
        raise Error(
            "ivf_flat search: the filter holds " + String(len(keep))
            + " flags for an index of " + String(index.n_rows) + " rows"
        )

    var dim = index.dim
    var n_lists = index.n_lists
    var n_probes = sp.n_probes

    if n_probes > IVF_SELECT_LIMIT:
        raise Error(
            "ivf_flat search: n_probes ("
            + String(n_probes)
            + ") exceeds the mode selection limit ("
            + String(IVF_SELECT_LIMIT)
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

    # ---- upload: the queries only (the index side is `dev`) -----------
    var dq = upload_f32(ctx, queries)
    var dq_norm = ctx.enqueue_create_buffer[DType.float32](n_queries)
    compute_row_norms(ctx, dq, dq_norm, n_queries, dim)
    if trace.enabled:
        trace.record_device(ctx, "ivf.query_norm", dq_norm, n_queries)

    # ---- step 1: query-to-centroid distances ---------------------------
    var dcoarse = ctx.enqueue_create_buffer[DType.float32](n_queries * n_lists)
    _expanded_distances(
        ctx, dcoarse, dq, 0, dev.dcenters, dq_norm, dev.dcenter_norm,
        n_queries, n_lists, dim, tile_tpb, expand_tpb,
    )
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
    _select_top_k(
        ctx, dcoarse, dprobe_dist, dprobe_idx, dpbuf_val, dpbuf_idx,
        n_queries, n_lists, n_probes, probe_buf_len,
    )
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

    # ---- steps 3-5, FAST on Apple: every query in one launch ------------
    comptime if IVF_FAST_SCAN or IVF_IDENTICAL_SCAN:
        if (
            not trace.enabled
            and not partial_storage
            and not filtered
            and k <= 32
            and dim <= FIVF_MAX_DIM
        ):
            var counts = List[Int32]()
            var enough = True
            for q in range(n_queries):
                var c = 0
                for p in range(n_probes):
                    c += index.list_size(Int(probe_ids[q * n_probes + p]))
                counts.append(Int32(c))
                if c < k:
                    enough = False
            if enough:
                var d_od = ctx.enqueue_create_buffer[DType.float32](n_queries * k)
                var d_oi = ctx.enqueue_create_buffer[DType.uint32](n_queries * k)
                var grid = (n_queries + FIVF_QPB - 1) // FIVF_QPB
                comptime for KM in [8, 16, 32]:
                    if k <= KM and (KM == 8 or k > KM // 2):
                        comptime if IVF_IDENTICAL_SCAN:
                            if _ivf_scan_grouped():
                                # lane neural-pass42: the (query, probe) pairs grouped by
                                # list on the host (a counting sort, ascending (q, p)
                                # within a list), the blocks GQPB pairs of one list each
                                var gcount = List[Int32](length=n_lists + 1, fill=Int32(0))
                                for q in range(n_queries):
                                    for p in range(n_probes):
                                        gcount[Int(probe_ids[q * n_probes + p]) + 1] += 1
                                var goff = List[Int32](length=n_lists + 1, fill=Int32(0))
                                for l in range(n_lists):
                                    goff[l + 1] = goff[l] + gcount[l + 1]
                                var cursor = List[Int32](length=n_lists, fill=Int32(0))
                                for l in range(n_lists):
                                    cursor[l] = goff[l]
                                var gq = List[Int32](length=max(n_queries * n_probes, 1), fill=Int32(0))
                                var gp = List[Int32](length=max(n_queries * n_probes, 1), fill=Int32(0))
                                for q in range(n_queries):
                                    for p in range(n_probes):
                                        var l = Int(probe_ids[q * n_probes + p])
                                        gq[Int(cursor[l])] = Int32(q)
                                        gp[Int(cursor[l])] = Int32(p)
                                        cursor[l] += 1
                                var bl = List[Int32]()
                                var bs = List[Int32]()
                                for l in range(n_lists):
                                    var j = Int(goff[l])
                                    while j < Int(goff[l + 1]):
                                        bl.append(Int32(l))
                                        bs.append(Int32(j))
                                        j += GQPB
                                var n_blocks = len(bl)
                                if n_blocks == 0:
                                    bl.append(Int32(0))
                                    bs.append(Int32(0))
                                var d_goff = upload_i32(ctx, goff)
                                var d_gq = upload_i32(ctx, gq)
                                var d_gp = upload_i32(ctx, gp)
                                var d_bl = upload_i32(ctx, bl)
                                var d_bs = upload_i32(ctx, bs)
                                var d_pd = ctx.enqueue_create_buffer[DType.float32](max(n_queries * n_probes * KM, 1))
                                var d_pi = ctx.enqueue_create_buffer[DType.uint32](max(n_queries * n_probes * KM, 1))
                                if n_blocks > 0:
                                    ctx.enqueue_function[identical_ivf_scan_grouped_kernel[KM]](
                                        dq.unsafe_ptr(), dq_norm.unsafe_ptr(),
                                        dev.dlist_data.unsafe_ptr(), dev.dlist_norm.unsafe_ptr(),
                                        dev.d_off.unsafe_ptr(), dev.d_ind.unsafe_ptr(),
                                        d_goff.unsafe_ptr(), d_gq.unsafe_ptr(), d_gp.unsafe_ptr(),
                                        d_bl.unsafe_ptr(), d_bs.unsafe_ptr(),
                                        d_pd.unsafe_ptr(), d_pi.unsafe_ptr(),
                                        Int32(dim), Int32(n_probes), Int32(k),
                                        grid_dim=n_blocks, block_dim=GQPB * WARP_SIZE,
                                    )
                                ctx.enqueue_function[identical_ivf_merge_kernel[KM]](
                                    d_pd.unsafe_ptr(), d_pi.unsafe_ptr(), d_od.unsafe_ptr(), d_oi.unsafe_ptr(),
                                    Int32(n_queries), Int32(n_probes), Int32(k),
                                    grid_dim=(n_queries + 255) // 256, block_dim=256,
                                )
                                ctx.synchronize()
                                _ = d_goff^
                                _ = d_gq^
                                _ = d_gp^
                                _ = d_bl^
                                _ = d_bs^
                                _ = d_pd^
                                _ = d_pi^
                                _ = gcount^
                                _ = goff^
                                _ = cursor^
                                _ = gq^
                                _ = gp^
                                _ = bl^
                                _ = bs^
                            elif _ivf_scan_staged():
                                ctx.enqueue_function[identical_ivf_scan_staged_kernel[KM]](
                                    dq.unsafe_ptr(), dq_norm.unsafe_ptr(),
                                    dev.dlist_data.unsafe_ptr(), dev.dlist_norm.unsafe_ptr(),
                                    dev.d_off.unsafe_ptr(), dev.d_ind.unsafe_ptr(),
                                    dprobe_idx.unsafe_ptr(),
                                    d_od.unsafe_ptr(), d_oi.unsafe_ptr(),
                                    Int32(n_queries), Int32(dim), Int32(n_probes),
                                    Int32(k),
                                    grid_dim=(n_queries + IIVF_QPB - 1) // IIVF_QPB,
                                    block_dim=IIVF_QPB * WARP_SIZE,
                                )
                            else:
                                ctx.enqueue_function[identical_ivf_scan_kernel[KM]](
                                    dq.unsafe_ptr(), dq_norm.unsafe_ptr(),
                                    dev.dlist_data.unsafe_ptr(), dev.dlist_norm.unsafe_ptr(),
                                    dev.d_off.unsafe_ptr(), dev.d_ind.unsafe_ptr(),
                                    dprobe_idx.unsafe_ptr(),
                                    d_od.unsafe_ptr(), d_oi.unsafe_ptr(),
                                    Int32(n_queries), Int32(dim), Int32(n_probes),
                                    Int32(k),
                                    grid_dim=(n_queries + IIVF_QPB - 1) // IIVF_QPB,
                                    block_dim=IIVF_QPB * WARP_SIZE,
                                )
                        else:
                            ctx.enqueue_function[fast_ivf_scan_kernel[KM]](
                                dq.unsafe_ptr(), dev.dlist_data.unsafe_ptr(),
                                dev.d_off.unsafe_ptr(), dev.d_ind.unsafe_ptr(),
                                dprobe_idx.unsafe_ptr(),
                                d_od.unsafe_ptr(), d_oi.unsafe_ptr(),
                                Int32(n_queries), Int32(dim), Int32(n_probes),
                                Int32(k),
                                grid_dim=grid, block_dim=FIVF_QPB * WARP_SIZE,
                            )
                var fd = download_f32(ctx, d_od, n_queries * k)
                var fi = download_u32(ctx, d_oi, n_queries * k)
                if not dist_is_identity:
                    postprocess_distances(fd, index.metric)
                _ = d_od^
                _ = d_oi^
                _ = dq^
                _ = dq_norm^
                _ = dcoarse^
                _ = dprobe_dist^
                _ = dprobe_idx^
                _ = dpbuf_val^
                _ = dpbuf_idx^
                return IvfSearchResult(fd^, fi^, counts^)
    # ---- steps 3-5: the candidates of each query -----------------------
    dev.ensure_layout(index)

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
    var out_dist = List[Float32]()
    var out_idx = List[UInt32]()
    var cand_counts = List[Int32]()
    var all_cand_idx = List[Int32]()
    var all_cand_dist = List[Float32]()

    for q in range(n_queries):
        var this_probe = List[UInt32]()
        for p in range(n_probes):
            this_probe.append(probe_ids[q * n_probes + p])

        var chunks = calc_chunk_indices(dev.list_sizes, this_probe, n_probes)
        var n_cand = n_samples_from_chunks(chunks, n_probes)
        var kept = List[Int32]()
        if filtered:
            kept = filter_candidate_slots(
                dev.layout, merge_probed_lists(dev.layout, this_probe, n_probes), keep
            )
            n_cand = len(kept)
        cand_counts.append(Int32(n_cand))
        if n_cand < k and not partial_storage:
            # DEVIATION 1794. Their `postprocess_neighbors_kernel` fills
            # the short slots with `kOutOfBoundsRecord`
            # (`ivf_common.cuh:106-108`); that fill is not implemented and the
            # implemented selector cannot take `k > len` at all. Refusing names
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
                + (" (after the filter)" if filtered else "")
                + ". Their kOutOfBoundsRecord short-fill"
                " (ivf_common.cuh:106-108) is not implemented. Raise n_probes,"
                " or lower k, or rebuild with fewer lists."
            )

        # Partial storage keeps global coarse probes but may own no candidates.
        # Padding has no numerical meaning; n_candidates gives min(k, count)
        # valid slots and the driver must ignore the remainder.
        var selected = min(k, n_cand)
        if selected == 0:
            for i in range(k):
                out_dist.append(Float32(0))
                out_idx.append(UInt32(0))
            continue

        var slots: List[Int32]
        if filtered:
            slots = kept.copy()
        else:
            slots = merge_probed_lists(dev.layout, this_probe, n_probes)
        var cand_vec = gather_candidate_vectors(dev.layout, slots)
        var cand_orig = gather_candidate_indices(dev.layout, slots)
        var cand_norm = gather_candidate_norms(slots, dev.list_norm)

        for i in range(n_cand * dim):
            hcand_vec.unsafe_ptr().unsafe_store(i, cand_vec[i])
        for i in range(n_cand):
            hcand_norm.unsafe_ptr().unsafe_store(i, cand_norm[i])
        ctx.enqueue_copy(dst_buf=dcand_vec, src_ptr=hcand_vec.unsafe_ptr())
        ctx.enqueue_copy(dst_buf=dcand_norm, src_ptr=hcand_norm.unsafe_ptr())
        _expanded_distances(
            ctx, dcand_dist, dq, q, dcand_vec, dq_norm, dcand_norm,
            1, n_cand, dim, tile_tpb, expand_tpb,
        )
        _select_top_k(
            ctx, dcand_dist, dsel_val, dsel_idx, dcbuf_val, dcbuf_idx,
            1, n_cand, selected, cand_buf_len,
        )
        var sel_dist = download_f32(ctx, dsel_val, selected)
        var sel_pos = download_u32(ctx, dsel_idx, selected)
        var sel_orig = postprocess_neighbors(sel_pos, cand_orig, selected)
        sort_slots_by_distance_then_index(sel_dist, sel_orig, 0, selected)
        # The root, if the metric wants one, AFTER the order is fixed on
        # the squared keys (their store-time `post_process`). `sqrt` is
        # monotone, so the rooted row is still ascending; two squared
        # values that root to one float keep their squared order.
        if not dist_is_identity and not partial_storage:
            postprocess_distances(sel_dist, index.metric)

        for i in range(selected):
            out_dist.append(sel_dist[i])
            out_idx.append(sel_orig[i])
        for i in range(selected, k):
            out_dist.append(Float32(0))
            out_idx.append(UInt32(0))
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
    _ = dq_norm^
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
