# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# Derivative work: the upstream file and its pinned commit are recorded in this lane's DERIVATION_MAP.tsv and in this file's own docstring. See NOTICE.
"""Brute-force k-nearest-neighbors: their DISPATCH, and their FALLBACK.

PORT OF `cuvs/src/neighbors/detail/knn_brute_force.cuh` at cuVS `94c2819`:
`brute_force_knn_impl`'s dispatch (`:443-447`) and `tiled_brute_force_knn`
(`:69-340`). Partial. Do not improve.

WHAT THEIR DISPATCH DOES, WHICH IS NOT WHAT THIS FILE USED TO SAY
------------------------------------------------------------------
`brute_force_knn_impl` chooses between two entirely different algorithms at
`:443`:

    if (k <= 64 && rowMajorQuery == rowMajorIndex && rowMajorQuery == true &&
        (metric == L2Unexpanded || L2SqrtUnexpanded ||
         L2Expanded  || L2SqrtExpanded)) {
      fusedL2Knn(...);                       // fused_l2_knn.cuh
    } else {
      switch (metric) {
        case Haversine: haversine_knn(...);  // not ported
        default:        tiled_brute_force_knn(...);
      }
    }

`tiled_brute_force_knn` is the **else**. For k <= 64 on row-major L2 - which
is every k-NN measurement this repository has ever taken - cuVS runs
`fusedL2Knn`, and until 2026-08-19 this tree had ported only the fallback and
compared it against scikit-learn as though it were their algorithm. That is
now `neighbors/gbdt/neighbors/detail/fused_l2_knn.mojo`, and
`brute_force_knn_impl` below is their dispatch rather than a direct call to
whichever function we happened to own.

The two paths are not interchangeable and the difference is structural, not a
constant factor. This one MATERIALIZES a `tile_rows x n_index` distance
matrix: the GEMM writes it, the epilogue reads and rewrites it, and the
selector reads it a third time. At 400,000 index points and
`bench/scaling_main.mojo:46`'s tile of 256 that is 409.6 MB per tile, about
23 GB of traffic across the run to carry 51.2 GFLOP of arithmetic. The fused
kernel never writes it at all.

WHAT IS THEIRS ON THIS PATH
----------------------------
- `cuvs::distance::pairwise_distance` at `:172-183` is cuBLAS underneath, and
  cuBLAS has no source to port, so this calls `linalg.matmul` through
  `core/gemm.mojo::gemm_nt`. That substitution is legitimate HERE and only
  here, because a materialized distance matrix is what their own fallback
  asks for on this path.
- `cuvs::selection::select_k` at `:265` and `:305`. Their `select_k`
  dispatches (`raft/matrix/detail/select_k-inl.cuh:47-72`) to WARPSORT for
  `2 < k <= 256` and to RADIX only above that, so radix is their second
  choice across the whole practical range. `select_radix.mojo` is ported and
  is the selector here; `nn.topk.top_k` stays reachable behind
  `use_vendor_topk` as a second opinion the ported selector can be checked
  against, and `neighbors/original/knn_check.mojo` runs that comparison.
- `raft::linalg::rowNorm` at `:110-146` -> `core/row_norms.mojo`, hoisted out
  of the tile loop exactly as their comment at `:107-109` says.
- The L2 epilogue at `:184-205` is `raft::linalg::map_offset` over their
  `l2_exp_cutlass_op`. That is RAFT's own portable elementwise map, so it is
  a port and not a substitution: `core/expand_distances.mojo`. It is MISSING
  one of the op's two clamp clauses; see the lane file.

WHAT IS NOT PORTED
------------------
Their `DistanceEpilogue` template, the bitmap/bitset filters at `:229-256`,
the sparse and non-expanded metrics, `haversine_knn`, and the multi-index
merge (`knn_merge_parts`). See `neighbors/NOT_IMPLEMENTED.tsv`.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from core.expand_distances import expand_distances_kernel
from core.gemm import gemm_nt
from core.row_norms import NORM_TPB, row_norm_kernel
from original.kernel_matrix import (
    K_LIB_SELECT_WARPSORT,
    TARGET_COLUMN,
    knn_auto_follows_their_dispatch_for,
    knn_warpsort_select_for,
    lib_block_size_for,
    lib_lane_width_for,
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
from layout import TileTensor
from layout.tile_layout import row_major
from nn.topk import top_k

from neighbors.derived.matrix.detail.select_radix import (
    SELECT_BLOCK,
    radix_topk_one_block_kernel,
)
from neighbors.derived.matrix.detail.select_warpsort import (
    MAX_CAPACITY,
    warpsort_topk_block_kernel,
)
from neighbors.derived.neighbors.detail.fused_l2_knn import (
    FKNN_MAX_NN,
    fused_l2_knn,
    fused_l2_knn_grid,
)


def compute_norms(
    ctx: DeviceContext,
    mut a: DeviceBuffer[DType.float32],
    mut a_norm: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_features: Int,
    take_sqrt: Bool,
) raises:
    """`knn_brute_force.cuh:110-146`, hoisted out of the tile loop.

    Cosine wants the L2 norm and L2 wants the SQUARED norm, which is their
    comment at `:117-118` and is the same flag `cluster/` carries.
    """
    ctx.enqueue_function[row_norm_kernel](
        a_norm.unsafe_ptr(),
        a.unsafe_ptr(),
        Int32(n_features),
        Int32(1 if take_sqrt else 0),
        grid_dim=(n_rows, 1, 1),
        block_dim=(NORM_TPB, 1, 1),
    )


def _warpsort_select_tile[
    capacity: Int
](
    ctx: DeviceContext,
    mut dist_tile: DeviceBuffer[DType.float32],
    mut dummy_idx: DeviceBuffer[DType.uint32],
    mut out_dist: DeviceBuffer[DType.float32],
    mut out_idx: DeviceBuffer[DType.uint32],
    out_offset: Int,
    rows: Int,
    n_index: Int,
    k: Int,
) raises:
    """One query tile's top-k through the ported RAFT warpsort
    (DEVIATION 1922; `select_warpsort.mojo::warpsort_topk_block_kernel`).

    LAUNCH GEOMETRY per that file's module docstring: the SINGLE-PASS form,
    `num_blocks == 1` and `grid = (1, rows, 1)`, which writes the FINAL
    answer with no second merging launch. Each block grid-strides one whole
    row of the materialized tile, exactly the shape `_run_warpsort` in
    `neighbors/original/warpsort_check.mojo` gates against radix and the
    host oracle. RAFT's `calc_launch_parameter` (`select_k-inl.cuh:1058`)
    would additionally split long rows over several blocks and merge with a
    second launch (`select_k_:1106-1120`); that caller loop is NOT ported
    (declared in the module docstring) and is the open item if one block
    per row underfills a device.

    `capacity` is their `bound_by_power_of_two(k)` clamped to a floor of
    32: below 32 RAFT shrinks `warp_width` to the capacity and rescales
    `block_warps`, a subwarp form nothing here instantiates; capacity 32 is
    correct for every `k <= 32` and keeps `warp_width == 32 == WARP_LANES`,
    which is the geometry the whole file's uniformity argument is written
    against.

    `dummy_idx` is the `has_in_idx == 0` contract: a valid, DISTINCT buffer
    that is never read; the payload is the column index, which is the
    neighbor id this caller wants.
    """
    comptime assert capacity >= 32 and capacity <= MAX_CAPACITY, (
        "warpsort selection: capacity must be in [32, kMaxCapacity]"
    )
    #: `block_warps * warp_width`, keyed off the kernel matrix so the ONE
    #: place that must widen on AMD (`lib_block_size_for`'s own note) is
    #: also the one this launch reads. On every column this row admits
    #: (32-lane), this is 256 = 8 warps, their `__launch_bounds__(256)`.
    comptime BDX = lib_block_size_for[K_LIB_SELECT_WARPSORT, TARGET_COLUMN]()
    comptime BW = BDX // 32
    ctx.enqueue_function[warpsort_topk_block_kernel[capacity, True, BW]](
        dist_tile.unsafe_ptr(),
        dummy_idx.unsafe_ptr(),
        out_dist.unsafe_ptr().unsafe_offset(out_offset),
        out_idx.unsafe_ptr().unsafe_offset(out_offset),
        Int32(n_index),
        Int32(k),
        Int32(0),
        grid_dim=(1, rows, 1),
        block_dim=(BDX, 1, 1),
    )


def _radix_select_tile(
    ctx: DeviceContext,
    mut dist_tile: DeviceBuffer[DType.float32],
    mut buf_val: DeviceBuffer[DType.float32],
    mut buf_idx: DeviceBuffer[DType.uint32],
    mut out_dist: DeviceBuffer[DType.float32],
    mut out_idx: DeviceBuffer[DType.uint32],
    out_offset: Int,
    rows: Int,
    n_index: Int,
    k: Int,
    buf_len: Int,
) raises:
    """One query tile's top-k through the ported RAFT radix selector --
    the FAST launch that sat inline in `tiled_brute_force_knn` before
    DEVIATION 1922 gave the selection two call sites. Byte-for-byte the
    same enqueue; hoisted, not changed.
    """
    ctx.enqueue_function[radix_topk_one_block_kernel](
        dist_tile.unsafe_ptr(),
        out_dist.unsafe_ptr().unsafe_offset(out_offset),
        out_idx.unsafe_ptr().unsafe_offset(out_offset),
        buf_val.unsafe_ptr(),
        buf_idx.unsafe_ptr(),
        Int32(n_index),
        Int32(k),
        Int32(buf_len),
        Int32(1),
        grid_dim=(rows, 1, 1),
        block_dim=(SELECT_BLOCK, 1, 1),
    )


def tiled_brute_force_knn(
    ctx: DeviceContext,
    mut queries: DeviceBuffer[DType.float32],
    mut query_norm: DeviceBuffer[DType.float32],
    mut index: DeviceBuffer[DType.float32],
    mut index_norm: DeviceBuffer[DType.float32],
    mut dist_tile: DeviceBuffer[DType.float32],
    mut buf_val: DeviceBuffer[DType.float32],
    mut buf_idx: DeviceBuffer[DType.uint32],
    mut out_dist: DeviceBuffer[DType.float32],
    mut out_idx: DeviceBuffer[DType.uint32],
    mut out_idx32: DeviceBuffer[DType.int32],
    n_queries: Int,
    n_index: Int,
    n_features: Int,
    k: Int,
    query_tile: Int,
    buf_len: Int,
    is_sqrt: Bool,
    use_vendor_topk: Bool = False,
) raises:
    """Tile the QUERIES, keep the whole index resident, top-k per query row.

    THEIR FALLBACK, reached from `brute_force_knn_impl` below only when the
    fused path's conditions fail (k > 64, or a metric fusion does not cover).

    Their loop tiles both axes. This tiles queries only, because the index
    axis is what the top-k reduces over and splitting it needs a merge of
    partial top-k lists (`:278-320`). That merge is NOT ported and is the
    largest remaining gap on this path; the lane file carries the full
    reading of their `num_col_tiles` / `temp_out_cols` machinery and what it
    would take. So `n_index` columns of one query tile must fit `dist_tile`,
    and `query_tile` is the knob that makes that true.
    """
    var q = 0
    while q < n_queries:
        var rows = min(query_tile, n_queries - q)

        # z = Q_tile . I^T
        # A `create_sub_buffer` window rather than a pointer offset, because
        # MAX's matmul takes a TileTensor over a DeviceBuffer and there is no
        # offset form of that. Same bytes, no copy.
        var cells = rows * n_index
        comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
            # IDENTITY_PATHS row 24. The vendor matmul's k-split is a
            # summation order nothing here can pin, so under IDENTICAL the
            # product and the epilogue are ONE kernel with the feature axis
            # walked ascending in a single thread. See
            # `neighbors/original/pinned_distance_tile.mojo` for the price
            # and for why no faster shape was chosen.
            ctx.enqueue_function[pinned_distance_tile_kernel](
                dist_tile.unsafe_ptr(),
                queries.unsafe_ptr().unsafe_offset(q * n_features),
                index.unsafe_ptr(),
                query_norm.unsafe_ptr().unsafe_offset(q),
                index_norm.unsafe_ptr(),
                Int32(rows),
                Int32(n_index),
                Int32(n_features),
                Int32(1 if is_sqrt else 0),
                grid_dim=(
                    (cells + PINNED_TILE_TPB - 1) // PINNED_TILE_TPB, 1, 1
                ),
                block_dim=(PINNED_TILE_TPB, 1, 1),
            )
        else:
            var q_tile = queries.create_sub_buffer[DType.float32](
                q * n_features, rows * n_features
            )
            gemm_nt(ctx, dist_tile, q_tile, index, rows, n_index, n_features)

            # The epilogue k-means fuses into its reduction has to be its
            # own pass here, because the top-k needs every distance to
            # survive.
            ctx.enqueue_function[expand_distances_kernel](
                dist_tile.unsafe_ptr(),
                query_norm.unsafe_ptr().unsafe_offset(q),
                index_norm.unsafe_ptr(),
                Int32(rows),
                Int32(n_index),
                Int32(1 if is_sqrt else 0),
                grid_dim=((cells + 255) // 256, 1, 1),
                block_dim=(256, 1, 1),
            )

        # THE SELECTION. Three implementations, and which one runs is a
        # parameter or a kernel-matrix row, never a preference.
        #
        # The ported RAFT radix select is the base default; on the columns
        # `knn_warpsort_select_for` admits (DEVIATION 1922), the FAST
        # selector for `2 < k <= 256` is the ported RAFT warpsort, which is
        # RAFT's own `select_k` choice for that band. `nn.topk.top_k` is a
        # device-wide call that consumes a materialized matrix, which is
        # exactly what this path already has, so it is a legitimate second
        # opinion HERE - and it is the reason it can never be the fused
        # path's selector, because a device-wide call cannot live inside a
        # kernel. `neighbors/original/knn_check.mojo` diffs the two.
        if use_vendor_topk:
            var dv = dist_tile.create_sub_buffer[DType.float32](
                0, rows * n_index
            )
            var ov = out_dist.create_sub_buffer[DType.float32](q * k, rows * k)
            var oi = out_idx32.create_sub_buffer[DType.int32](q * k, rows * k)
            top_k[largest=False, target="gpu"](
                TileTensor(dv, row_major(rows, n_index)),
                k,
                1,
                TileTensor(ov, row_major(rows, k)),
                TileTensor(oi, row_major(rows, k)),
                False,
                ctx,
            )
        else:
            comptime if PIN_DETERMINISM:
                # IDENTITY_PATHS row 11's closure, DEVIATIONS 500/501: the
                # composite (distance, index) key and the ranked placement.
                # The ported selector keeps RAFT's tie handling, which is
                # atomic-ordered by construction; this one has no tie class
                # and no arrival order in its output.
                #
                # **`PIN_DETERMINISM`, NOT `== NUMERIC_IDENTICAL`, SINCE
                # 2026-08-29.** "Atomic-ordered by construction" is a
                # RUN-TO-RUN property: two runs of the same fit on the same
                # GPU can place a tied neighbour in different slots, because
                # which thread's `atomicAdd` lands first is not a function of
                # the input. That is the middle tier's whole promise, and the
                # pin was keyed to the top tier only -- so a DETERMINISTIC
                # k-NN build took RAFT's selector and was not deterministic.
                # IDENTICAL is unmoved; `PIN_DETERMINISM` is true there too.
                #
                # THE REFUSAL BELOW NOW BINDS THE MIDDLE TIER TOO, and that
                # is a real narrowing of what `deterministic` accepts rather
                # than a formality: `k > SELECT_BLOCK` raises where a FAST
                # build would have answered. Refusing beats returning an
                # answer that moves between two runs of the same call.
                if k > SELECT_BLOCK:
                    raise Error(
                        "select_radix ("
                        + numeric_mode_name()
                        + "): k > "
                        + String(SELECT_BLOCK)
                        + " is refused. The rank pass gives one thread to"
                        " each output slot; a larger k needs a loop, which"
                        " is not written until something asks for it."
                    )
                ctx.enqueue_function[radix_topk_identical_kernel](
                    dist_tile.unsafe_ptr(),
                    out_dist.unsafe_ptr().unsafe_offset(q * k),
                    out_idx.unsafe_ptr().unsafe_offset(q * k),
                    buf_val.unsafe_ptr(),
                    buf_idx.unsafe_ptr(),
                    Int32(n_index),
                    Int32(k),
                    Int32(buf_len),
                    Int32(1),
                    grid_dim=(rows, 1, 1),
                    block_dim=(SELECT_BLOCK, 1, 1),
                )
            else:
                # DEVIATION 1922 (kernel-matrix row `knn_warpsort_select_for`):
                # RAFT's OWN `select_k` dispatch sends `2 < k <= 256` to the
                # WARPSORT family and only `k > 256` to radix
                # (`select_k-inl.cuh:38`), and this tree ran radix alone
                # across that whole band. On the columns the row admits
                # (32-lane FAST), the band takes the ported warpsort in the
                # single-pass block form; everything else -- k outside the
                # band, excluded columns, and BOTH UPPER TIERS, whose
                # selector is pinned above -- keeps radix byte for byte.
                # The `comptime if` is load-bearing, not style: this
                # function is non-generic, so any kernel named under a
                # RUNTIME `if` here instantiates whenever the module
                # builds, reachable or not -- the exact shape that killed
                # the first MI300X gbdt build (DEVIATION 1910's lesson).
                # The 32-lane warpsort kernel must not be INSTANTIATED on
                # a column the row excludes, so the exclusion is comptime.
                comptime if knn_warpsort_select_for[TARGET_COLUMN, False]():
                    if k > 2 and k <= MAX_CAPACITY:
                        # `bound_by_power_of_two(k)` with a floor of 32;
                        # the four instantiations mirror RAFT's template
                        # dispatch over capacities.
                        if k <= 32:
                            _warpsort_select_tile[32](
                                ctx, dist_tile, buf_idx, out_dist, out_idx,
                                q * k, rows, n_index, k,
                            )
                        elif k <= 64:
                            _warpsort_select_tile[64](
                                ctx, dist_tile, buf_idx, out_dist, out_idx,
                                q * k, rows, n_index, k,
                            )
                        elif k <= 128:
                            _warpsort_select_tile[128](
                                ctx, dist_tile, buf_idx, out_dist, out_idx,
                                q * k, rows, n_index, k,
                            )
                        else:
                            _warpsort_select_tile[256](
                                ctx, dist_tile, buf_idx, out_dist, out_idx,
                                q * k, rows, n_index, k,
                            )
                    else:
                        _radix_select_tile(
                            ctx, dist_tile, buf_val, buf_idx, out_dist,
                            out_idx, q * k, rows, n_index, k, buf_len,
                        )
                else:
                    _radix_select_tile(
                        ctx, dist_tile, buf_val, buf_idx, out_dist,
                        out_idx, q * k, rows, n_index, k, buf_len,
                    )
        q += rows
    ctx.synchronize()


#: WHICH SIDE OF `knn_brute_force.cuh:443` THIS PORT TAKES BY DEFAULT.
#:
#: DEVIATION 36 (REVISED 2026-08-19, second measurement round): cuVS sends
#: `k <= 64` + row-major + L2 to `fusedL2Knn` unconditionally; we send it
#: there ONLY when `fused_l2_knn_grid` -- their own `launchConfigGenerator`,
#: fed the hardware matrix's target column (Apple column = the M4 values
#: every table below was measured on) -- picks `grid_x == 1`, and to
#: `tiledBruteForceKnn` (their `else`)
#: when it would engage the x-split. Both arms are theirs; which one runs
#: unasked is decided by their launch computation evaluated on our hardware.
#:
#: HISTORY, because this default has now flipped once and the reason matters.
#: The first port ran the fused kernel at a fixed `grid = (1, ceil(m/16))`
#: (~125 blocks at 2,000 queries, independent of index size) and measured
#: 0.84x-0.88x against tiled at every index size 20k-400k, both arm orders.
#: That table set the original DEVIATION 36 default = TILED. It was a
#: measurement of the WRONG GEOMETRY: the launch computation is part of their
#: algorithm, and porting its A100 output instead of the computation itself
#: starved the M4. With `launchConfigGenerator` ported (grid capped at
#: minGridSize = 120 with a row grid-stride) the same kernel re-measured
#: 2026-08-19 evening, arms interleaved in the repeat loop, BOTH orders
#: pooled, n = 6/size, 32 features, k = 10, 2,000 queries:
#:
#:     n index    tiled med ms   fused med ms   fused/tiled
#:      20,000          23.8           26.1        0.91x
#:      50,000          67.9           63.6        1.07x
#:     100,000         174.2          154.4        1.13x
#:     200,000         323.9          274.1        1.18x
#:     400,000         615.7          517.9        1.19x
#:
#: Per-size ranges overlap (a noisy thermal window), so no single row is a
#: verdict; the query sweep at a fixed 200,000-row index is, in both
#: directions at once:
#:
#:     queries    tiled med ms   fused med ms   fused/tiled   grid
#:         64            10.6           11.4        0.93x     (16, 4)
#:        250            40.8           48.6        0.84x     (7, 16)
#:        500            56.5          293.7        0.19x     (4, 32)
#:      1,000           111.0          499.7        0.22x     (2, 63)
#:      2,000           268.0          264.6        1.01x     (1, 120)
#:      8,000         1,247.7        1,121.2        1.11x     (1, 120)
#:     32,000         5,575.6        4,445.3        1.25x     (1, 120)
#:
#: THE SIGN FLIPS WITH THE GRID SHAPE. Everywhere the computation picks
#: `grid_x == 1` the fused kernel is at worst a tie and at 32,000 queries a
#: clean non-overlapping 1.25x win; everywhere it engages the x-split the
#: mutex merge is a loss, and at 500-1,000 queries a CATASTROPHIC one (down
#: to 0.19x, with a 2.1-second worst sample at 500 queries against tiled's
#: 43 ms). The mutex protocol is CORRECT on Metal (the acquire-load spin /
#: weak-CAS / release-store spelling, probe-verified under contention), but
#: paying serialized cross-block merges per row costs more here than the
#: occupancy it buys. So:
#:
#:   - `KNN_METHOD_AUTO` (the DEFAULT): fused iff `fused_l2_knn_grid` says
#:     `grid_x == 1`, else tiled. The number the default flips on is by
#:     construction the number the launch would use.
#:   - `KNN_METHOD_FUSED` restores cuVS's dispatch exactly, x-split included.
#:   - `KNN_METHOD_TILED` restores the 2026-08-19-morning default.
#:
#: THE k HOLE, CLOSED. The tables above are all k = 10, and fused's domain
#: runs to k = 64 with register queues that grow with k, so AUTO could have
#: been shipping an unmeasured high-k loss. Swept 2026-08-19 late evening,
#: interleaved, 2,000 queries (grid (1, 120)), fused/tiled median ratios:
#:
#:     k      100,000 idx   400,000 idx
#:     10        1.13x         1.06x
#:     32        1.10x         1.00x
#:     64        1.04x         1.05x
#:
#: No k-degradation; every range overlaps (INDISTINGUISHABLE singly), fused
#: never behind on a median. AUTO holds across the whole fused k domain.
#:
#: This does not change any answer. Both arms match the host Float64 oracle
#: slot for slot on the same fixtures (`check_fused_l2_knn`,
#: `check_fused_griddimx_merge`, `check_fused_edge_shapes`), so the switch
#: changes a wait and not an output.
comptime KNN_METHOD_FUSED = 0
comptime KNN_METHOD_TILED = 1
comptime KNN_METHOD_AUTO = 2


def brute_force_knn_impl(
    ctx: DeviceContext,
    mut queries: DeviceBuffer[DType.float32],
    mut query_norm: DeviceBuffer[DType.float32],
    mut index: DeviceBuffer[DType.float32],
    mut index_norm: DeviceBuffer[DType.float32],
    mut dist_tile: DeviceBuffer[DType.float32],
    mut buf_val: DeviceBuffer[DType.float32],
    mut buf_idx: DeviceBuffer[DType.uint32],
    mut out_dist: DeviceBuffer[DType.float32],
    mut out_idx: DeviceBuffer[DType.uint32],
    mut out_idx32: DeviceBuffer[DType.int32],
    n_queries: Int,
    n_index: Int,
    n_features: Int,
    k: Int,
    query_tile: Int,
    buf_len: Int,
    is_sqrt: Bool,
    use_vendor_topk: Bool = False,
    row_major_query: Bool = True,
    row_major_index: Bool = True,
    knn_method: Int = KNN_METHOD_AUTO,
) raises:
    """`brute_force_knn_impl`'s dispatch, `knn_brute_force.cuh:443-447`.

    Their four conditions, in their order:

        k <= 64
        rowMajorQuery == rowMajorIndex
        rowMajorQuery == true
        metric is one of L2Unexpanded / L2SqrtUnexpanded / L2Expanded /
                         L2SqrtExpanded

    The metric test is not a runtime test here because this port carries only
    the expanded-L2 arm, so it is satisfied by construction; `is_sqrt`
    selects between L2Expanded and L2SqrtExpanded, both of which are in their
    set. If a metric outside that set is ever ported, this is where it has to
    become a switch, and the `Haversine` case at `:481-487` goes with it.

    `k > n_index` is refused outright rather than dispatched, because their
    `n < k` fill at `:157-166` is not ported on either arm and the fallback's
    selector cannot take `k > len`. See the raise below.

    `query_tile`, `buf_len`, `dist_tile`, `buf_val`, `buf_idx` and
    `out_idx32` are only read on the fallback path. The fused path needs none
    of them, which is the whole point of it.

    **THE DEFAULT ARM IS AUTO, AND WHAT AUTO MEANS IS PER COLUMN.** This
    paragraph used to say `knn_method` defaults to `KNN_METHOD_TILED`; that
    was stale the moment the signature below said `KNN_METHOD_AUTO`, and is
    deleted rather than annotated. Under FAST, AUTO picks by DEVIATION 36's
    shape test (fused iff the launch computation says `grid_x == 1`) except
    on columns where `knn_auto_follows_their_dispatch_for` (DEVIATION 1923)
    restores cuVS's unconditional dispatch; under IDENTICAL, DEVIATION 509
    pins AUTO to the tiled arm on every column. The four conditions above
    are still evaluated and still decide the arm whenever
    `knn_method == KNN_METHOD_FUSED`. Read DEVIATION 36 above the constants
    before changing any of it.
    """
    # THEIR `n < k` CASE IS NOT PORTED ON EITHER ARM, SO REFUSE IT.
    #
    # `knn_brute_force.cuh:157-166` fills the output with
    # `numeric_limits<DistanceT>::lowest()` and, for a signed index type,
    # `-1`, so a row with fewer than k candidates comes back marked. That
    # fill is not ported. Worse, the selector this path would reach cannot
    # take `k > len` at all: `select_radix.mojo:329` looks for the bucket
    # where `prev_count < current_k <= cur_count`, and when the whole row
    # holds fewer than k elements no bucket ever satisfies it, so `ctr` is
    # never updated, every later pass drops every element, and `last_filter`
    # reads a buffer nothing wrote. That is a silent wrong answer, and
    # returning one is worse than refusing.
    #
    # Read from their file and from ours, NOT measured on hardware. Porting
    # the fill needs the selection clamped to `min(k, n)` and its packed
    # output scattered back to a stride of `k`, which is the same scatter the
    # two-axis tiling needs; both are in the lane file as one open item.
    if k > n_index:
        raise Error(
            "brute_force_knn_impl: k > n_index. Their `n < k` short-fill at"
            " knn_brute_force.cuh:157-166 is not ported and the ported radix"
            " selector cannot take k > len; see the lane file."
        )

    # Their four conditions, unchanged, AND our own method switch. The
    # switch is a separate clause rather than a rewrite of theirs so that
    # `knn_method = KNN_METHOD_FUSED` restores their dispatch exactly. The
    # AUTO default asks the launch computation itself which regime this shape
    # is in; see DEVIATION 36 above the constants for the measurements.
    var want_fused = knn_method == KNN_METHOD_FUSED
    if knn_method == KNN_METHOD_AUTO:
        comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
            # THE ARM PIN, IDENTITY_PATHS row 23. AUTO normally chooses by
            # SHAPE -- fused when the launch computation says `grid_x == 1`,
            # tiled when it would engage the x-split -- and the two arms
            # resolve a tie in the k-th distance differently: the tiled arm's
            # composite key takes the LOWEST INDEX (DEVIATION 500) while the
            # fused arm's FAISS queue compares the distance only. A
            # shape-dependent arm therefore makes the ANSWER depend on how
            # many queries the caller happened to pass, which is not a
            # property an identical column can have.
            #
            # DEVIATION 509 SUPERSEDES 502'S CHOICE OF ARM, and the reason is
            # a column, not a preference. 502 pinned AUTO to the FUSED arm
            # (cuVS's own dispatch, `knn_brute_force.cuh:443`) and that is
            # well-defined on a 32-lane column -- but `fused_l2_knn` REFUSES
            # at its entry wherever `lib_lane_width_for[TARGET_COLUMN]() !=
            # 32`, because the FAISS queue is a 32-lane bitonic network and
            # a 64-wide wavefront addresses the wrong half of it (row 23's
            # own refusal). So under 502 the IDENTICAL build of k-NN did not
            # merely lose its tie guarantee on AMD, it RAISED: the pinned
            # arm was one that cannot exist on one of the three columns this
            # tree claims. A dispatch that is unrunnable on a column is not
            # that column's dispatch.
            #
            # AUTO therefore pins to the TILED arm on EVERY column under
            # IDENTICAL. That arm is the one whose answer is NAMED rather
            # than merely reproducible:
            #
            #   - `select_radix_identical` ranks over the composite
            #     `(twiddle_in(distance) << 32) | index` key, so equidistant
            #     neighbours come back LOWEST INDEX FIRST by construction and
            #     the tie class does not exist (DEVIATIONS 500, 501);
            #   - `pinned_distance_tile` accumulates each cell in ONE thread
            #     over the whole feature axis, ascending, with no vendor
            #     matmul and no k-split (DEVIATION 505);
            #   - neither kernel contains a warp primitive or reads a lane
            #     width, and the selector's only block collective is a
            #     prefix sum over Int32 histogram counts, which is exact at
            #     any tree shape.
            #
            # The price is measured, not argued: `price-unsupervised-identity`
            # puts `knn.tiled` at 2.85x `knn.auto`'s FAST arm on the M4, so
            # this is what the identical column costs for `k <= 64`. It buys
            # the only thing the column is for -- the same bits on Metal, PTX
            # and AMDGPU -- and it buys it on all three rather than on two.
            #
            # `KNN_METHOD_FUSED` is untouched: it restores their dispatch
            # exactly, keeps 502's `grid_x = 1` pin so its tie set stays a
            # pure function of `(m, n, k)`, and keeps refusing on a column
            # whose lane width is not 32.
            want_fused = False
        else:
            comptime if knn_auto_follows_their_dispatch_for[
                TARGET_COLUMN, False
            ]():
                # DEVIATION 1923 (kernel-matrix row): on this column AUTO
                # restores cuVS's dispatch UNCONDITIONALLY -- fused for the
                # whole `k <= 64` row-major L2 band, x-split and mutex merge
                # included. DEVIATION 36's `grid_x == 1` gate was measured
                # entirely on Apple milliseconds (the Metal x-split cliff,
                # 0.19x at 500-1,000 queries); on the vendor whose own
                # scheduler the mutex protocol was written for, the gate was
                # sending the H100's 4,000-query benchmark shape to the
                # TILED arm cuVS never takes there. The row's docstring in
                # `original/kernel_matrix.mojo` carries the argument;
                # `run_knn`'s `ours-fused` / `ours-tiled` arms are the A/B
                # that confirms or flips it.
                want_fused = True
            else:
                var g = fused_l2_knn_grid(n_queries, n_index)
                want_fused = g[0] == 1

    # DEVIATION 512: THE ARM MUST EXIST ON THIS COLUMN BEFORE AUTO PICKS IT.
    #
    # Found by compiling this file against `MOJOLEARN_COLUMN_AMD` on the Mac
    # (2026-08-23), which is a build the four-column matrix supports and
    # nothing had ever run: `check_knn_fused_tie_set_is_geometry_invariant`
    # died with `fusedL2kNN: the FAISS warp queue is a 32-lane bitonic
    # network and this target column's lane width is 64`.
    #
    # That refusal is CORRECT (row 23) and it was being reached through the
    # DEFAULT. `fused_l2_knn` is refused wherever the lane width is not 32,
    # but AUTO's geometry test does not know that, so on a 64-wide wavefront
    # every `k <= 64` row-major query whose launch computation returns
    # `grid_x == 1` -- the COMMON shape, and the one DEVIATION 36 measured
    # the fused arm to be fastest at -- raised instead of running. A caller
    # who never named an arm got an exception for a shape class, on a
    # vendor, in the SHIPPED mode. Under IDENTICAL, DEVIATION 509 had
    # already moved AUTO off the fused arm; this is the same hole in FAST,
    # and it is a correctness bug rather than an identity one.
    #
    # The refusal stays exactly where it is for an EXPLICIT
    # `KNN_METHOD_FUSED`: asking for an arm this column cannot express
    # should say so rather than silently substituting another one with a
    # different tie rule. AUTO, which by definition did not ask, takes the
    # tiled arm.
    comptime fused_arm_exists = lib_lane_width_for[TARGET_COLUMN]() == 32
    comptime if not fused_arm_exists:
        if knn_method == KNN_METHOD_AUTO:
            want_fused = False

    var fused_ok = (
        k <= FKNN_MAX_NN
        and row_major_query == row_major_index
        and row_major_query
        and want_fused
    )
    if fused_ok:
        fused_l2_knn(
            ctx,
            queries,
            query_norm,
            index,
            index_norm,
            out_dist,
            out_idx,
            n_queries,
            n_index,
            n_features,
            k,
            is_sqrt,
        )
    else:
        tiled_brute_force_knn(
            ctx,
            queries,
            query_norm,
            index,
            index_norm,
            dist_tile,
            buf_val,
            buf_idx,
            out_dist,
            out_idx,
            out_idx32,
            n_queries,
            n_index,
            n_features,
            k,
            query_tile,
            buf_len,
            is_sqrt,
            use_vendor_topk,
        )
