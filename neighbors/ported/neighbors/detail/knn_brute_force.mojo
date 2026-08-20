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
now `neighbors/ported/neighbors/detail/fused_l2_knn.mojo`, and
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
  against, and `neighbors/mojo_only/knn_check.mojo` runs that comparison.
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
merge (`knn_merge_parts`). See `neighbors/UNPORTED.tsv`.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from core.expand_distances import expand_distances_kernel
from core.gemm import gemm_nt
from core.row_norms import NORM_TPB, row_norm_kernel
from layout import TileTensor
from layout.tile_layout import row_major
from nn.topk import top_k

from neighbors.ported.matrix.detail.select_radix import (
    SELECT_BLOCK,
    radix_topk_one_block_kernel,
)
from neighbors.ported.neighbors.detail.fused_l2_knn import (
    FKNN_MAX_NN,
    fused_l2_knn,
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
        var q_tile = queries.create_sub_buffer[DType.float32](
            q * n_features, rows * n_features
        )
        gemm_nt(ctx, dist_tile, q_tile, index, rows, n_index, n_features)

        # The epilogue k-means fuses into its reduction has to be its own
        # pass here, because the top-k needs every distance to survive.
        var cells = rows * n_index
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

        # THE SELECTION. Two implementations, and which one runs is a
        # parameter rather than a preference.
        #
        # The ported RAFT radix select is the default. `nn.topk.top_k` is a
        # device-wide call that consumes a materialized matrix, which is
        # exactly what this path already has, so it is a legitimate second
        # opinion HERE - and it is the reason it can never be the fused
        # path's selector, because a device-wide call cannot live inside a
        # kernel. `neighbors/mojo_only/knn_check.mojo` diffs the two.
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
            ctx.enqueue_function[radix_topk_one_block_kernel](
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
        q += rows
    ctx.synchronize()


#: WHICH SIDE OF `knn_brute_force.cuh:443` THIS PORT TAKES BY DEFAULT.
#:
#: DEVIATION 36: WE DEFAULT TO THE TILED ARM AND cuVS DEFAULTS TO THE FUSED
#: ONE. Their dispatch sends `k <= 64` + row-major + L2 to `fusedL2Knn`; ours
#: sends it to `tiledBruteForceKnn`, their `else`. Both arms are theirs. Only
#: which one runs unasked has changed.
#:
#: Measured on an M4, 32 features, k = 10, 2,000 queries, ARMS INTERLEAVED
#: INSIDE THE REPEAT LOOP, min of 3, and RE-RUN WITH THE ARMS IN THE OPPOSITE
#: ORDER (the second run agreed with the first to about 1%, so the ordering is
#: not carrying the result). OURS AGAINST OURS:
#:
#:     n index     tiled ms   fused ms   fused/tiled
#:      20,000        15.64      18.55       0.84x
#:      50,000        39.60      45.39       0.87x
#:     100,000        79.44      90.34       0.88x
#:     200,000       155.25     180.22       0.86x
#:     400,000       306.10     359.47       0.85x
#:
#: **The fused kernel is slower at every size measured and faster at none.**
#: That is the same shape of argument the DBSCAN default rests on (DEVIATION
#: 35), pointed the other way, and it was a surprise: fusion was expected to
#: win by removing the distance matrix entirely.
#:
#: WHY IT LOSES, AS FAR AS IT HAS BEEN MEASURED. The ported `fusedL2Knn` runs
#: at `gridDim.x == 1`, so its block count is `ceil(n_queries / Mblk)` and does
#: NOT depend on n_index: at 2,000 queries that is ~125 blocks whether the
#: index holds 20,000 rows or 400,000, and each block streams the whole index.
#: Holding n_index at 200,000 and raising the query count confirms this is part
#: of it, because the deficit shrinks monotonically as blocks are added:
#:
#:     queries     tiled ms   fused ms   fused/tiled
#:        500         39.60      59.91       0.66x
#:      2,000        154.89     180.12       0.86x
#:      8,000        618.95     673.95       0.92x
#:     32,000      2,498.22   2,696.12       0.93x
#:
#: **but it never crosses.** It asymptotes near 0.93x, so under-parallelisation
#: explains the size of the gap and not its sign. The leading hypothesis for
#: the residual is that the tiled arm's contraction is `linalg.matmul` while
#: the fused arm's is our transliteration of RAFT's Policy4x4 tile, and
#: `core/gemm.mojo` already records those measured at 248 against 15 GFLOP/s on
#: a standalone product. **That is a hypothesis, not a measurement of THIS
#: kernel**, and it is not what decided the default; the table above is.
#:
#: WHAT WOULD REVERSE THIS: their `gridDim.x > 1` split (the mutex protocol
#: at `fused_l2_knn.cuh:241-338`) LANDED on 2026-08-19, along with their
#: `launchConfigGenerator` (M4 inputs, `pairwise_distance_base.mojo`), so the
#: fused arm now fields the grid their computation chooses instead of a fixed
#: 1 x ceil(m/16). NOTE WHAT THAT COMPUTATION SAYS AT THE BENCH SHAPE: 2,000
#: queries is 125 y-chunks against a 120-block capacity, so it still picks
#: `grid_x == 1` there (with `grid_y` capped at 120 and a row grid-stride);
#: the x-split engages below ~1,905 queries, e.g. (16, 4) at 53 queries. The
#: table above is therefore STALE in its launch geometry but not yet
#: re-measured; the default stays TILED until the orchestrator re-times both
#: arms. It stays reachable through `knn_method` precisely so that measuring
#: it again costs one argument rather than a revert.
#:
#: This does not change any answer. `check_fused_l2_knn` and
#: `check_fused_edge_shapes` match the host Float64 oracle slot for slot and in
#: order, on the same fixtures the tiled arm is checked against, so the flip
#: changes a wait and not an output.
comptime KNN_METHOD_FUSED = 0
comptime KNN_METHOD_TILED = 1


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
    knn_method: Int = KNN_METHOD_TILED,
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

    **BUT THE FUSED PATH IS NOT WHAT THIS RUNS BY DEFAULT.** `knn_method`
    defaults to `KNN_METHOD_TILED`, which is their `else` at `:447`, because
    the fused arm measured slower at every size on this hardware. The four
    conditions above are still evaluated and still decide the arm whenever
    `knn_method == KNN_METHOD_FUSED`. Read DEVIATION 36 above the constants
    before changing either.
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
    # `knn_method = KNN_METHOD_FUSED` restores their dispatch exactly. See
    # DEVIATION 36 above the constants for why the default is the other side.
    var fused_ok = (
        k <= FKNN_MAX_NN
        and row_major_query == row_major_index
        and row_major_query
        and knn_method == KNN_METHOD_FUSED
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
