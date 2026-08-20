"""Column means, centering, and the covariance product.

MOSTLY NOT PORTS OF FILES. These stand in for `raft::stats::mean`,
`raft::stats::cov` and `raft::stats::meanAdd`, which `raft/linalg/detail/
pca.cuh::pca_fit` calls in that order. RAFT is a general library this tree
does not mirror file for file, so what is reproduced is the CALL SITE and its
semantics. Same rule as `core/row_norms.mojo`.

THE GRAM PRODUCT IS NOT IN THIS FILE ANY MORE
---------------------------------------------
`raft::stats::cov` and `tsvd_fit` both ask cuBLAS for `A^T A`. This file used
to carry a hand-written contraction for that shape, a port of the
`isRowMajor == false` arm of `raft/linalg/detail/contractions.cuh` with a
split-K row partition of our own on top, on the belief that MAX's matmul could
not do it because `transpose_a` is unsupported. **`transpose_a` is still
unsupported and that belief was still wrong**: `Xt = transpose(X)` makes
`Xt . Xt^T` the N-T shape MAX does support, and `transpose_kernel` below is
the twenty lines that get there. The ported contraction and its split-K
reduction were dead code by then and are deleted. About 250 lines, none of
it reached.

`core/gemm.mojo::gemm_tn` now DISPATCHES that shape: the transpose route
above for outputs with enough tiles to fill the device, and the split-K
Gram kernel in `core/gram_splitk.mojo` for the tile-starved outputs the
shipped fits actually have, where the vendor matmul measured ~25 GFLOP/s
(one 32x32 output = one tile of parallelism). The tuned-BLAS rule stands
where the tuned kernel has parallelism; the measurement, not the rule, is
what put the small shapes on the hand-written kernel.

THE ONE SEMANTIC THAT IS EASY TO GET WRONG
------------------------------------------
`pca_fit` centers the input IN PLACE, computes the covariance from the
centered data, and then calls `raft::stats::meanAdd` to put the input BACK
(`detail/pca.cuh:186`). The input is an in-out parameter that ends the call
unchanged. Skipping the restore leaves the caller's matrix silently centered,
which is invisible until they use it for something else.

`cov` uses the `n_rows - 1` denominator, the sample covariance, because
`pca_fit` later multiplies the explained variances by exactly `n_rows - 1` to
recover singular values.
"""

from mojo_only.kernel_matrix import (
    K_LIB_COLUMN_STATS,
    TARGET_COLUMN,
    lib_block_size_for,
)


from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.memory import AddressSpace
from max.gpu.primitives.block import sum as block_sum
from max.gpu.sync import barrier
from std.memory import stack_allocation


# READ FROM THE MATRIX, not restated here. `mojo_only/kernel_matrix.mojo`
# owns every tunable in this tree; changing TARGET_COLUMN there rebuilds
# this kernel for another vendor with no edit in this file.
comptime STATS_TPB = lib_block_size_for[K_LIB_COLUMN_STATS, TARGET_COLUMN]()


def column_mean_kernel(
    mu: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    n_cols_in: Int32,
):
    """`raft::stats::mean` along columns. One block per column."""
    var n_rows = Int(n_rows_in)
    var n_cols = Int(n_cols_in)
    var col = Int(block_idx.x)
    var tid = Int(thread_idx.x)

    var acc = Float32(0.0)
    var r = tid
    while r < n_rows:
        acc += x.unsafe_load(r * n_cols + col)
        r += STATS_TPB

    # `cub::BlockReduce`'s counterpart. MAX ships the block
    # collectives at `max.gpu.primitives.block`, so the hand-written
    # shared-memory tree reduction this replaced is gone. Same
    # arithmetic, one call, and the reduction shape is Modular's to
    # tune rather than ours to guess. See VENDOR_LIBRARIES.md.
    var s0 = block_sum[block_size=STATS_TPB](acc)
    if tid == 0:
        mu.unsafe_store(col, s0 / Float32(n_rows))


def shift_columns_kernel(
    x: MutPointer[Float32, MutAnyOrigin],
    mu: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    n_cols_in: Int32,
    sign_in: Float32,
):
    """Center (`sign = -1`) or restore (`sign = +1`), in place.

    Both directions in one kernel because they are the same operation and
    because pairing them makes the restore hard to forget.
    """
    var n_cols = Int(n_cols_in)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= Int(n_rows_in) * n_cols:
        return
    var col = idx % n_cols
    x.unsafe_store(
        idx, x.unsafe_load(idx) + sign_in * mu.unsafe_load(col)
    )


def xty_kernel(
    out_v: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    y: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    n_cols_in: Int32,
):
    """`A^T b`. Stands in for `raft::linalg::gemv(..., trans=true)`.

    One block per feature, striding rows. This and the Gram product in
    `core/gemm.mojo::gemm_tn` are the only two things in ordinary least
    squares that touch rows at all; everything after them is
    `n_cols x n_cols`.
    """
    var n_rows = Int(n_rows_in)
    var n_cols = Int(n_cols_in)
    var col = Int(block_idx.x)
    var tid = Int(thread_idx.x)

    var acc = Float32(0.0)
    var r = tid
    while r < n_rows:
        acc += x.unsafe_load(r * n_cols + col) * y.unsafe_load(r)
        r += STATS_TPB

    # `cub::BlockReduce`'s counterpart. MAX ships the block
    # collectives at `max.gpu.primitives.block`, so the hand-written
    # shared-memory tree reduction this replaced is gone. Same
    # arithmetic, one call, and the reduction shape is Modular's to
    # tune rather than ours to guess. See VENDOR_LIBRARIES.md.
    var s0 = block_sum[block_size=STATS_TPB](acc)
    if tid == 0:
        out_v.unsafe_store(col, s0)


def divide_columns_by_nonzero_kernel(
    qs: MutPointer[Float32, MutAnyOrigin],
    q: MutPointer[Float32, MutAnyOrigin],
    s_vec: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    thresh_in: Float32,
):
    """`QS <- Q invS` with `DivideByNonZero`, `lstsq.cuh`'s matrixVectorOp.

    Column `k` of `Q` is divided by eigenvalue `k`, and a column whose
    eigenvalue is at or below the threshold is ZEROED rather than divided.

    That zeroing is the whole numerical story of solving least squares
    through the normal equations. `A^T A` squares the condition number, so a
    direction the data barely constrains shows up as a tiny eigenvalue, and
    dividing by it would amplify noise without bound. Dropping the direction
    instead is a pseudo-inverse, and it is why their default OLS solver is
    SVD rather than this one.
    """
    var n = Int(n_in)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= n * n:
        return
    var col = idx % n
    var lam = s_vec.unsafe_load(col)
    if lam > thresh_in or lam < -thresh_in:
        qs.unsafe_store(idx, q.unsafe_load(idx) / lam)
    else:
        qs.unsafe_store(idx, Float32(0.0))


def diagonal_to_vector_kernel(
    out_v: MutPointer[Float32, MutAnyOrigin],
    a: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    """Pull the eigenvalues off the diagonal Jacobi leaves behind.

    cuSOLVER hands back a separate eigenvalue array; Jacobi leaves them on
    the diagonal of the matrix it consumed. One kernel bridges the two
    conventions so the rest of the port reads like theirs.
    """
    var n = Int(n_in)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < n:
        out_v.unsafe_store(i, a.unsafe_load(i * n + i))


def scale_in_place_kernel(
    a: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    scale_in: Float32,
):
    """Apply cuBLAS's `alpha` after the fact.

    `raft::linalg::gemm` takes `alpha` and folds the scale into the product.
    MAX's `matmul` has no alpha argument, so the `1 / (n_rows - 1)` that turns
    a Gram matrix into a covariance is a separate pass over `n_cols^2`
    elements. That is a deviation in launch count, not in arithmetic, and it
    is tiny: the product is O(rows * cols^2) and this is O(cols^2).
    """
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        a.unsafe_store(i, a.unsafe_load(i) * scale_in)



comptime TRANSPOSE_TILE = 32


def transpose_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    src: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    n_cols_in: Int32,
):
    """`dst[n_cols x n_rows] = src[n_rows x n_cols]^T`, tiled through shared.

    NOT A PORT. `linalg.transpose` exists, compiles, and SIGNALS at runtime on
    device buffers — it dispatches into a HOST strided-copy path. This is the
    twenty-line kernel that `core/gemm.mojo` named as the alternative route
    and that nobody had written.

    Tiled and padded so both the read and the write are coalesced: a naive
    transpose is coalesced on exactly one side and strided on the other,
    which costs more than the whole operation is worth.
    """
    var n_rows = Int(n_rows_in)
    var n_cols = Int(n_cols_in)
    var tile = stack_allocation[
        TRANSPOSE_TILE * (TRANSPOSE_TILE + 1),
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()

    var tx = Int(thread_idx.x)
    var ty = Int(thread_idx.y)
    var r0 = Int(block_idx.y) * TRANSPOSE_TILE
    var c0 = Int(block_idx.x) * TRANSPOSE_TILE

    var r = r0 + ty
    var c = c0 + tx
    if r < n_rows and c < n_cols:
        tile[ty * (TRANSPOSE_TILE + 1) + tx] = src.unsafe_load(r * n_cols + c)
    barrier()

    var r2 = c0 + ty
    var c2 = r0 + tx
    if r2 < n_cols and c2 < n_rows:
        dst.unsafe_store(
            r2 * n_rows + c2, tile[tx * (TRANSPOSE_TILE + 1) + ty]
        )
