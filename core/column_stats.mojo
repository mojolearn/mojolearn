"""Column means, centering, and the covariance product.

NOT PORTS OF FILES. These stand in for `raft::stats::mean`,
`raft::stats::cov` and `raft::stats::meanAdd`, which `raft/linalg/detail/
pca.cuh::pca_fit` calls in that order. RAFT is a general library this tree
does not mirror file for file, so what is reproduced is the CALL SITE and its
semantics; the kernels are ours. Same rule as `core/row_norms.mojo`.

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

from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.memory import stack_allocation


comptime STATS_TPB = 128
comptime COV_TILE = 16


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

    var s = stack_allocation[
        STATS_TPB,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    s[tid] = acc
    barrier()
    var half = STATS_TPB // 2
    while half > 0:
        if tid < half:
            s[tid] = s[tid] + s[tid + half]
        barrier()
        half //= 2
    if tid == 0:
        mu.unsafe_store(col, s[0] / Float32(n_rows))


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


def covariance_kernel(
    cov: MutPointer[Float32, MutAnyOrigin],
    xc: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    n_cols_in: Int32,
    scale_in: Float32,
):
    """`X^T X * scale`. Serves BOTH of RAFT's users of this product.

    `raft::stats::cov` on centered data with `scale = 1 / (n_rows - 1)` is
    PCA's covariance. `tsvd_fit` calls `raft::linalg::gemm` with
    `CUBLAS_OP_T, CUBLAS_OP_N` and `alpha = 1` on UNCENTERED data
    (`detail/tsvd.cuh`), which is the same product with `scale = 1`.

    One kernel with a scale rather than two, because the two differ only in
    that constant and in whether the caller centered first. Which of those
    happens is the caller's business and is exactly what separates PCA from
    truncated SVD.

    A different GEMM shape from `core/gemm.mojo`, which computes `X Y^T`.
    Here both operands are the same matrix and the CONTRACTED axis is the row
    axis, so the tiles walk down rows rather than across features.

    **This is the only part of PCA that scales with rows.** Everything after
    it works on an `n_cols x n_cols` matrix, which is why the eigen step can
    sit on the host without breaking `HOST_AND_DEVICE.md`'s rule.

    NUMERIC: the contraction order over rows fixes the summation order, so
    the tile size moves the last bits. `COV_TILE` is a fixed constant for the
    same reason `GEMM_TILE` is.
    """
    var n_rows = Int(n_rows_in)
    var n_cols = Int(n_cols_in)
    var tx = Int(thread_idx.x)
    var ty = Int(thread_idx.y)
    var i = Int(block_idx.y) * COV_TILE + ty
    var j = Int(block_idx.x) * COV_TILE + tx

    var s_a = stack_allocation[
        COV_TILE * COV_TILE,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var s_b = stack_allocation[
        COV_TILE * COV_TILE,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()

    # Both tiles are loaded as [contraction_row][feature], because the
    # contracted axis here is the ROW axis and both operands are the same
    # matrix. Getting these two indices the wrong way round produces a
    # plausible but NON-SYMMETRIC matrix, which then makes the Jacobi
    # eigensolver run forever rather than return a wrong answer. That is how
    # this bug was found; see PORTING.md 23.
    var acc = Float32(0.0)
    var tile = 0
    while tile * COV_TILE < n_rows:
        var row = tile * COV_TILE + ty
        var fa = Int(block_idx.y) * COV_TILE + tx
        var fb = Int(block_idx.x) * COV_TILE + tx

        if row < n_rows and fa < n_cols:
            s_a[ty * COV_TILE + tx] = xc.unsafe_load(row * n_cols + fa)
        else:
            s_a[ty * COV_TILE + tx] = Float32(0.0)
        if row < n_rows and fb < n_cols:
            s_b[ty * COV_TILE + tx] = xc.unsafe_load(row * n_cols + fb)
        else:
            s_b[ty * COV_TILE + tx] = Float32(0.0)
        barrier()

        for t in range(COV_TILE):
            acc += s_a[t * COV_TILE + ty] * s_b[t * COV_TILE + tx]

        barrier()
        tile += 1

    if i < n_cols and j < n_cols:
        cov.unsafe_store(i * n_cols + j, acc * scale_in)
