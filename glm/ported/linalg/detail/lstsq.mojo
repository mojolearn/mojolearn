"""Least squares through the normal equations and an eigendecomposition.

PORT OF `raft/linalg/detail/lstsq.cuh::lstsqEig` at RAFT `9aa17e5`.
Transliterated. Do not improve.

This is cuML's OLS solver `algo = 1` (`cuml/cpp/src/glm/ols.cuh:105`). Their
six steps, copied:

    covA = A^T A                 O(rows * cols^2)
    Ab   = A^T b                 O(rows * cols)
    Q S Q* = eig(covA)           O(cols^3)
    QS   = Q invS                O(cols^2)   with DivideByNonZero
    covA = QS Q^T                O(cols^3)   == inv(A^T A)
    w    = covA Ab               O(cols^2)

**Only the first two touch rows.** That is the same shape as PCA and it is
why this section cost almost nothing: `covariance_kernel` with scale 1 is
step 1, `jacobi_eigh_kernel` is step 3, and `gemm_nt_kernel` is steps 5 and
6. The genuinely new code is `xty_kernel` and the column division.

WHY THEY DEFAULT TO SVD AND NOT TO THIS
---------------------------------------
`olsFit`'s default is `algo = 0`, `lstsqSvdJacobi`. Forming `A^T A` SQUARES
the condition number, so this route loses roughly twice the digits an SVD
route would on an ill-conditioned design. `DivideByNonZero` is the guard: a
direction the data barely constrains appears as a near-zero eigenvalue and
gets DROPPED rather than divided by, which turns the inverse into a
pseudo-inverse.

Porting their non-default solver is a deliberate choice and it is recorded
in `glm/UNPORTED.tsv`: it is the one that reuses machinery this repository
already has, and their SVD route needs a one-sided Jacobi SVD that does not
exist here yet. The accuracy difference is real and belongs in any
comparison against scikit-learn, whose `LinearRegression` uses LAPACK
`gelsd`, an SVD route.

THE STREAM OVERLAP IS NOT PORTED
--------------------------------
Theirs computes `A^T A` and `A^T b` on TWO CUDA streams concurrently, with
events to join them (`lstsq.cuh`, `multAbStream`). Mojo's `DeviceContext`
gives one queue here, so ours runs them in sequence. That is a real
throughput deviation and not a correctness one, and it is exactly the kind of
control-plane concurrency `HOST_AND_DEVICE.md` says the incumbents get for
free from CUDA and we do not.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from core.column_stats import (
    COV_TILE,
    STATS_TPB,
    covariance_kernel,
    diagonal_to_vector_kernel,
    divide_columns_by_nonzero_kernel,
    xty_kernel,
)
from core.gemm import (
    GEMM_MBLK,
    GEMM_NBLK,
    GEMM_THREADS,
    gemm_nt_kernel,
)
from decomposition.mojo_only.jacobi_eigh_device import (
    JACOBI_TPB,
    jacobi_eigh_kernel,
)


def lstsq_eig(
    ctx: DeviceContext,
    mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    mut w: DeviceBuffer[DType.float32],
    mut cov_a: DeviceBuffer[DType.float32],
    mut q: DeviceBuffer[DType.float32],
    mut qs: DeviceBuffer[DType.float32],
    mut s_vec: DeviceBuffer[DType.float32],
    mut ab: DeviceBuffer[DType.float32],
    mut inv: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_cols: Int,
) raises:
    """`w = inv(A^T A) A^T b`, their step order."""
    # covA <- A^T A. alpha = 1, so scale 1: a Gram matrix, not a covariance.
    ctx.enqueue_function[covariance_kernel](
        cov_a.unsafe_ptr(),
        a.unsafe_ptr(),
        Int32(n_rows),
        Int32(n_cols),
        Float32(1.0),
        grid_dim=(
            (n_cols + COV_TILE - 1) // COV_TILE,
            (n_cols + COV_TILE - 1) // COV_TILE,
            1,
        ),
        block_dim=(COV_TILE, COV_TILE, 1),
    )
    # Ab <- A^T b. Theirs overlaps this with the line above on a second
    # stream; see the module docstring.
    ctx.enqueue_function[xty_kernel](
        ab.unsafe_ptr(),
        a.unsafe_ptr(),
        b.unsafe_ptr(),
        Int32(n_rows),
        Int32(n_cols),
        grid_dim=(n_cols, 1, 1),
        block_dim=(STATS_TPB, 1, 1),
    )
    ctx.synchronize()

    # Q S Q* <- covA. Jacobi consumes covA and leaves S on its diagonal.
    ctx.enqueue_function[jacobi_eigh_kernel](
        cov_a.unsafe_ptr(),
        q.unsafe_ptr(),
        Int32(n_cols),
        Int32(80),
        Float32(1.0e-10),
        grid_dim=(1, 1, 1),
        block_dim=(JACOBI_TPB, 1, 1),
    )
    ctx.enqueue_function[diagonal_to_vector_kernel](
        s_vec.unsafe_ptr(),
        cov_a.unsafe_ptr(),
        Int32(n_cols),
        grid_dim=((n_cols + 255) // 256, 1, 1),
        block_dim=(256, 1, 1),
    )
    ctx.synchronize()

    # QS <- Q invS, with DivideByNonZero.
    ctx.enqueue_function[divide_columns_by_nonzero_kernel](
        qs.unsafe_ptr(),
        q.unsafe_ptr(),
        s_vec.unsafe_ptr(),
        Int32(n_cols),
        Float32(1.0e-10),
        grid_dim=((n_cols * n_cols + 255) // 256, 1, 1),
        block_dim=(256, 1, 1),
    )
    # inv <- QS Q^T == Q invS Q^T == inv(A^T A)
    ctx.enqueue_function[gemm_nt_kernel](
        inv.unsafe_ptr(),
        qs.unsafe_ptr(),
        q.unsafe_ptr(),
        Int32(n_cols),
        Int32(n_cols),
        Int32(n_cols),
        grid_dim=(
                (n_cols + GEMM_NBLK - 1) // GEMM_NBLK,
                (n_cols + GEMM_MBLK - 1) // GEMM_MBLK,
                1,
            ),
            block_dim=(GEMM_THREADS, 1, 1),
    )
    # w <- inv Ab. A d x d matrix against one vector, expressed as the same
    # `X Y^T` the GEMM already does with n = 1.
    ctx.enqueue_function[gemm_nt_kernel](
        w.unsafe_ptr(),
        inv.unsafe_ptr(),
        ab.unsafe_ptr(),
        Int32(n_cols),
        Int32(1),
        Int32(n_cols),
        grid_dim=(
                (1 + GEMM_NBLK - 1) // GEMM_NBLK,
                (n_cols + GEMM_MBLK - 1) // GEMM_MBLK,
                1,
            ),
            block_dim=(GEMM_THREADS, 1, 1),
    )
    ctx.synchronize()
