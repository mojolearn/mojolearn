"""Truncated SVD, which is PCA without the centering.

PORT OF `raft/linalg/detail/tsvd.cuh::tsvd_fit` at RAFT `9aa17e5`. Partial.
Do not improve.

Their `tsvd_fit` is three steps:

    1  input_cross_mult = X^T X      (raft::linalg::gemm, OP_T / OP_N, alpha=1)
    2  cal_eig on it
    3  truncate to n_components

Set that beside `pca_fit` and the whole difference is visible: PCA subtracts
the column means first and divides by `n_rows - 1`, truncated SVD does
neither. Same product, same eigensolver, same truncation. **That is why this
file is thirty lines and not three hundred**, and it is the payoff for having
mirrored their structure rather than writing a PCA that happened to work.

The consequence is the one that matters to a user: truncated SVD is NOT
translation invariant. Shift a column and its first component swings onto the
shift, because nothing centered it. PCA is invariant to exactly that. The
check asserts both, in one place, because the difference is the reason both
exist.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from cluster.mojo_only.reduce_by_key import copy_f32_kernel
from core.gemm import gemm_tn
from decomposition.ported.linalg.detail.pca import PCAResult, eig_and_truncate


def tsvd_fit(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    mut gram: DeviceBuffer[DType.float32],
    mut x_alias: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_cols: Int,
    n_components: Int,
) raises -> PCAResult:
    """`tsvd_fit`. No centering, no `n_rows - 1`, and the input is untouched.

    Unlike `pca_fit` this does not modify `x` at all, so there is no restore
    step to forget. Theirs has the same property for the same reason.
    """
    if n_cols <= 1:
        raise Error("Parameter n_cols: number of columns cannot be less than two")
    if n_rows <= 1:
        raise Error("Parameter n_rows: number of rows cannot be less than two")
    if n_components <= 0:
        raise Error(
            "Parameter n_components: number of components cannot be less than one"
        )
    if n_components > n_cols:
        raise Error("n_components cannot exceed n_cols")

    ctx.enqueue_function[copy_f32_kernel](
        x_alias.unsafe_ptr(),
        x.unsafe_ptr(),
        Int32(n_rows * n_cols),
        grid_dim=((n_rows * n_cols + 255) // 256, 1, 1),
        block_dim=(256, 1, 1),
    )
    # Step 1. `alpha = 1`, so scale 1: the raw Gram matrix, not a covariance.
    # Same tuned matmul as PCA, with alpha = 1 and no centering. Their
    # `tsvd_fit` asks cuBLAS for exactly this: CUBLAS_OP_T, CUBLAS_OP_N.
    gemm_tn(ctx, gram, x, x_alias, n_cols, n_cols, n_rows)

    ctx.synchronize()

    # Steps 2 and 3. `singular_scale = 1` because these eigenvalues already
    # ARE the squared singular values; PCA's are variances and need the
    # `n_rows - 1` factor put back.
    return eig_and_truncate(ctx, gram, n_cols, n_components, 1)
