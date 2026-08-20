"""Truncated SVD, which is PCA without the centering.

PORT OF `cuml/cpp/src/tsvd/tsvd.cuh::tsvdFit` at cuML `00094f7`
(branch-25.08). Partial. Do not improve.

The path this file used to cite, `raft/linalg/detail/tsvd.cuh`, does not
exist and never has. Truncated SVD lives in cuML.

Their `tsvdFit` (`tsvd.cuh:190-238`) is three steps:

    1  input_cross_mult = X^T X      (raft::linalg::gemm, OP_T / OP_N, alpha=1)
    2  calEig on it                  (`tsvd.cuh:230`)
    3  truncZeroOrigin + seqRoot     (`tsvd.cuh:233-237`)

DIVERGENCE: WHAT `tsvdFit` RETURNS IS TWO ARRAYS, NOT FIVE
-----------------------------------------------------------
`tsvdFit`'s out-parameters are `components` and `singular_vals` and NOTHING
else. It does not call `truncCompExpVars`, so it computes no
`explained_var`, no `explained_var_ratio` and no `noise_vars` -- only
`pcaFit` does that (`pca.cuh:132`). This port reaches `eig_and_truncate`,
which is the PCA-shaped routine, so it fills all five fields of `PCAResult`
on the tSVD path too.

Three of those five are OURS, and the third one is the one to watch:
**their `explained_var` for truncated SVD is not the eigenvalue at all.**
`tsvdFitTransform` computes it as `raft::stats::vars` of the TRANSFORMED
data, after the transform and after the sign flip (`tsvd.cuh:272-276`). On
centered data those agree with the eigenvalues up to the `n_rows - 1`
factor; on the RAW data truncated SVD is defined over they do not, because
`vars` subtracts the column mean of the scores and the eigenvalue does not.
So a caller comparing `PCAResult.explained_var` from `tsvd_fit` against
cuML's `TruncatedSVD.explained_variance_` is comparing two different
quantities. NOT FIXED here: it is an arithmetic change and this lane is
read-mostly. See the dispatch-audit lane report.

`tsvdFit` also passes NO `set_neg_zero` to `seqRoot` (`tsvd.cuh:237`), where
`pcaFit` passes `true` (`pca.cuh:136`) -- so the clamp that turns a negative
eigenvalue into a zero singular value is a PCA-only step, and this port
performs it on neither.

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

from core.gemm import gemm_tn
from decomposition.ported.linalg.detail.pca import PCAResult, eig_and_truncate


def tsvd_fit(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    mut gram: DeviceBuffer[DType.float32],
    mut x_alias: DeviceBuffer[DType.float32],
    mut x_alias2: DeviceBuffer[DType.float32],
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

    # Step 1. `alpha = 1`, so scale 1: the raw Gram matrix, not a covariance,
    # and no centering. Their `tsvd_fit` asks cuBLAS for exactly this:
    # CUBLAS_OP_T, CUBLAS_OP_N. Same `gemm_tn` dispatch as PCA (split-K
    # kernel at these output widths; see core/gram_splitk.mojo).
    gemm_tn(ctx, gram, x, x_alias, x_alias2, n_cols, n_cols, n_rows)

    ctx.synchronize()

    # Steps 2 and 3. `singular_scale = 1` because these eigenvalues already
    # ARE the squared singular values; PCA's are variances and need the
    # `n_rows - 1` factor put back.
    return eig_and_truncate(ctx, gram, n_cols, n_components, 1)
