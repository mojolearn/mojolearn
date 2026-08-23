"""`cuml/cpp/src_prims/functions/linearReg.cuh::linearRegH` -- what
`cdPredict` calls (`cd.cuh:305`).

    linearRegH(handle, input, n_rows, n_cols, coef, pred, intercept, stream):
        gemm(handle, input, n_rows, n_cols, coef, pred, n_rows, 1, N, N)   :28-29
        if intercept != 0: addScalar(pred, pred, intercept, n_rows)       :31

`input` is column-major `n_rows x n_cols` (cuML's `F` order); the gemm is
`pred[n_rows x 1] = input . coef` through cuBLAS (CLOSED).

UNDER IDENTICAL the product is `mojolearn.identical.gemm.fp32.v1` `OP_TN`
with `A = input` read as the row-major `k x m = n_cols x n_rows` matrix it
already is in memory (`A_eff[i, p] = A[p * m + i]`, contract section 3),
`B = coef` as `k x 1`, `m = n_rows, n = 1, k = n_cols`. No transpose is
materialized and the per-row fold is a pure function of `n_cols`.

UNDER FAST the mirror is `linalg.gemv.gemv_gpu` (`core/gemm.mojo::gemv_n`),
which wants its matrix ROW-major `m x k`; the column-major input is
transposed on the device first (`core/column_stats.mojo::transpose_kernel`,
which moves bits and performs no arithmetic). One `n_rows x n_cols` scratch
per predict; an execution-plan detail, cuBLAS reads either layout natively.

`addScalar` is one thread per row; `ftz` on the store under IDENTICAL.
`linearRegLossGrads` and the penalty gradients in the same file are the SGD
solver's and are not ported (`solver/UNPORTED.tsv`).
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.gpu import block_dim, block_idx, thread_idx

from core.column_stats import TRANSPOSE_TILE, transpose_kernel
from core.gemm import gemv_n
from gemm.mojo_only.gemm_identical import identical_gemm
from gemm.mojo_only.gemm_oracle import OP_TN
from mojo_only.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz

comptime LINREG_ELEM_TPB = 256


def add_scalar_kernel(
    v: MutPointer[Float32, MutAnyOrigin], n_in: Int32, s: Float32
):
    """`raft::linalg::addScalar(v, v, s, n)`."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        v.unsafe_store(i, ftz(v.unsafe_load(i) + s))


def linear_reg_h(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_cols: Int,
    mut coef: DeviceBuffer[DType.float32],
    mut pred: DeviceBuffer[DType.float32],
    intercept: Float32,
) raises:
    """`linearRegH`. SYNCHRONIZES (the IDENTICAL gemm entry does, and the
    FAST arm's transpose scratch must outlive its launch)."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        identical_gemm(ctx, pred, x, coef, n_rows, 1, n_cols, OP_TN)
    else:
        var xt = ctx.enqueue_create_buffer[DType.float32](n_rows * n_cols)
        # src is the column-major n_rows x n_cols matrix = a row-major
        # n_cols x n_rows one; its transpose is row-major n_rows x n_cols.
        ctx.enqueue_function[transpose_kernel](
            xt.unsafe_ptr(),
            x.unsafe_ptr(),
            Int32(n_cols),
            Int32(n_rows),
            grid_dim=(
                (n_rows + TRANSPOSE_TILE - 1) // TRANSPOSE_TILE,
                (n_cols + TRANSPOSE_TILE - 1) // TRANSPOSE_TILE,
                1,
            ),
            block_dim=(TRANSPOSE_TILE, TRANSPOSE_TILE, 1),
        )
        gemv_n(ctx, pred, xt, coef, n_rows, n_cols)
        ctx.synchronize()
        _ = xt^
    if intercept != Float32(0.0):
        ctx.enqueue_function[add_scalar_kernel](
            pred.unsafe_ptr(),
            Int32(n_rows),
            intercept,
            grid_dim=((n_rows + LINREG_ELEM_TPB - 1) // LINREG_ELEM_TPB, 1, 1),
            block_dim=(LINREG_ELEM_TPB, 1, 1),
        )
    ctx.synchronize()
