"""The LINEAR and RBF kernel matrices: `GramMatrixBase::linear`,
`RBFKernel::evaluate`, `rbf_kernel_expanded`, `matrixRowNormL2`.

PORT OF `cuvs/cpp/src/distance/detail/kernels/kernel_matrices.cu` at cuVS
`94c2819` (the `cuvs::distance::kernels` cuML 26.08 links; RAFT's
`distance/detail/kernels/kernel_matrices.cuh` is the same code one
repository earlier). Dense, row-major, FP32. POLYNOMIAL and TANH are NOT
ported (refused by name in `svm_parameter.mojo`; `svm/UNPORTED.tsv`).

THE ROUNDING SEQUENCE (svm/README.md, identity content section 1). Theirs:

    linear:  out = x1 . x2^T                          (cuBLAS gemm)
    rbf:     out = exp(-1.0 * gain * (norm_x[i] + norm_y[j] - out * 2))
             with norm = rowNorm<L2Norm> (SQUARED; a block fold), and for
             math_t = float the `-1.0 *` promotes to DOUBLE so the exp is
             the double one, rounded to float on store. No clamp at zero.

Ours, both modes the same association, the pins under IDENTICAL:

    dot  = gemm_nt (MAX matmul) under FAST / identical_gemm v1 under IDENTICAL
    norm = per-row SERIAL ascending chain, acc = ftz(fma(x, x, acc))
    s    = ftz( ftz(norm_x + norm_y) - ftz(2 * dot) )
    e    = ftz( (-gamma) * s )
    K    = ftz( identical_exp(e) )

# =========================================================================
# DEVIATION 630: the RBF exponential is FLOAT32 through `identical_exp`,
# not their promoted double `exp`. There is no float64 on the Apple GPU, so
# the double arm cannot exist on this column; the float arm is the one
# arithmetic on every vendor (IDENTITY_PATHS row 12). Measured against the
# host Float64 reference of THEIR spelling in `svc_check.mojo::
# check_rbf_float_vs_double_reference` (max ULP distance printed there).
# The norm is a serial chain rather than their block fold for the same
# reason `pinned_distance_tile.mojo` is one thread per cell: no fold shape
# to pin. Under FAST the only differences from theirs are the missing
# double promotion and the norm's fold shape; FAST makes no bit claim.
# =========================================================================
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.math import exp
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext

from core.gemm import gemm_nt
from gemm.mojo_only.gemm_identical import (
    identical_gemm_into,
    identical_gemm_workspace_max_floats,
)
from gemm.mojo_only.gemm_oracle import OP_NT
from mojo_only.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    ftz,
    identical_exp,
    identical_mul_add,
)
from svm.ported.svm.svm_parameter import KERNEL_LINEAR, KERNEL_RBF, KernelParams


#: SABOTAGE (svc_check "std exp under IDENTICAL"): route the RBF exponential
#: through the stdlib `exp` instead of `identical_exp`. Must FAIL the
#: device-vs-oracle gate under IDENTICAL on the RBF fixtures.
comptime SAB_STD_EXP = is_defined["MOJOLEARN_SVM_SABOTAGE_STD_EXP"]()

#: SABOTAGE ("drop ftz at the f seam"): see `smosolver.mojo`; listed here
#: so every sabotage define has one place where its name is spelled.
comptime SAB_NO_FTZ = is_defined["MOJOLEARN_SVM_SABOTAGE_NO_FTZ"]()

comptime KM_TPB = 256


def _grid(n: Int) -> Int:
    return (n + KM_TPB - 1) // KM_TPB


def row_norm_l2sq_kernel(
    out_norm: MutPointer[Float32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    n_cols_in: Int32,
):
    """`matrixRowNormL2` -> `raft::linalg::rowNorm<L2Norm>` (the squared
    norm, no sqrt), one thread per row, ascending serial chain."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_rows_in):
        var k = Int(n_cols_in)
        var acc = Float32(0.0)
        for c in range(k):
            var v = ftz(x.unsafe_load(i * k + c))
            acc = ftz(identical_mul_add(v, v, acc))
        out_norm.unsafe_store(i, ftz(acc))


def rbf_kernel_expanded_kernel(
    inout: MutPointer[Float32, MutAnyOrigin],
    rows_in: Int32,
    cols_in: Int32,
    norm_x: MutPointer[Float32, MutAnyOrigin],
    norm_y: MutPointer[Float32, MutAnyOrigin],
    gain: Float32,
):
    """`rbf_kernel_expanded`: `inout[i, j] = exp(-gain * (norm_x[i] +
    norm_y[j] - inout[i, j] * 2))`, row-major `[rows x cols]` here where
    theirs is column-major with `ld`. One thread per cell."""
    var rows = Int(rows_in)
    var cols = Int(cols_in)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < rows * cols:
        var i = t // cols
        var j = t - i * cols
        var dot = inout.unsafe_load(t)
        comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
            var s = ftz(
                ftz(ftz(norm_x.unsafe_load(i)) + ftz(norm_y.unsafe_load(j)))
                - ftz(Float32(2.0) * ftz(dot))
            )
            var e = ftz((-gain) * s)
            comptime if SAB_STD_EXP:
                inout.unsafe_store(t, ftz(exp(e)))
            else:
                inout.unsafe_store(t, ftz(identical_exp(e)))
        else:
            var s = norm_x.unsafe_load(i) + norm_y.unsafe_load(j) - dot * Float32(2.0)
            inout.unsafe_store(t, exp(-gain * s))


def row_norms_l2sq(
    ctx: DeviceContext,
    mut out_norm: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    n_rows: Int,
    n_cols: Int,
) raises:
    """`ML::SVM::matrixRowNorm(handle, matrix, out, L2Norm)`."""
    if n_rows <= 0:
        return
    ctx.enqueue_function[row_norm_l2sq_kernel](
        out_norm.unsafe_ptr(), x.unsafe_ptr(), Int32(n_rows), Int32(n_cols),
        grid_dim=_grid(n_rows), block_dim=KM_TPB,
    )


def kernel_workspace_floats(m: Int, n: Int, k: Int) -> Int:
    """What `kernel_op` needs in `ws` for an `m x n` product over `k`."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return identical_gemm_workspace_max_floats(m, n, k)
    return 1


def kernel_op(
    ctx: DeviceContext,
    kp: KernelParams,
    mut out: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32],
    mut b: DeviceBuffer[DType.float32],
    m: Int,
    n: Int,
    k: Int,
    mut norm_a: DeviceBuffer[DType.float32],
    mut norm_b: DeviceBuffer[DType.float32],
    mut ws: DeviceBuffer[DType.float32],
) raises:
    """`KernelOp(handle, kernel, x1, n1, n_cols, x2, n2, out, norm_x1,
    norm_x2)`: `out[m x n] = K(a_i, b_j)`, row-major. `GramMatrixBase::
    evaluate` (linear) or `RBFKernel::evaluate` (linear + expansion).
    ASYNCHRONOUS; `ws` is the caller's identical-GEMM workspace, at least
    `kernel_workspace_floats(m, n, k)` floats."""
    if m <= 0 or n <= 0:
        return
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        identical_gemm_into(ctx, out, a, b, ws, m, n, k, OP_NT)
    else:
        gemm_nt(ctx, out, a, b, m, n, k)
    if kp.kernel == KERNEL_RBF:
        ctx.enqueue_function[rbf_kernel_expanded_kernel](
            out.unsafe_ptr(), Int32(m), Int32(n),
            norm_a.unsafe_ptr(), norm_b.unsafe_ptr(), Float32(kp.gamma),
            grid_dim=_grid(m * n), block_dim=KM_TPB,
        )
    elif kp.kernel != KERNEL_LINEAR:
        raise Error("svm kernel_op: unported kernel " + String(kp.kernel))
