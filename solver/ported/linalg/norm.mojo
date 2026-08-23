"""`raft/linalg/detail/norm.cuh::colNormCaller<L2Norm, rowMajor=false>` --
the squared column norms `cdFit` precomputes (`cd.cuh:172-173`).

    raft::linalg::colNorm<raft::linalg::NormType::L2Norm, false>(
        squared.data(), input, int64_t(n_cols), int64_t(n_rows), stream);

`norm.cuh:56-58`: L2Norm is `reduce<rowMajor, false>(dots, data, D, N, 0,
stream, false, sq_op, add_op, fin_op)` with `fin_op = identity` (the
`<..., false>` template argument is `rowMajor`; there is NO sqrt -- the
name says norm, the value is the SUM OF SQUARES, and `cdUpdateCoefKernel`
divides by it as such). `reduce<false, false>` is `coalescedReduction(dots,
data, N, D)` (`detail/reduce.cuh:40-42`), i.e. `n_cols` reductions of
contiguous length `n_rows`.

TWO ARMS, per `solver/ported/linalg/coalesced_reduction.mojo`'s header:
FAST is their Medium kernel with `sq_op` (one block per column, per-thread
KBN, two `block.sum`s); IDENTICAL is the profile dot of the column with
itself, one `gemm.fp32.v1` cell per column, whose fold is a pure function
of `n_rows`. The IDENTICAL arm's `a` and `b` are two sub-buffer views of
the same column: the contract's 5a/5b flushes load each operand, the leaf
chain is `fma(x, x, acc)`, and `x * x` of a normal float is what `sq_op`
computes, so the per-element value is the same and only the fold differs
-- which is the point.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from mojo_only.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from solver.mojo_only.profile_dot import profile_dot_into
from solver.ported.linalg.coalesced_reduction import coalesced_sum_medium


def col_norm_l2_squared(
    ctx: DeviceContext,
    mut squared: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    n_cols: Int,
    n_rows: Int,
    mut ws: DeviceBuffer[DType.float32],
    plan: Int = -1,
) raises:
    """`colNorm<L2Norm, false>(squared, x, n_cols, n_rows)`. `x` is
    column-major `n_rows x n_cols`. `ws` and `plan` are the IDENTICAL arm's
    (`profile_dot_into`); FAST ignores both."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        for j in range(n_cols):
            var a = x.create_sub_buffer[DType.float32](j * n_rows, n_rows)
            var b = x.create_sub_buffer[DType.float32](j * n_rows, n_rows)
            var c = squared.create_sub_buffer[DType.float32](j, 1)
            profile_dot_into(ctx, c, a, b, ws, n_rows, plan)
            _ = a^
            _ = b^
            _ = c^
    else:
        coalesced_sum_medium[True](ctx, squared, x, n_rows, n_cols, Float32(1.0))
