# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`raft/stats/detail/mean.cuh::mean<rowMajor=false>` and
`raft/stats/detail/mean_center.cuh::{meanCenter, meanAdd}<false, true>` --
what `GLM::preProcessData` / `postProcessData` call (`preprocess.cuh:58-66`,
`:98-99`) and nothing else in this section.

    mean<false>(mu, data, D, N, stream):                    mean.cuh:16-33
        ratio = OutType(1) / OutType(N)                     (a HOST float)
        reduce<rowMajor=false, alongRows=false>(mu, data, D, N, 0,
               identity_op, add_op, mul_const_op(ratio))

so each column's mean is its `coalescedReduction` SUM times `1/N` as a
multiply (not a divide by N; `core/column_stats.mojo::column_mean_kernel`
divides, which is why it is not reused -- `glm/README.md` records the same
distinction for the logistic bias gradient). The two arms of the sum are
`solver/derived/linalg/norm.mojo`'s: their Medium kernel with `identity_op`
and the `ratio` as `final_op` under FAST; under IDENTICAL the profile dot
of the column against ONES (`profile_dot.mojo`'s header: `fma(x, 1, acc)`
is `x + acc` exactly, so it is the profile sum) and then `ftz(sum * ratio)`
on the device in a one-thread kernel, the multiply being a SEAM the
centering reads (row 10).

    meanCenter<rowMajor=false, bcastAlongRows=true>(out, data, mu, D, N):
        matrixVectorOp<false, true>(out, data, mu, D, N, sub_op)
    meanAdd  ... the same with add_op                     mean_center.cuh:14-31

For a column-major `N x D` matrix with `bcastAlongRows = true` the vector
is indexed by COLUMN: `out[i, j] = data[i, j] -/+ mu[j]`. One thread per
cell, no reduction, `ftz` on the store under IDENTICAL (the centered matrix
is what every norm and dot then reads). `matrixVectorOp`'s own kernel
(`matrix_vector_op.cuh`) vectorizes the load by `VecLen`; that is a
scheduling choice with no arithmetic in it and is not mirrored.
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.gpu import block_dim, block_idx, thread_idx

from original.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz
from solver.original.profile_dot import profile_dot_into
from solver.derived.linalg.coalesced_reduction import coalesced_sum_medium

#: SCHEDULING. One thread, one cell.
comptime MEAN_ELEM_TPB = 256


def scale_n_kernel(
    v: MutPointer[Float32, MutAnyOrigin], n_in: Int32, ratio: Float32
):
    """`mul_const_op(ratio)` applied as the `final_op` of `n` sums already
    in `v` (the IDENTICAL arm, where the sum came from the profile dot)."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        v.unsafe_store(i, ftz(v.unsafe_load(i) * ratio))


def column_mean(
    ctx: DeviceContext,
    mut mu: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    d: Int,
    n: Int,
    mut ones: DeviceBuffer[DType.float32],
    mut ws: DeviceBuffer[DType.float32],
    plan: Int = -1,
) raises:
    """`raft::stats::mean<false>(mu, x, D=d, N=n)`: `d` column means of a
    column-major `n x d` matrix. `ones` holds `n` floats of `1.0` and `ws`
    the profile workspace (IDENTICAL arm only; FAST ignores both)."""
    var ratio = Float32(1.0) / Float32(n)
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        for j in range(d):
            var a = x.create_sub_buffer[DType.float32](j * n, n)
            var c = mu.create_sub_buffer[DType.float32](j, 1)
            profile_dot_into(ctx, c, a, ones, ws, n, plan)
            _ = a^
            _ = c^
        ctx.enqueue_function[scale_n_kernel](
            mu.unsafe_ptr(),
            Int32(d),
            ratio,
            grid_dim=((d + MEAN_ELEM_TPB - 1) // MEAN_ELEM_TPB, 1, 1),
            block_dim=(MEAN_ELEM_TPB, 1, 1),
        )
    else:
        coalesced_sum_medium[False](ctx, mu, x, n, d, ratio)


def mean_shift_columns_kernel(
    x: MutPointer[Float32, MutAnyOrigin],
    mu: MutPointer[Float32, MutAnyOrigin],
    n_rows_in: Int32,
    n_cols_in: Int32,
    sign: Float32,
):
    """`matrixVectorOp<false, true>(x, x, mu, D, N, sub_op | add_op)` in
    place: `x[i + j*n_rows] = x[...] + sign * mu[j]`, `sign = -1` for
    `meanCenter`, `+1` for `meanAdd`. `x - mu` and `x + (-mu)` are the same
    IEEE operation (negation is exact), so one kernel carries both ops
    without a second rounding."""
    var n_rows = Int(n_rows_in)
    var n_cols = Int(n_cols_in)
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx < n_rows * n_cols:
        var j = idx // n_rows
        var m = ftz(mu.unsafe_load(j))
        x.unsafe_store(idx, ftz(ftz(x.unsafe_load(idx)) + sign * m))


def mean_center(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    mut mu: DeviceBuffer[DType.float32],
    d: Int,
    n: Int,
) raises:
    """`meanCenter<false, true>(x, x, mu, D=d, N=n)`."""
    ctx.enqueue_function[mean_shift_columns_kernel](
        x.unsafe_ptr(), mu.unsafe_ptr(), Int32(n), Int32(d), Float32(-1.0),
        grid_dim=((n * d + MEAN_ELEM_TPB - 1) // MEAN_ELEM_TPB, 1, 1),
        block_dim=(MEAN_ELEM_TPB, 1, 1),
    )


def mean_add(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    mut mu: DeviceBuffer[DType.float32],
    d: Int,
    n: Int,
) raises:
    """`meanAdd<false, true>(x, x, mu, D=d, N=n)`."""
    ctx.enqueue_function[mean_shift_columns_kernel](
        x.unsafe_ptr(), mu.unsafe_ptr(), Int32(n), Int32(d), Float32(1.0),
        grid_dim=((n * d + MEAN_ELEM_TPB - 1) // MEAN_ELEM_TPB, 1, 1),
        block_dim=(MEAN_ELEM_TPB, 1, 1),
    )
