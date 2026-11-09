# SPDX-License-Identifier: Apache-2.0
"""The small MLP's elementwise cell (bias add, bias + ReLU, ReLU backward,
row sum), shared by the device kernel (autograd/small_mlp_op.mojo) and the
host arm (autograd/small_mlp_host.mojo), so both columns compute the same
words. No GPU and no host-thread imports.

Restored 2026-11-09 (lane r1-restore-measured-winners) from
training/mlp_ops.mojo `_mlp_kernel` / `_add` at f59d913b0b: the body is
that kernel's, unchanged, moved into an @always_inline helper so the
launched kernel and the host arm both call it (a launched kernel is never
called from another kernel).

    operation 0  output[i] = source[i] + other[i % cols]          (bias)
    operation 1  output[i] = max(source[i] + other[i % cols], +0) (bias + ReLU)
    operation 2  output[i] = ftz(other[i]) if source[i] > 0 else +0 (ReLU backward,
                 source = the forward's activation)
    operation 3  output[c] = sum over rows of source[row, c], rows ascending
                 (one owner per column; `other` unread)

Additions are checks.numerics' pinned FMA with multiplier one, operands and
result flushed (the IDENTICAL FTZ).
"""
from checks.numerics import ftz, identical_mul_add

comptime SMALL_MLP_BIAS = 0
comptime SMALL_MLP_BIAS_RELU = 1
comptime SMALL_MLP_RELU_BACKWARD = 2
comptime SMALL_MLP_SUM_ROWS = 3

comptime _P = MutPointer[Float32, MutAnyOrigin]


@always_inline
def small_mlp_add(left: Float32, right: Float32) -> Float32:
    # Existing pinned FP32 arithmetic: fma(1, left, right) is a single
    # rounded addition. Both operands and the result use the IDENTICAL FTZ.
    return ftz(identical_mul_add(Float32(1), ftz(left), ftz(right)))


@always_inline
def small_mlp_cell_count(rows: Int, cols: Int, operation: Int) -> Int:
    """Items of one launch: one per column for the row sum, else one per cell."""
    return cols if operation == SMALL_MLP_SUM_ROWS else rows * cols


@always_inline
def small_mlp_cell(source: _P, other: _P, dst: _P, rows: Int, cols: Int, operation: Int, i: Int):
    """Item `i` of `operation` (header); callers bound i by small_mlp_cell_count."""
    if operation == SMALL_MLP_SUM_ROWS:
        var value = Float32(0)
        # One owner per column. The fold order does not depend on GPU
        # width, warp size, launch partition, scheduling or host threads.
        for row in range(rows):
            value = small_mlp_add(value, source.unsafe_load(row * cols + i))
        dst.unsafe_store(i, value)
        return
    if operation == SMALL_MLP_RELU_BACKWARD:
        var activation = source.unsafe_load(i)
        var g = Float32(0)
        if activation > Float32(0):
            g = ftz(other.unsafe_load(i))
        dst.unsafe_store(i, g)
        return
    var value = small_mlp_add(source.unsafe_load(i), other.unsafe_load(i % cols))
    if operation == SMALL_MLP_BIAS_RELU and value <= Float32(0):
        value = Float32(0)
    dst.unsafe_store(i, value)
