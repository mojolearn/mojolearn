# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel.
"""NI54 column moment kernel; opt-in canonical fold lives in adafactor.mojo.

One block owns a parameter column. Each lane folds one absolute 64-row leaf;
its partial is stored in shared memory, lane zero adds the live leaves in
ascending order, then the block advances to the next 32-leaf slab. Slabs are
scheduling only: the host adds those same leaves in that same order. There is
no vendor reduction intrinsic or dataset-dependent graph.

Default OFF. No compile, identity, optimizer quality or timing run performed.
"""
from std.gpu import block_idx, thread_idx
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from checks.numerics import ftz, identical_sqrt
from sequence.ops import FP, add, fma3, ld, lerp, mul, st
from sequence.adafactor import AF_COL_LEAF, div

comptime AF_COL_LANES = 32


def af_col_chunk_kernel(grad: FP, col_var: FP, rows_in: Int32, cols_in: Int32, beta: Float32):
    var col = Int(block_idx.x)
    var lane = Int(thread_idx.x)
    var rows = Int(rows_in)
    var cols = Int(cols_in)
    if col >= cols:
        return
    var partial = stack_allocation[AF_COL_LANES, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var total = Float32(0.0)
    var chunks = (rows + AF_COL_LEAF - 1) // AF_COL_LEAF
    var slab = 0
    while slab < chunks:
        var chunk = slab + lane
        var leaf = Float32(0.0)
        if chunk < chunks:
            var lo = chunk * AF_COL_LEAF
            var hi = min(rows, lo + AF_COL_LEAF)
            for row in range(lo, hi):
                var g = ld(grad, row * cols + col)
                leaf = fma3(g, g, leaf)
        partial[lane] = leaf
        barrier()
        if lane == 0:
            for j in range(min(AF_COL_LANES, chunks - slab)):
                total = add(total, partial[j])
        barrier()
        slab += AF_COL_LANES
    if lane == 0:
        var nrm = ftz(identical_sqrt(total))
        st(col_var, col, lerp(ld(col_var, col), div(mul(nrm, nrm), Float32(rows)), beta))
