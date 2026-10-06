# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Squared L2 norm of every row, which the expanded identity needs twice.

DOES NOT FOLLOW cuVS. Their call is
`raft::linalg::norm<L2Norm, Apply::ALONG_ROWS>` (`detail/kmeans.cuh:770`,
`minClusterDistanceCompute.cu:44`), and RAFT is a separate library whose
primitives this tree does not mirror file for file. Only the CALL SITES and
their semantics are theirs, and those are copied exactly:

- X's norms are computed ONCE per fit, before the iteration loop, and reused
  by every Lloyd iteration and by k-means++ (`detail/kmeans.cuh:786-790`).
- Centroid norms are recomputed EVERY assignment, because the centroids move
  (`minClusterDistanceCompute.cu:43-49`).
- For L2 the norm is left SQUARED. For cosine it is passed through `sqrt`,
  because the cosine branch of the reduction divides by `||x|| ||y||` rather
  than subtracting. Getting that backward gives a plausible, wrong answer on
  every row, which is why the flag is a parameter and not a comment.

**This reduction IS numeric**, unlike the argmin it feeds. It is a float sum
over the feature axis, so the block size changes the summation order and
therefore the last bits. It is listed in the `IDENTICAL` column's scope.

WHAT `IDENTICAL` DOES TO IT (IDENTITY_PATHS row 19, DEVIATION 503/504)
----------------------------------------------------------------------
Three separate pathways reach the last bits of a norm, and each takes a
different one of the ledger's three moves:

1. THE FOLD WIDTH. `NORM_TPB` is `lib_block_size_for[K_LIB_ROW_NORM]`, a
   row the matrix labelled SCHEDULING and which is a summation order.
   PINNED at the accessor (DEVIATION 508); bit-inert today because every
   column carries 128, and the point is the column that does not yet.
2. THE FOLD SHAPE. `block.sum` folds across lanes at the HARDWARE width,
   32 on Apple and NVIDIA and 64 on AMD. REPLACED under IDENTICAL by
   `core/pinned_reduce.pinned_block_sum`, a halving tree with no lane
   primitive in it.
3. CONTRACTION. `acc += v * v` is one rounding or two at the codegen's
   whim -- Metal measured UNFUSED, CUDA contracts by default. PINNED to
   one `fma` under IDENTICAL through `numerics.identical_mul_add`.

Plus `ftz` on the accumulator and on the value written, because a squared
feature difference is exactly where a denormal appears and Metal flushes
where CUDA does not (row 10). All four are comptime no-ops under FAST, so
the shipped bits do not move.
"""

from checks.kernel_matrix import (
    K_LIB_ROW_NORM,
    TARGET_COLUMN,
    lib_block_size_for,
)


from std.gpu import block_dim, block_idx, thread_idx
from std.math import sqrt
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from std.memory import stack_allocation

from core.pinned_reduce import pinned_block_sum
from checks.numerics import ftz, identical_mul_add, identical_sqrt


# READ FROM THE MATRIX, not restated here. `checks/kernel_matrix.mojo`
# owns every tunable in this tree; changing TARGET_COLUMN there rebuilds
# this kernel for another vendor with no edit in this file.
comptime NORM_TPB = lib_block_size_for[K_LIB_ROW_NORM, TARGET_COLUMN]()


def row_norm_kernel(
    out_norm: MutPointer[Float32, MutAnyOrigin],
    a: MutPointer[Float32, MutAnyOrigin],
    n_cols_in: Int32,
    take_sqrt_in: Int32,
):
    """One block per row, block sum of squares, optional square root."""
    var n_cols = Int(n_cols_in)
    var row = Int(block_idx.x)
    var tid = Int(thread_idx.x)

    var acc = Float32(0.0)
    var col = tid
    while col < n_cols:
        var v = ftz(a.unsafe_load(row * n_cols + col))
        # `acc += v * v`, with the contraction pinned under IDENTICAL
        # (row 9) and the running partial flushed under row 10's rule that
        # a pinned expression stores its intermediates through `ftz`.
        acc = ftz(identical_mul_add(v, v, acc))
        col += NORM_TPB

    # `cub::BlockReduce`'s counterpart. Under FAST this IS
    # `max.gpu.primitives.block.sum`, bit for bit -- the reduction shape
    # stays Modular's to tune. Under IDENTICAL it is the halving tree that
    # no lane width can reach; see `core/pinned_reduce.mojo`.
    var s0 = pinned_block_sum[NORM_TPB](acc)

    if tid == 0:
        var total = ftz(s0)
        if take_sqrt_in != 0:
            if total <= Float32(0.0):
                total = Float32(0.0)
            # PINNED. This was the stdlib `sqrt`, and it was the ONE
            # unpinned operation in a kernel whose every other step is
            # `identical_mul_add`, `ftz` and `pinned_block_sum`. On NVIDIA
            # the stdlib routes to the approximate PTX square root
            # (DEVIATION 258), so a norm computed here could differ by an
            # ulp from Apple's while every surrounding stage agreed. That is
            # exactly the one-ulp, sqrt-localized divergence the identity
            # card was built to find, and leaving it here meant the tree
            # shipped a live instance of its own worked example.
            #
            # `identical_sqrt` branches on the mode itself, taking
            # `portable_sqrtf` under IDENTICAL and the stdlib otherwise, so
            # FAST is byte-for-byte unchanged and only the strict tier moves.
            # Apple's stdlib sqrt is correctly rounded and so is
            # `portable_sqrtf`, so Apple's cards should not move either; the
            # column this fixes is NVIDIA.
            total = ftz(identical_sqrt(total))
        out_norm.unsafe_store(row, total)


# C06 schedules independent rows in one block. Each row retains NORM_TPB
# logical lanes and the incumbent halving fold. 2/4 rows are independent
# arms: 2/4 times the shared storage, fewer blocks, no shape-specific route.
# NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
from experiments.classical_identical_ideas.shared_controls import C06_ROWS2, C06_ROWS4
from max.gpu.host import DeviceContext, DeviceBuffer
comptime CLASSICAL_NORM_ROWS = 4 if C06_ROWS4 else (2 if C06_ROWS2 else 1)


def batched_row_norm_kernel[rows_per_block: Int](
    out_norm: MutPointer[Float32, MutAnyOrigin],
    a: MutPointer[Float32, MutAnyOrigin], n_rows_in: Int32,
    n_cols_in: Int32, take_sqrt_in: Int32,
):
    var lane = Int(thread_idx.x)
    var local_row = Int(thread_idx.y)
    var row = Int(block_idx.x) * rows_per_block + local_row
    var slab = stack_allocation[NORM_TPB * rows_per_block, Float32, address_space=AddressSpace.SHARED]()
    var acc = Float32(0)
    if row < Int(n_rows_in):
        var col = lane
        while col < Int(n_cols_in):
            var v = ftz(a.unsafe_load(row * Int(n_cols_in) + col))
            acc = ftz(identical_mul_add(v, v, acc))
            col += NORM_TPB
    var at = local_row * NORM_TPB + lane
    slab[at] = acc
    barrier()
    var step = NORM_TPB // 2
    while step > 0:
        if lane < step:
            slab[at] = ftz(slab[at] + slab[at + step])
        barrier()
        step //= 2
    if lane == 0 and row < Int(n_rows_in):
        var total = ftz(slab[at])
        if take_sqrt_in != 0:
            if total <= Float32(0):
                total = Float32(0)
            total = ftz(identical_sqrt(total))
        out_norm.unsafe_store(row, total)


def enqueue_row_norms(
    ctx: DeviceContext, mut output: DeviceBuffer[DType.float32],
    mut values: DeviceBuffer[DType.float32], rows: Int, cols: Int,
    take_sqrt: Int = 0,
) raises:
    if rows <= 0:
        return
    comptime if C06_ROWS2 or C06_ROWS4:
        ctx.enqueue_function[batched_row_norm_kernel[CLASSICAL_NORM_ROWS]](
            output.unsafe_ptr(), values.unsafe_ptr(), Int32(rows), Int32(cols), Int32(take_sqrt),
            grid_dim=(rows + CLASSICAL_NORM_ROWS - 1) // CLASSICAL_NORM_ROWS,
            block_dim=(NORM_TPB, CLASSICAL_NORM_ROWS),
        )
    else:
        ctx.enqueue_function[row_norm_kernel](
            output.unsafe_ptr(), values.unsafe_ptr(), Int32(cols), Int32(take_sqrt),
            grid_dim=rows, block_dim=NORM_TPB,
        )
