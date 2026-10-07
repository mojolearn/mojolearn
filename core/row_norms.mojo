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
# logical lanes and the incumbent halving fold. One control,
# MOJOLEARN_CLASSICAL_C06_NORM_ROWS=2|4 (shared_controls.mojo); no
# shape-specific route. NOT MEASURED.
# C06 small-d arm (MOJOLEARN_CLASSICAL_C06_SMALL_D_THREAD): one thread per row
# while d <= C06_THREAD_MAX_D, replaying the incumbent tree (see the kernel).
from experiments.classical_identical_ideas.shared_controls import C06_NORM_ROWS, C06_ROWS_ON, C06_SMALL_D_THREAD
from max.gpu.host import DeviceContext, DeviceBuffer
comptime CLASSICAL_NORM_ROWS = C06_NORM_ROWS
#: Cost bound of the small-d arm: one thread per row costs d fma + 31 adds;
#: past 32 columns a NORM_TPB-lane block per row keeps a quarter of its lanes
#: busy and the block form is kept. Not a board shape (the bound is the
#: register replay's width, 2 x 16 tree lanes).
comptime C06_THREAD_MAX_D = 32
comptime C06_THREAD_TPB = 128


def thread_row_norm_kernel(
    out_norm: MutPointer[Float32, MutAnyOrigin],
    a: MutPointer[Float32, MutAnyOrigin], n_rows_in: Int32,
    n_cols_in: Int32, take_sqrt_in: Int32,
):
    """`row_norm_kernel`'s value for one row, in one thread, for
    `n_cols <= C06_THREAD_MAX_D`. The incumbent gives lane j the single term
    `ftz(fma(v_j, v_j, 0))` (lanes past d hold 0) and folds with
    `two_phase_halving_sum[NORM_TPB]`: phase 1 adds lanes t + 16j for j < G in
    a halving tree; with only lanes < 32 nonzero that tree's last step is
    `lane t + lane t + 16`, every other add being `x + 0 = x` (terms are >= +0).
    Phase 2 is the 16-lane halving tree. Replayed here term for term, so the
    bits are the incumbent's under IDENTICAL."""
    comptime assert NORM_TPB >= 32 and (NORM_TPB & (NORM_TPB - 1)) == 0, (
        "the replay needs NORM_TPB a power of two >= 32"
    )
    var row = Int(block_idx.x) * C06_THREAD_TPB + Int(thread_idx.x)
    if row >= Int(n_rows_in):
        return
    var d = Int(n_cols_in)
    var lanes = InlineArray[Float32, 32](fill=Float32(0.0))
    comptime for j in range(32):
        if j < d:
            var v = ftz(a.unsafe_load(row * d + j))
            lanes[j] = ftz(identical_mul_add(v, v, Float32(0.0)))
    var w = InlineArray[Float32, 16](fill=Float32(0.0))
    comptime for t in range(16):
        w[t] = lanes[t] + lanes[t + 16]
    comptime for k in range(4):
        comptime S = 8 >> k
        comptime for t in range(S):
            w[t] = w[t] + w[t + S]
    var total = ftz(w[0])
    if take_sqrt_in != 0:
        if total <= Float32(0):
            total = Float32(0)
        total = ftz(identical_sqrt(total))
    out_norm.unsafe_store(row, total)


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
    comptime if C06_SMALL_D_THREAD:
        if cols <= C06_THREAD_MAX_D:
            ctx.enqueue_function[thread_row_norm_kernel](
                output.unsafe_ptr(), values.unsafe_ptr(), Int32(rows), Int32(cols), Int32(take_sqrt),
                grid_dim=(rows + C06_THREAD_TPB - 1) // C06_THREAD_TPB,
                block_dim=C06_THREAD_TPB,
            )
            return
    comptime if C06_ROWS_ON:
        comptime assert CLASSICAL_NORM_ROWS == 2 or CLASSICAL_NORM_ROWS == 4, (
            "MOJOLEARN_CLASSICAL_C06_NORM_ROWS takes 2 or 4"
        )
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
