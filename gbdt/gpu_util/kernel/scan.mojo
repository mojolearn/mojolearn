# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`ScanVector` over `ui32`: the unsegmented device-wide prefix sum.

Reference: `catboost/cuda/cuda_util/kernel/scan.cu:10-19` (`NKernel::ScanVector`)
reached from `cuda_util/scan.h:14-60` (`TScanVectorKernel`, the
`IsNonNegativeSegmentedScan == false` arm at `:52-58`) (CatBoost `54a8143a`).

The reference is two lines:

    if (inclusive) return cub::DeviceScan::InclusiveSum(..., input, output, size, ...);
    else           return cub::DeviceScan::ExclusiveSum(..., input, output, size, ...);

## Where the CTR block needs it, and it is not where you would guess

`TCtrBinBuilder::ComputeCurrentBins` (`ctrs/ctr_bins_builder.h:134-146`):

    ExtractMask(indices, dst, false)   // END flags, 1 per segment end
    ScanVector(dst, tmp, false)        // EXCLUSIVE  <-- this file
    ScatterWithMask(dst, tmp, indices, Mask)

The exclusive scan of END flags at sorted position `i` counts the segments
that closed strictly before `i`, which is `i`'s own segment index; the
scatter moves it from sorted position back to original row. That number is
`UpdateBordersMask`'s third test, the one that keeps a feature TENSOR
segmented on the whole combination rather than on the newest feature alone.

For a SIMPLE ctr the answer is all zeros -- the builder starts with no flags
set, so only the last element is an end and nothing precedes it. This file
runs anyway because their code runs, and because the day tree CTRs land the
answer stops being zero and nothing would have said so.

# =========================================================================
# DEVIATION BLOCK: the device-wide decoupling is written out, exactly as in
# `reorder_one_bit.mojo` and `segmented_scan.mojo`. CUB does it internally
# with a decoupled lookback; MAX ships no device-wide scan (archive/reference/VENDOR_LIBS.md
# 3b/3c, re-checked 2026-08-20), so there is nothing to call. Phase 1 is
# `max.gpu.primitives.block.prefix_sum`, Modular's counterpart to
# `cub::BlockScan::ExclusiveSum`, which is the collective CUB itself uses
# per block. Phase 2 scans the per-block totals with ONE THREAD.
#
# This is the THIRD copy of that three-phase shape in this tree, and the
# reason it is a copy rather than a shared function is that the three differ
# in the only place that matters: `reorder_one_bit` scans a `ui8` flag,
# `segmented_scan` scans under a custom operator with a reset, and this one
# scans a `ui32` value. CUB spells the same difference as three template
# instantiations.
# =========================================================================

An exclusive `ui32` sum overflows above 2^32; theirs does too (`ScanVector`
is instantiated at `ui32` with a `ui32` output, `scan.cu:98-104`), and the
only input this implementation scans is a 0/1 flag per row.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.primitives.block import prefix_sum

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL

#: FAST on Apple, lane/apple-fast-trees-scan (2026-10-02), OPT-IN
#: `-D MOJOLEARN_SCAN_U32_BLOCK=1`. CAUSE: `launch_scan_vector_u32` runs
#: `scan_block_sums_u32_kernel` as ONE THREAD (grid 1 x block 1, this
#: file:~174 on main) over `ceil(size / 512)` block totals, on the CTR
#: path (`ctr_bins_builder.compute_current_bins`,
#: `ctr_calcers.visit_equal_up_to_prior_freq_ctrs`: taxi 4.1M rows is
#: ~8,000 dependent round trips per call). EFFECT: with the define the scan
#: runs on one `SCAN_SUMS_TPB`-thread block, stripe per thread, one
#: `prefix_sum` for the stripe carries. Integer sums: the same prefixes.
#: IDENTICAL compiles the old launch.
comptime SCAN_SUMS_TPB = 256
comptime SCAN_U32_BLOCK_SUMS = (
    GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL
    and has_apple_gpu_accelerator()
    and is_defined["MOJOLEARN_SCAN_U32_BLOCK"]()
)


comptime SCAN_BLOCK = 512
"""The block. Not `GetScanBlockSize()`'s 768: that constant belongs to the
SEGMENTED family (`cuda_util/kernel/scan.cuh:7`, read by
`segmented_scan.mojo`), and CUB picks its own tile for `DeviceScan`. 512 is
the block `reorder_one_bit.mojo` already runs its scan phases at, which is
their `blockSize` for the reorder family (`reorder_one_bit.cu:35`); reusing
it keeps one geometry for every unsegmented scan in the tree. No shared
memory is claimed beyond what `prefix_sum` itself uses, so there is no
budget to query and no kernel-matrix row."""


def scan_block_u32_kernel(
    values: MutPointer[UInt32, MutAnyOrigin],
    size_in: Int32,
    scanned: MutPointer[UInt32, MutAnyOrigin],
    block_sums: MutPointer[UInt32, MutAnyOrigin],
):
    """Phase 1: the block-local EXCLUSIVE sum, plus each block's total.

    A thread past `size` contributes 0, so the last thread's exclusive
    prefix plus its own value is the block total whether or not the block is
    full -- the padding argument `block_scan_flags_kernel` makes.
    """
    var size = Int(size_in)
    var tid = Int(thread_idx.x)
    var start = Int(block_idx.x) * SCAN_BLOCK

    var v = Int32(0)
    if start + tid < size:
        v = Int32(Int(values.unsafe_load(start + tid)))

    var exclusive = prefix_sum[block_size=SCAN_BLOCK, exclusive=True](v)

    if start + tid < size:
        scanned.unsafe_store(start + tid, UInt32(Int(exclusive)))
    if tid == SCAN_BLOCK - 1:
        block_sums.unsafe_store(
            Int(block_idx.x), UInt32(Int(exclusive + v))
        )


def scan_block_sums_u32_kernel(
    block_sums: MutPointer[UInt32, MutAnyOrigin], n_blocks_in: Int32
):
    """Phase 2: exclusive scan of the per-block totals, one thread."""
    if Int(thread_idx.x) != 0 or Int(block_idx.x) != 0:
        return
    var n = Int(n_blocks_in)
    var running = UInt32(0)
    for b in range(n):
        var v = block_sums.unsafe_load(b)
        block_sums.unsafe_store(b, running)
        running += v


def scan_block_sums_u32_block_kernel(
    block_sums: MutPointer[UInt32, MutAnyOrigin], n_blocks_in: Int32
):
    """`scan_block_sums_u32_kernel` with `SCAN_SUMS_TPB` threads in ONE
    block: thread t sums a contiguous stripe of the totals, the block's
    exclusive `prefix_sum` gives each stripe its carry, and the stripe
    writes its exclusive prefixes (the form `reorder_one_bit.mojo`'s
    `scan_block_sums_parallel_kernel` runs). The totals are counts below
    `size`, so they ride through `prefix_sum`'s Int32 exactly as
    `scan_block_u32_kernel`'s do. Launch with `grid_dim=1,
    block_dim=SCAN_SUMS_TPB`."""
    var n = Int(n_blocks_in)
    var tid = Int(thread_idx.x)
    var per = (n + SCAN_SUMS_TPB - 1) // SCAN_SUMS_TPB
    var lo = min(tid * per, n)
    var hi = min(lo + per, n)
    var local = Int32(0)
    for b in range(lo, hi):
        local += Int32(Int(block_sums.unsafe_load(b)))
    var carry = prefix_sum[block_size=SCAN_SUMS_TPB, exclusive=True](local)
    var running = UInt32(Int(carry))
    for b in range(lo, hi):
        var v = block_sums.unsafe_load(b)
        block_sums.unsafe_store(b, running)
        running += v


def scan_add_carry_u32_kernel(
    scanned: MutPointer[UInt32, MutAnyOrigin],
    block_sums: MutPointer[UInt32, MutAnyOrigin],
    size_in: Int32,
):
    """Phase 3: add each block's carry to its elements."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(size_in):
        scanned.unsafe_store(
            i,
            scanned.unsafe_load(i)
            + block_sums.unsafe_load(Int(block_idx.x)),
        )


def scan_to_inclusive_u32_kernel(
    values: MutPointer[UInt32, MutAnyOrigin],
    scanned: MutPointer[UInt32, MutAnyOrigin],
    size_in: Int32,
):
    """`cub::DeviceScan::InclusiveSum` from the exclusive answer: add each
    element's own value back. In place on `scanned`."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(size_in):
        scanned.unsafe_store(
            i, scanned.unsafe_load(i) + values.unsafe_load(i)
        )


def launch_scan_vector_u32(
    ctx: DeviceContext,
    size: Int,
    inclusive: Bool,
    mut values: DeviceBuffer[DType.uint32],
    mut output: DeviceBuffer[DType.uint32],
    mut block_sums: DeviceBuffer[DType.uint32],
) raises:
    """`ScanVector<ui32, ui32>` (`cuda_util/scan.h:52-58`, `scan.cu:10-19`).

    `block_sums` must hold at least `ceil(size / SCAN_BLOCK)` entries; it is
    their `context.PartResults`, allocated by `PrepareContext` from
    `ScanVectorTempSize` (`scan.h:26-31`) and likewise passed in rather than
    allocated inside.

    `values` and `output` must be DIFFERENT buffers: theirs are too
    (`ComputeCurrentBins` passes `dst` and `tmp`), and the inclusive arm
    reads `values[i]` after phase 3 has written `output[i]`.
    """
    if size <= 0:
        return
    var n_blocks = (size + SCAN_BLOCK - 1) // SCAN_BLOCK

    ctx.enqueue_function[scan_block_u32_kernel](
        values.unsafe_ptr(), Int32(size), output.unsafe_ptr(),
        block_sums.unsafe_ptr(),
        grid_dim=n_blocks, block_dim=SCAN_BLOCK,
    )
    comptime if SCAN_U32_BLOCK_SUMS:
        # one 256-thread block (lane/apple-fast-trees-scan)
        ctx.enqueue_function[scan_block_sums_u32_block_kernel](
            block_sums.unsafe_ptr(), Int32(n_blocks),
            grid_dim=1, block_dim=SCAN_SUMS_TPB,
        )
    else:
        ctx.enqueue_function[scan_block_sums_u32_kernel](
            block_sums.unsafe_ptr(), Int32(n_blocks),
            grid_dim=1, block_dim=1,
        )
    ctx.enqueue_function[scan_add_carry_u32_kernel](
        output.unsafe_ptr(), block_sums.unsafe_ptr(), Int32(size),
        grid_dim=n_blocks, block_dim=SCAN_BLOCK,
    )
    if inclusive:
        ctx.enqueue_function[scan_to_inclusive_u32_kernel](
            values.unsafe_ptr(), output.unsafe_ptr(), Int32(size),
            grid_dim=n_blocks, block_dim=SCAN_BLOCK,
        )
