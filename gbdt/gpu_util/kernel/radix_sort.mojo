"""LSD radix sort over `(bin, permutation position)`: their `ReorderBins`.

PORT OF the path `TCtrBinBuilder` takes to put the rows in CTR order,
`catboost/cuda/ctrs/ctr_bins_builder.h:223`:

    GatherWithMask(Bins, DecompressedTempBins, Indices, Mask, Stream);
    const ui32 newBits = NCB::IntLog2(uniqueValues);
    ReorderBins(Bins, Indices, 0, newBits, Tmp, DecompressedTempBins, Stream);
    UpdateBordersMask(Bins, currentBins, Indices, Stream);

`ReorderBins` (`cuda_util/sort.cpp:544`) is `TRadixSortKernel<ui32, ui32>`
with `descending=false`, `firstBit=offset`, `lastBit=offset + bits` --
keys are the cat bins, values are the permutation `Indices`.

## Why the sort has to be STABLE, and what it is sorting by

The brief for this block calls the key `(bin || permutationPosition)`, and
that composite key is never materialized anywhere in CatBoost. It does not
have to be. `Indices` arrives in learn-permutation order, and a STABLE sort
by bin leaves each bin's rows in that order, which IS the second half of
the key. The whole ordered-target-statistic story downstream --
`SegmentedScanAndScatterNonNegativeVector` summing "the rows of my bin that
came before me" -- is true only if the tie order is the permutation order.
So stability is not a quality of the implementation here, it is half the
algorithm, and `mojo_only/radix_sort_check.mojo` gates it separately from
sortedness for that reason.

## What their dispatch calls, and what is written here instead

# =========================================================================
# DEVIATION BLOCK: theirs is `cub::DeviceRadixSort::SortPairs`
# (`cuda_util/kernel/sort_templ.cuh:26`). CUB is OPEN and therefore a port
# candidate under PORTING_RULES 0b-i, and MAX ships no device sort
# (VENDOR_LIBS.md, re-checked 2026-08-20), so there is nothing to call.
#
# What is written here is NOT a fresh design: it is CatBoost's OWN
# `NKernel::ReorderOneBit<ui32, ui32>` (`cuda_util/kernel/reorder_one_bit.cu
# :11`, the `<ui32, ui32>` instantiation at `:61`) looped over the bit
# range. Their `ReorderOneBit<TMapping>` (`cuda_util/reorder_bins.cpp:74`)
# is the single-bit spelling of the same operation on the same buffers, so
# the pass is theirs and the LOOP is ours.
#
# Three differences from their CUB call, each named:
#
# 1. ONE BIT PER PASS, where CUB's SortPairs does a radix DIGIT per pass
#    (4 to 8 bits, a shared-memory histogram per digit). The answer is
#    identical -- LSD radix over stable one-bit passes is stable and total
#    -- and the cost is passes: `bits` of them rather than `bits/5`. At the
#    CTR shape `newBits = IntLog2(uniqueValues)`, so a 1000-category
#    feature is 10 passes of 4 launches. Priced, not measured; nothing
#    calls this yet.
# 2. PING-PONG, not a copy per pass. Their single-bit wrapper memcpys the
#    live buffers into temporaries before every pass
#    (`reorder_one_bit.cu:21-22`) because its API writes back in place. A
#    multi-pass driver does not need that and CUB does not do it either:
#    `sort_templ.cuh:10` builds a `cub::DoubleBuffer` and copies back only
#    at the end, `if (doubleBufferKeys.Current() != keys)` (`:53`). That
#    final conditional copy is mirrored below and is the odd/even switch
#    the check exercises on both sides.
# 3. NO DESCENDING ARM. `TRadixSortContext::Descending`
#    (`cuda_util/kernel/sort.cuh:25`) exists, and `ReorderBinsImpl` passes
#    `false` (`sort.cpp:558`). An unported arm nothing dispatches to is
#    left unported rather than written blind.
# =========================================================================

## What is reused rather than rewritten

Phases 2 and 3 of the device-wide scan -- `scan_block_sums_kernel` and
`add_block_carry_kernel` -- are imported from `reorder_one_bit.mojo`
unchanged, and so is `REORDER_BLOCK`, their `blockSize = 512`
(`reorder_one_bit.cu:35`). Only phase 1 differs, and it differs the way it
differs in THEIR code: `SortWithoutCub` scans a flag array while
`ReorderOneBit` scans `TScanBitIterator<ui32>`
(`reorder_one_bit_impl.cuh:109`, `(Ldg(Bins + n) >> Bit) & 1`), which is
the same `cub::DeviceScan::ExclusiveSum` over a different input iterator.
"""

from std.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.primitives.block import prefix_sum

from gbdt.gpu_util.copy import COPY_BLOCK, copy_u32_kernel
from gbdt.gpu_util.kernel.reorder_one_bit import (
    REORDER_BLOCK,
    REORDER_UNROLL,
    add_block_carry_kernel,
    scan_block_sums_kernel,
)


def scan_key_bit_kernel(
    keys: MutPointer[UInt32, MutAnyOrigin],
    bit_in: Int32,
    size_in: Int32,
    offsets: MutPointer[Int32, MutAnyOrigin],
    block_sums: MutPointer[Int32, MutAnyOrigin],
):
    """Phase 1 over `TScanBitIterator<ui32>` (`reorder_one_bit_impl.cuh:104
    -110`) instead of over a flag byte.

    Their line is
        `cub::DeviceScan::ExclusiveSum<TInput, int*>(..., inputIter, offsets,
        size, stream)` (`reorder_one_bit.cu:27-32`)
    where `inputIter[n]` is `(static_cast<ui32>(Ldg(Bins + n)) >> Bit) & 1`.
    The block-local half is `max.gpu.primitives.block.prefix_sum`, the same
    call `block_scan_flags_kernel` makes; a thread past `size` contributes
    zero so the last thread's prefix plus its own bit is the block total
    either way.
    """
    var size = Int(size_in)
    var bit = Int(bit_in)
    var tid = Int(thread_idx.x)
    var start = Int(block_idx.x) * REORDER_BLOCK

    var v = Int32(0)
    if start + tid < size:
        v = Int32((Int(keys.unsafe_load(start + tid)) >> bit) & 1)

    var exclusive = prefix_sum[block_size=REORDER_BLOCK, exclusive=True](v)

    if start + tid < size:
        offsets.unsafe_store(start + tid, exclusive)
    if tid == REORDER_BLOCK - 1:
        block_sums.unsafe_store(Int(block_idx.x), exclusive + v)


def reorder_one_bit_u32_kernel(
    temp_keys: MutPointer[UInt32, MutAnyOrigin],
    temp_values: MutPointer[UInt32, MutAnyOrigin],
    offsets: MutPointer[Int32, MutAnyOrigin],
    bit_in: Int32,
    keys: MutPointer[UInt32, MutAnyOrigin],
    values: MutPointer[UInt32, MutAnyOrigin],
    size_in: Int32,
):
    """`ReorderOneBitImpl<ui32, ui32, N=1, BlockSize=512>`
    (`reorder_one_bit_impl.cuh:127`), the `<ui32, ui32>` instantiation.

    Identical arithmetic to `reorder_one_bit_kernel` in
    `reorder_one_bit.mojo`, which is the `<bool, ui32>` one; that file's
    header carries the derivation. The two exist separately because their
    template does, and because this one has no leaf `base`: their
    `ReorderBins` sorts a whole buffer, their `SortByFlagsInLeaf` sorts a
    slice of one.
    """
    var size = Int(size_in)
    var bit = Int(bit_in)

    # `const int totalOnes = __ldg(offsets + size - 1)
    #                      + ((Ldg(tempKeys + size - 1) >> bit) & 1);`
    var last_flag = Int32((Int(temp_keys.unsafe_load(size - 1)) >> bit) & 1)
    var total_ones = offsets.unsafe_load(size - 1) + last_flag
    var total_zeros = Int32(size) - total_ones

    var i = Int(block_idx.x) * REORDER_BLOCK + Int(thread_idx.x)

    comptime for k in range(REORDER_UNROLL):
        var idx = i + k * REORDER_BLOCK
        if idx < size:
            var ones_before = offsets.unsafe_load(idx)
            var key = temp_keys.unsafe_load(idx)
            var val = temp_values.unsafe_load(idx)
            var zeroes_before = Int32(idx) - ones_before

            var is_zero = ((Int(key) >> bit) & 1) == 0
            var dst = zeroes_before if is_zero else (
                total_zeros + ones_before
            )
            keys.unsafe_store(Int(dst), key)
            values.unsafe_store(Int(dst), val)


def _radix_pass(
    ctx: DeviceContext,
    size: Int,
    bit: Int,
    n_blocks: Int,
    mut src_keys: DeviceBuffer[DType.uint32],
    mut src_values: DeviceBuffer[DType.uint32],
    mut dst_keys: DeviceBuffer[DType.uint32],
    mut dst_values: DeviceBuffer[DType.uint32],
    mut offsets: DeviceBuffer[DType.int32],
    mut block_sums: DeviceBuffer[DType.int32],
) raises:
    """One pass of `NKernel::ReorderOneBit<ui32, ui32>`
    (`reorder_one_bit.cu:11`): scan the bit, then reorder by it.

    Their `cudaMemcpyAsync` prologue (`:21-22`) is absent by design; see
    the ping-pong note in the deviation block. Everything after it is
    theirs launch for launch.
    """
    ctx.enqueue_function[scan_key_bit_kernel](
        src_keys.unsafe_ptr(), Int32(bit), Int32(size),
        offsets.unsafe_ptr(), block_sums.unsafe_ptr(),
        grid_dim=n_blocks, block_dim=REORDER_BLOCK,
    )
    ctx.enqueue_function[scan_block_sums_kernel](
        block_sums.unsafe_ptr(), Int32(n_blocks),
        grid_dim=1, block_dim=1,
    )
    ctx.enqueue_function[add_block_carry_kernel](
        offsets.unsafe_ptr(), block_sums.unsafe_ptr(), Int32(size),
        grid_dim=n_blocks, block_dim=REORDER_BLOCK,
    )
    ctx.enqueue_function[reorder_one_bit_u32_kernel](
        src_keys.unsafe_ptr(), src_values.unsafe_ptr(),
        offsets.unsafe_ptr(), Int32(bit),
        dst_keys.unsafe_ptr(), dst_values.unsafe_ptr(), Int32(size),
        grid_dim=n_blocks, block_dim=REORDER_BLOCK,
    )


def launch_radix_sort_bins(
    ctx: DeviceContext,
    size: Int,
    first_bit: Int,
    last_bit: Int,
    mut keys: DeviceBuffer[DType.uint32],
    mut values: DeviceBuffer[DType.uint32],
    mut temp_keys: DeviceBuffer[DType.uint32],
    mut temp_values: DeviceBuffer[DType.uint32],
    mut offsets: DeviceBuffer[DType.int32],
    mut block_sums: DeviceBuffer[DType.int32],
) raises:
    """`ReorderBins(bins, indices, offset, bits, tmpBins, tmpIndices)`
    (`cuda_util/sort.cpp:544`), with `bits = last_bit - first_bit`.

    Their guards, both of them: `if (bits == 0) return;` (`sort.cpp:553`)
    and `CB_ENSURE((offset + bits) <= 32)` (`:557`). The empty-input guard
    is `if (size)` from `ReorderOneBit` (`reorder_one_bit.cu:20`), and it
    also keeps `offsets[size - 1]` in the reorder from reading before the
    buffer.

    `keys` and `values` hold the answer on return, whichever side of the
    ping-pong the last pass wrote.
    """
    if size <= 0:
        return
    if last_bit <= first_bit:
        return
    if last_bit > 32 or first_bit < 0:
        raise Error(
            String("radix sort bit range out of a ui32: [")
            + String(first_bit) + ", " + String(last_bit) + ")"
        )

    var n_blocks = (size + REORDER_BLOCK - 1) // REORDER_BLOCK

    # `cub::DoubleBuffer<K> doubleBufferKeys(keys, context.GetTempKeys<K>())`
    # (`sort_templ.cuh:10`), and the same for the values (`:15`). Mojo will
    # not let two device pointers of different origins swap places in one
    # variable, so the buffers are selected by the parity instead of the
    # pointers being exchanged. Same double buffer, said differently.
    var flip = False
    for bit in range(first_bit, last_bit):
        if flip:
            _radix_pass(
                ctx, size, bit, n_blocks, temp_keys, temp_values,
                keys, values, offsets, block_sums,
            )
        else:
            _radix_pass(
                ctx, size, bit, n_blocks, keys, values,
                temp_keys, temp_values, offsets, block_sums,
            )
        flip = not flip

    # `if (doubleBufferKeys.Current() != keys) cudaMemcpyAsync(...)`
    # (`sort_templ.cuh:53`), and the same for the values (`:33`). An odd
    # number of passes leaves the answer in the temporaries.
    if flip:
        var copy_blocks = (size + COPY_BLOCK - 1) // COPY_BLOCK
        ctx.enqueue_function[copy_u32_kernel](
            keys.unsafe_ptr(), temp_keys.unsafe_ptr(), Int32(size),
            grid_dim=copy_blocks, block_dim=COPY_BLOCK,
        )
        ctx.enqueue_function[copy_u32_kernel](
            values.unsafe_ptr(), temp_values.unsafe_ptr(), Int32(size),
            grid_dim=copy_blocks, block_dim=COPY_BLOCK,
        )
