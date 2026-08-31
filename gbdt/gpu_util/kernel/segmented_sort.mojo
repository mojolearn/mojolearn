# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""LSD radix sort WITHIN each of a partition's contiguous segments.

PORT OF `catboost/cuda/cuda_util/kernel/segmented_sort.{cu,cuh}` at CatBoost
`54a8143a` -- their `NKernel::SegmentedRadixSort<K, V>` and the
`TSegmentedRadixSortContext` that carries its bit range. THIS FILE IS THE
MIRROR ADDRESS FOR THAT ONE, which is why it is named for their file rather
than for what it does.

The API above it is `cuda_util/segmented_sort.{h,cpp}`; the caller in this
port's reach is `ComputeWeightedQuantile`
(`methods/leaves_estimation/leaves_estimation_helper.h:110-112`):

    SegmentedRadixSort(orderedTargets, orderedWeights, tmpTargets,
                       tmpWeights, binsOffsets, binCount, 10, 32);

# =========================================================================
# DEVIATION 65: their whole file is a CUB call. `NKernel::SegmentedRadixSort`
# (`segmented_sort.cu:7-45`) sets up two `cub::DoubleBuffer`s and hands them
# to `cub::DeviceSegmentedRadixSort::SortPairs`, so the body being ported
# here is the SortPairs itself. CUB is OPEN and therefore a port candidate
# under PORTING_RULES 0b-i, and MAX ships no device sort at all
# (VENDOR_LIBS.md, re-checked 2026-08-20), so there is nothing to call.
#
# Their `Descending` arm (`SortPairsDescending`, `:29-36`) is NOT PORTED:
# `ComputeWeightedQuantile` passes the default `compareGreater = false`
# (`segmented_sort.h:14`), and PORTING_RULES 3 says a branch no caller
# reaches is not done. `TSegmentedRadixSortContext`'s `FirstBit`/`LastBit`
# are the two arguments `launch_segmented_radix_sort` takes.
#
# What is written here is NOT a fresh design. It is the SAME construction
# `gbdt/gpu_util/kernel/radix_sort.mojo` already carries -- CatBoost's own
# `NKernel::ReorderOneBit` (`cuda_util/kernel/reorder_one_bit.cu:11`)
# looped over a bit range -- with one axis added: their own driver already
# runs that reorder PER LEAF (`SortWithoutCub`, `split_points.cu:692-700`,
# called once per part), so a segment dimension is a shape their code has
# and not a structure this file invented. The three kernels below are
# `radix_sort.mojo`'s three with `block_idx.y` as the segment, and the
# fourth is the per-segment block-sum scan their per-leaf call gets for
# free by being per-leaf.
#
# WHAT THIS BUYS OVER THE PER-LEAF LOOP, and it is the reason for the
# axis: their driver's shape costs `4 * bits * segments` launches. At
# depth 6 and their bit range that is 4 * 22 * 64 = 5,632 launches per
# tree on a machine where a launch is ~30 us and the whole per-tree fixed
# cost budget is 9.43 ms (`PERF_2026-08-20_fixed-cost.md`). Batched it is
# 4 * 22 = 88, independent of depth. No arithmetic differs; every row
# lands in the same slot either way, because a segment never reads or
# writes outside `[offset, offset + size)` in either shape.
# =========================================================================

## THE KEY TRANSFORM, which is CUB's and not ours

A radix sort orders by unsigned integer value, and IEEE-754 float bits do
not order that way: negatives run backwards and `-0.0` sorts above `+0.0`.
CUB handles this in `NumericTraits<float>::Digit`, and passing floats to
`DeviceSegmentedRadixSort` applies it invisibly -- which is why their call
site sorts `float` keys and says nothing about bit twiddling.

`float_to_sortable` below is that transform:

    negative (sign set)  ->  flip every bit
    positive             ->  flip the sign bit only

It is a monotone bijection onto `ui32`, so sorting the transformed key is
sorting the float, and `sortable_to_float` inverts it exactly.

## THEIR BIT RANGE IS NOT THE WHOLE KEY, and that is deliberate

`(10, 32)` drops the bottom TEN mantissa bits, so their exact leaf value is
the weighted quantile of the targets *rounded to about four significant
decimal digits*. It is theirs; it is copied; and it is the second place
their exact estimator is approximate, the first being the sixteen-iteration
binary search that follows it (`exact_estimation.cu:31-39`).
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.memory import bitcast
from max.gpu.primitives.block import prefix_sum
from std.gpu import block_dim, block_idx, thread_idx

from gbdt.gpu_util.kernel.reorder_one_bit import REORDER_BLOCK


@always_inline
def float_to_sortable(bits: UInt32) -> UInt32:
    """CUB's `NumericTraits<float>::Digit`, the forward half."""
    if (bits & UInt32(0x80000000)) != UInt32(0):
        return ~bits
    return bits | UInt32(0x80000000)


@always_inline
def sortable_to_float(key: UInt32) -> UInt32:
    """The inverse. Used by the check, not by the sort."""
    if (key & UInt32(0x80000000)) != UInt32(0):
        return key & UInt32(0x7FFFFFFF)
    return ~key


def float_key_kernel(
    src: MutPointer[Float32, MutAnyOrigin],
    size_in: Int32,
    keys: MutPointer[UInt32, MutAnyOrigin],
    values: MutPointer[UInt32, MutAnyOrigin],
):
    """Materialize the sortable key and the identity payload.

    Their call passes `float` keys and `float` values straight to CUB.
    Ours sorts `(ui32 key, ui32 index)` and gathers the two float columns
    afterwards, which is the same permutation applied to the same rows --
    and is what their `ComputeByLeafOrder` already does one step earlier
    (`leaves_estimation_helper.h:36-49`: sort bins, carry `indices`,
    `Gather` the columns by them).
    """
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(size_in):
        return
    keys.unsafe_store(
        i, float_to_sortable(bitcast[DType.uint32](src.unsafe_load(i)))
    )
    values.unsafe_store(i, UInt32(i))


def gather_f32_by_index_kernel(
    src: MutPointer[Float32, MutAnyOrigin],
    index: MutPointer[UInt32, MutAnyOrigin],
    size_in: Int32,
    dst: MutPointer[Float32, MutAnyOrigin],
):
    """`Gather(orderedTargets, singleDevTargets, indices)`
    (`leaves_estimation_helper.h:99`, `:102`)."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(size_in):
        return
    dst.unsafe_store(i, src.unsafe_load(Int(index.unsafe_load(i))))


def seg_scan_key_bit_kernel(
    keys: MutPointer[UInt32, MutAnyOrigin],
    bit_in: Int32,
    seg_offsets: MutPointer[UInt32, MutAnyOrigin],
    seg_sizes: MutPointer[UInt32, MutAnyOrigin],
    blocks_wide_in: Int32,
    offsets: MutPointer[Int32, MutAnyOrigin],
    block_sums: MutPointer[Int32, MutAnyOrigin],
):
    """`scan_key_bit_kernel` (`radix_sort.mojo`) with the segment axis.

    `block_idx.y` is the segment, exactly as their per-leaf driver's outer
    loop index is (`split_points.cu:694`). Rows outside this segment's
    `[offset, offset + size)` contribute a zero bit and store nothing, so
    the block total is right whether or not the segment fills its blocks.
    """
    var seg = Int(block_idx.y)
    var base = Int(seg_offsets.unsafe_load(seg))
    var size = Int(seg_sizes.unsafe_load(seg))
    var tid = Int(thread_idx.x)
    var start = Int(block_idx.x) * REORDER_BLOCK
    var bit = Int(bit_in)

    var v = Int32(0)
    if start + tid < size:
        v = Int32(
            (Int(keys.unsafe_load(base + start + tid)) >> bit) & 1
        )

    var exclusive = prefix_sum[block_size=REORDER_BLOCK, exclusive=True](v)

    if start + tid < size:
        offsets.unsafe_store(base + start + tid, exclusive)
    if tid == REORDER_BLOCK - 1:
        block_sums.unsafe_store(
            seg * Int(blocks_wide_in) + Int(block_idx.x), exclusive + v
        )


def seg_scan_block_sums_kernel(
    block_sums: MutPointer[Int32, MutAnyOrigin],
    seg_sizes: MutPointer[UInt32, MutAnyOrigin],
    blocks_wide_in: Int32,
):
    """`scan_block_sums_kernel` per segment: one thread, one segment.

    Their per-leaf call reaches the SAME serial scan
    (`reorder_one_bit.mojo`'s `scan_block_sums_kernel`, grid 1 block 1),
    once per leaf; this runs all of them in one launch. The scan is
    exclusive and in place, over only the blocks this segment actually
    covers -- a segment's unused tail slots are never read, so they need
    no initialisation.
    """
    var seg = Int(block_idx.x)
    var size = Int(seg_sizes.unsafe_load(seg))
    var wide = Int(blocks_wide_in)
    var used = (size + REORDER_BLOCK - 1) // REORDER_BLOCK
    var acc = Int32(0)
    for b in range(used):
        var v = block_sums.unsafe_load(seg * wide + b)
        block_sums.unsafe_store(seg * wide + b, acc)
        acc += v


def seg_add_block_carry_kernel(
    offsets: MutPointer[Int32, MutAnyOrigin],
    block_sums: MutPointer[Int32, MutAnyOrigin],
    seg_offsets: MutPointer[UInt32, MutAnyOrigin],
    seg_sizes: MutPointer[UInt32, MutAnyOrigin],
    blocks_wide_in: Int32,
):
    """`add_block_carry_kernel` with the segment axis."""
    var seg = Int(block_idx.y)
    var base = Int(seg_offsets.unsafe_load(seg))
    var size = Int(seg_sizes.unsafe_load(seg))
    var i = Int(block_idx.x) * REORDER_BLOCK + Int(thread_idx.x)
    if i >= size:
        return
    var carry = block_sums.unsafe_load(
        seg * Int(blocks_wide_in) + Int(block_idx.x)
    )
    offsets.unsafe_store(
        base + i, offsets.unsafe_load(base + i) + carry
    )


def seg_reorder_one_bit_kernel(
    temp_keys: MutPointer[UInt32, MutAnyOrigin],
    temp_values: MutPointer[UInt32, MutAnyOrigin],
    offsets: MutPointer[Int32, MutAnyOrigin],
    bit_in: Int32,
    seg_offsets: MutPointer[UInt32, MutAnyOrigin],
    seg_sizes: MutPointer[UInt32, MutAnyOrigin],
    keys: MutPointer[UInt32, MutAnyOrigin],
    values: MutPointer[UInt32, MutAnyOrigin],
):
    """`ReorderOneBitImpl<ui32, ui32>` (`reorder_one_bit_impl.cuh:127`)
    with the segment axis, and with `base` restored.

    `radix_sort.mojo`'s copy of this kernel dropped `base` because their
    `ReorderBins` sorts a whole buffer; their OTHER caller
    (`SortByFlagsInLeaf`, `split_points.cu:692`) sorts a slice of one and
    keeps it. This is that second shape, batched.

    THE STABILITY THAT MAKES IT A SORT: every zero-bit row keeps its
    relative order at the front of the segment and every one-bit row keeps
    its relative order behind them, so an LSD loop over ascending bits
    leaves the segment fully ordered. `radix_sort.mojo`'s header carries
    the argument at length.
    """
    var seg = Int(block_idx.y)
    var base = Int(seg_offsets.unsafe_load(seg))
    var size = Int(seg_sizes.unsafe_load(seg))
    if size <= 0:
        return
    var bit = Int(bit_in)

    # `totalOnes = offsets[size - 1] + ((tempKeys[size - 1] >> bit) & 1)`
    var last_flag = Int32(
        (Int(temp_keys.unsafe_load(base + size - 1)) >> bit) & 1
    )
    var total_ones = offsets.unsafe_load(base + size - 1) + last_flag
    var total_zeros = Int32(size) - total_ones

    var i = Int(block_idx.x) * REORDER_BLOCK + Int(thread_idx.x)
    if i >= size:
        return
    var ones_before = offsets.unsafe_load(base + i)
    var key = temp_keys.unsafe_load(base + i)
    var val = temp_values.unsafe_load(base + i)
    var zeroes_before = Int32(i) - ones_before

    var is_zero = ((Int(key) >> bit) & 1) == 0
    var dst = zeroes_before if is_zero else (total_zeros + ones_before)
    keys.unsafe_store(base + Int(dst), key)
    values.unsafe_store(base + Int(dst), val)


def launch_segmented_radix_sort(
    ctx: DeviceContext,
    size: Int,
    n_segments: Int,
    max_segment_size: Int,
    first_bit: Int,
    last_bit: Int,
    mut keys: DeviceBuffer[DType.uint32],
    mut values: DeviceBuffer[DType.uint32],
    mut temp_keys: DeviceBuffer[DType.uint32],
    mut temp_values: DeviceBuffer[DType.uint32],
    mut seg_offsets: DeviceBuffer[DType.uint32],
    mut seg_sizes: DeviceBuffer[DType.uint32],
    mut offsets: DeviceBuffer[DType.int32],
    mut block_sums: DeviceBuffer[DType.int32],
) raises:
    """`SegmentedRadixSort(keys, values, tmpKeys, tmpValues, offsets,
    partCount, firstBit, lastBit)` (`segmented_sort.h:5-16`).

    `keys` and `values` hold the answer on return, whichever side of the
    ping-pong the last pass wrote -- the same contract
    `launch_radix_sort_bins` has, and the reason both take their temporary
    buffers rather than allocating them.

    `block_sums` must hold `n_segments * blocks_wide` Int32.
    """
    if size <= 0 or n_segments <= 0:
        return
    if last_bit <= first_bit:
        return
    if last_bit > 32 or first_bit < 0:
        raise Error(
            String("segmented radix sort bit range out of a ui32: [")
            + String(first_bit) + ", " + String(last_bit) + ")"
        )

    var blocks_wide = (
        max_segment_size + REORDER_BLOCK - 1
    ) // REORDER_BLOCK
    if blocks_wide < 1:
        blocks_wide = 1

    var bit = first_bit
    var parity = 0
    while bit < last_bit:
        # the ping-pong: even passes read `keys`, odd passes read `temp`
        if parity == 0:
            _seg_radix_pass(
                ctx, bit, blocks_wide, n_segments,
                keys, values, temp_keys, temp_values,
                seg_offsets, seg_sizes, offsets, block_sums,
            )
        else:
            _seg_radix_pass(
                ctx, bit, blocks_wide, n_segments,
                temp_keys, temp_values, keys, values,
                seg_offsets, seg_sizes, offsets, block_sums,
            )
        parity = 1 - parity
        bit += 1

    if parity == 1:
        # an ODD number of passes left the answer in the temporaries;
        # copy it back so the contract above holds either way
        ctx.enqueue_copy(dst_buf=keys, src_buf=temp_keys)
        ctx.enqueue_copy(dst_buf=values, src_buf=temp_values)


def _seg_radix_pass(
    ctx: DeviceContext,
    bit: Int,
    blocks_wide: Int,
    n_segments: Int,
    mut src_keys: DeviceBuffer[DType.uint32],
    mut src_values: DeviceBuffer[DType.uint32],
    mut dst_keys: DeviceBuffer[DType.uint32],
    mut dst_values: DeviceBuffer[DType.uint32],
    mut seg_offsets: DeviceBuffer[DType.uint32],
    mut seg_sizes: DeviceBuffer[DType.uint32],
    mut offsets: DeviceBuffer[DType.int32],
    mut block_sums: DeviceBuffer[DType.int32],
) raises:
    """One pass of `ReorderOneBit`, over every segment at once."""
    ctx.enqueue_function[seg_scan_key_bit_kernel](
        src_keys.unsafe_ptr(), Int32(bit),
        seg_offsets.unsafe_ptr(), seg_sizes.unsafe_ptr(),
        Int32(blocks_wide),
        offsets.unsafe_ptr(), block_sums.unsafe_ptr(),
        grid_dim=(blocks_wide, n_segments, 1),
        block_dim=(REORDER_BLOCK, 1, 1),
    )
    ctx.enqueue_function[seg_scan_block_sums_kernel](
        block_sums.unsafe_ptr(), seg_sizes.unsafe_ptr(),
        Int32(blocks_wide),
        grid_dim=(n_segments, 1, 1), block_dim=(1, 1, 1),
    )
    ctx.enqueue_function[seg_add_block_carry_kernel](
        offsets.unsafe_ptr(), block_sums.unsafe_ptr(),
        seg_offsets.unsafe_ptr(), seg_sizes.unsafe_ptr(),
        Int32(blocks_wide),
        grid_dim=(blocks_wide, n_segments, 1),
        block_dim=(REORDER_BLOCK, 1, 1),
    )
    ctx.enqueue_function[seg_reorder_one_bit_kernel](
        src_keys.unsafe_ptr(), src_values.unsafe_ptr(),
        offsets.unsafe_ptr(), Int32(bit),
        seg_offsets.unsafe_ptr(), seg_sizes.unsafe_ptr(),
        dst_keys.unsafe_ptr(), dst_values.unsafe_ptr(),
        grid_dim=(blocks_wide, n_segments, 1),
        block_dim=(REORDER_BLOCK, 1, 1),
    )
