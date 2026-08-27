"""Stable partition by one bit: their scan-plus-reorder path.

PORT OF `catboost/cuda/cuda_util/kernel/reorder_one_bit_impl.cuh:127`
(`ReorderOneBitImpl`) and the `SortWithoutCub` driver that calls it,
`catboost/cuda/methods/greedy_subsets_searcher/kernel/split_points.cu:692`.

## Why this file exists, and why it is not `split_points`

`SortByFlagsInLeaf` (`split_points.cu:741`) picks between TWO partition paths
by leaf size:

    if (part.Size > FastSortSize()) {          // FastSortSize() == 500000
        cub::DeviceRadixSort::SortPairs(..., 0, 1, stream);
    } else {
        SortWithoutCub(leafId, ...);
    }

**Below 500,000 rows the CUB sort is never used.** After the first split every
leaf of an 800k dataset is under that, so the path CatBoost actually runs
almost everywhere is `SortWithoutCub`, and that is ORDINARY PORTABLE CUDA
living in `cuda_util/kernel/`, not a vendor call. Reading only the CUB call in
`split_points.cu` hid that for the whole port.

`SortWithoutCub` is two steps:

1. `cub::DeviceScan::ExclusiveSum` over the flag bits, per leaf
2. `ReorderOneBitImpl<bool, ui32, N=1, blockSize=512>`

Step 2 is transliterated below. Step 1 has no CatBoost implementation to port
-- it IS the vendor call -- and `nn.cumsum` ships CPU-only (re-checked against
the docs 2026-08-19: one overload, no `ctx`, no `target`), so the DEVICE-WIDE
scan is written here as the standard two-level decoupled scan. That is what
CUB does internally and it is recorded as a gap rather than presented as a
choice. See `VENDOR_LIBS.md` sections 3b and 3c.

The BLOCK-level half of it is not hand-written any more:
`max.gpu.primitives.block.prefix_sum[exclusive=True]` is
`cub::BlockScan::ExclusiveSum`'s counterpart and is what pass 1 calls. What
remains ours is the three-pass decoupling around it, which is exactly the part
with no shipped counterpart.

## Their reorder arithmetic, verbatim

    totalOnes    = offsets[size-1] + ((tempKeys[size-1] >> bit) & 1)
    totalZeros   = size - totalOnes
    onesBefore   = offsets[idx]
    zeroesBefore = idx - onesBefore
    isZero       = ((key >> bit) & 1) == 0
    offset       = isZero ? zeroesBefore : (totalZeros + onesBefore)
    keys[offset] = key;  values[offset] = value

`offsets` is the EXCLUSIVE prefix count of ONES, so `zeroesBefore` is derived
by subtraction rather than scanned separately. That is the whole trick: one
scan serves both destinations.
"""

from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.primitives.block import prefix_sum

#: `const int blockSize = 512` at their call site (`split_points.cu:722`),
#: and again at `reorder_one_bit.cu:35`.
comptime REORDER_BLOCK = 512

#: `const int N = 1` at their call site (`split_points.cu:723`), and again
#: at `reorder_one_bit.cu:36`.
comptime REORDER_UNROLL = 1

#: `FastSortSize()` (`split_points.cu:737`). Above this they take the CUB
#: 1-bit radix sort instead. This was recorded as a gap while the scan path
#: ran at every size; DEVIATION 1907 closed it with the single-pass
#: decoupled-lookback partition (`reorder_single_pass.mojo`), routed to
#: FAST NVIDIA/AMD by the `reorder_single_pass_for` kernel-matrix row and
#: thresholded on THIS constant. Apple and IDENTICAL keep the scan path at
#: every size.
comptime FAST_SORT_SIZE = 500000


def block_scan_flags_kernel(
    flags: MutPointer[UInt8, MutAnyOrigin],
    offset_in: Int32,
    size_in: Int32,
    offsets: MutPointer[Int32, MutAnyOrigin],
    block_sums: MutPointer[Int32, MutAnyOrigin],
):
    """Pass 1 of `cub::DeviceScan::ExclusiveSum`: scan within a block.

    Writes the block-local exclusive prefix of ONES into `offsets`, and each
    block's total into `block_sums`. A second pass scans `block_sums` and a
    third adds the carry, which is the decoupled shape CUB uses internally.

    NO CATBOOST COUNTERPART: they call CUB here. `nn.cumsum` ships CPU-only
    (VENDOR_LIBS.md 3b), so the DEVICE-WIDE decoupling is written out. The
    within-block scan is `max.gpu.primitives.block.prefix_sum`, so what is
    hand-written here is only the part CUB would call `DeviceScan` for. If
    Modular gives `cumsum` a target, these three kernels become one call.
    """
    var size = Int(size_in)
    var base = Int(offset_in)
    var tid = Int(thread_idx.x)
    var start = Int(block_idx.x) * REORDER_BLOCK

    var v = Int32(0)
    if start + tid < size:
        v = Int32(Int(flags.unsafe_load(base + start + tid)) & 1)

    # `max.gpu.primitives.block.prefix_sum[exclusive=True]`, Modular's
    # counterpart to `cub::BlockScan::ExclusiveSum`, which is the collective
    # `cub::DeviceScan::ExclusiveSum` uses per block internally. The
    # hand-written Hillis-Steele scan and its shared-memory page that stood
    # here are gone: 16 barriers per block became one call. Same arithmetic,
    # and the shape is Modular's to tune rather than ours to guess.
    var exclusive = prefix_sum[block_size=REORDER_BLOCK, exclusive=True](v)

    if start + tid < size:
        offsets.unsafe_store(start + tid, exclusive)
    # The block total. A thread past `size` contributes `v == 0`, so the last
    # thread's exclusive prefix plus its own flag is the block's count of ones
    # whether or not the block is full.
    if tid == REORDER_BLOCK - 1:
        block_sums.unsafe_store(Int(block_idx.x), exclusive + v)


def scan_block_sums_kernel(
    block_sums: MutPointer[Int32, MutAnyOrigin], n_blocks_in: Int32
):
    """Pass 2: exclusive scan of the per-block totals, one block, serial.

    `n_blocks` is `size / 512`, so at 800k rows it is under 1600 entries and a
    single-thread scan of it is not on any critical path.
    """
    if Int(thread_idx.x) != 0 or Int(block_idx.x) != 0:
        return
    var n = Int(n_blocks_in)
    var running = Int32(0)
    for i in range(n):
        var v = block_sums.unsafe_load(i)
        block_sums.unsafe_store(i, running)
        running += v


def add_block_carry_kernel(
    offsets: MutPointer[Int32, MutAnyOrigin],
    block_sums: MutPointer[Int32, MutAnyOrigin],
    size_in: Int32,
):
    """Pass 3: add each block's carry to its elements."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(size_in):
        offsets.unsafe_store(
            i, offsets.unsafe_load(i) + block_sums.unsafe_load(Int(block_idx.x))
        )


def reorder_one_bit_kernel(
    temp_keys: MutPointer[UInt8, MutAnyOrigin],
    temp_values: MutPointer[UInt32, MutAnyOrigin],
    offsets: MutPointer[Int32, MutAnyOrigin],
    bit_in: Int32,
    keys: MutPointer[UInt8, MutAnyOrigin],
    values: MutPointer[UInt32, MutAnyOrigin],
    base_in: Int32,
    size_in: Int32,
):
    """`ReorderOneBitImpl` (`reorder_one_bit_impl.cuh:127`), transliterated.

    `base` is ours: their pointers are pre-offset to the leaf by the caller,
    and carrying the offset as an argument keeps one launch able to serve a
    leaf without pointer arithmetic at the call site.
    """
    var size = Int(size_in)
    var base = Int(base_in)
    var bit = Int(bit_in)

    # `const int totalOnes = __ldg(offsets + size - 1)
    #                      + ((tempKeys[size - 1] >> bit) & 1);`
    var last_flag = Int32(
        (Int(temp_keys.unsafe_load(base + size - 1)) >> bit) & 1
    )
    var total_ones = offsets.unsafe_load(size - 1) + last_flag
    var total_zeros = Int32(size) - total_ones

    var i = Int(block_idx.x) * REORDER_BLOCK + Int(thread_idx.x)

    @parameter
    for k in range(REORDER_UNROLL):
        var idx = i + k * REORDER_BLOCK
        if idx < size:
            var ones_before = offsets.unsafe_load(idx)
            var key = temp_keys.unsafe_load(base + idx)
            var val = temp_values.unsafe_load(base + idx)
            var zeroes_before = Int32(idx) - ones_before

            var is_zero = ((Int(key) >> bit) & 1) == 0
            var dst = zeroes_before if is_zero else (
                total_zeros + ones_before
            )
            keys.unsafe_store(base + Int(dst), key)
            values.unsafe_store(base + Int(dst), val)


def launch_reorder_one_bit(
    ctx: DeviceContext,
    offset: Int,
    size: Int,
    mut temp_flags: DeviceBuffer[DType.uint8],
    mut temp_values: DeviceBuffer[DType.uint32],
    mut offsets: DeviceBuffer[DType.int32],
    mut block_sums: DeviceBuffer[DType.int32],
    mut flags: DeviceBuffer[DType.uint8],
    mut values: DeviceBuffer[DType.uint32],
) raises:
    """`SortWithoutCub` (`split_points.cu:692`): scan, then reorder.

    One leaf, `[offset, offset + size)`. Their driver calls this per leaf.

    The empty guard is `if (part.Size)` (`split_points.cu:694`, and `:674` on
    the CUB path): a zero-row leaf produces a zero-extent grid, and it also
    makes `offsets[size - 1]` in the reorder kernel a read before the buffer.
    """
    if size <= 0:
        return
    var n_blocks = (size + REORDER_BLOCK - 1) // REORDER_BLOCK

    ctx.enqueue_function[block_scan_flags_kernel](
        temp_flags.unsafe_ptr(), Int32(offset), Int32(size),
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
    ctx.enqueue_function[reorder_one_bit_kernel](
        temp_flags.unsafe_ptr(), temp_values.unsafe_ptr(),
        offsets.unsafe_ptr(), Int32(0),
        flags.unsafe_ptr(), values.unsafe_ptr(),
        Int32(offset), Int32(size),
        grid_dim=n_blocks, block_dim=REORDER_BLOCK,
    )
