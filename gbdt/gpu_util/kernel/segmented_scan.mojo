"""Segmented scan: their two CTR entry points, three-phase and decoupled.

PORT OF `catboost/cuda/cuda_util/kernel/segmented_scan.cu:22`
(`SegmentedScanCub`, reached from `cuda_util/segmented_scan.h:8`
`SegmentedScanVector`) and `cuda_util/kernel/scan.cu:47`
(`SegmentedScanAndScatterNonNegativeVector`, reached from
`cuda_util/scan.h:120`), plus the operators and output iterators they run
through, `cuda_util/kernel/segmented_scan_helpers.cuh:11-34,42,184` and
`cuda_util/kernel/index_wrapper.cuh:5`.

## Why the CTR block needs both of these

`THistoryBasedCtrCalcer` calls the scatter form three times and the plain
form once, and they are the whole of the ordered-target-statistic
arithmetic:

    ctr_calcers.h:75   SegmentedScanAndScatterNonNegativeVector(weights,  ...)
    ctr_calcers.h:93   SegmentedScanAndScatterNonNegativeVector(weights,  ...)
    ctr_calcers.h:137  SegmentedScanAndScatterNonNegativeVector(binStats, ...)
    ctr_calcers.h:182  SegmentedScanVector(Tmp, Indices, Dst, false, 1u << 31)

A segment is one (cat bin, everything decided before it) run of the sorted
order, and the running sum inside it is "the target statistic of the rows
that came BEFORE this one", which is the thing CatBoost is named after.

## Where the segment flag lives, and it is two different places

`ctr_calcers.h:182` passes `1u << 31` as the mask, so `SegmentedScanVector`
reads the flag from BIT 31 of a separate `ui32` array -- the permutation
`Indices` itself, whose low 30 bits are the row and whose top bit is the
segment start. That packing is `TIndexWrapper`
(`index_wrapper.cuh:13,17,21`): `UpdateMask` ORs `isOnBorder << 31`,
`IsSegmentStart()` is `Idx >> 31`, and `Index()` is `Idx & 0x3FFFFFFF`.
`UpdateBordersMaskImpl` (`ctrs/kernel/ctr_calcers.cu:131`) is what writes it.

The `NonNegative` family reads the flag from the SIGN BIT OF THE VALUE
instead (`TNonNegativeSegmentedSum`, `segmented_scan_helpers.cuh:11`, over
`ExtractSignBit`/`OrSignBit`, `kernel_helpers.cuh:20,31`). That is why its
inputs are built by `GatherTrivialWeights` and `WriteMask`
(`ctr_calcers.cu:10,34`), which negate the value at a segment start -- and
why the test has to be a BIT TEST and not `x < 0`: a zero weight at a
segment start is `-0.0f`, whose sign bit is set and which compares equal to
zero.

Both are supported here through one scan by a comptime `from_sign_bit`
parameter, which is the same two instantiations their two operators are.

## Their exclusive scan is an INCLUSIVE scan written one slot to the right

This is the part that is easy to get wrong by paraphrase, so it is
transcribed rather than described. Neither entry point runs an exclusive
scan. Both run `cub::DeviceScan::InclusiveScan` and then shift:

  * `SegmentedScanVector` exclusive (`segmented_scan.cu:33-45`) uses
    `TSegmentedScanOutputIterator<T, false>`, whose assignment is
    `if ((Ptr + 1) != End) Ptr[1] = val.Second` (`:206-209`) -- element `i`
    writes slot `i + 1` -- and then launches `ZeroSegmentStartsImpl`
    (`:11`) to put a 0 in every flagged slot.
  * The scatter form exclusive (`scan.cu:57-61`) fills the output with 0
    first (`FillBuffer`, `:59`), then
    `TNonNegativeSegmentedScanOutputIterator<..., false>` reads
    `Index[1]`, writes `Ptr[indexWrapper.Index()]`, and stores 0 rather
    than the sum when that next index is a segment start (`:68-73`).

**Slot 0 is never written by the shift.** In their code it is covered only
because `UpdateBordersMaskImpl` always flags `i == 0`
(`ctr_calcers.cu:139`, `i == 0 ||`), so `ZeroSegmentStarts` reaches it.
Ported as-is: `launch_segmented_scan_vector` leaves `output[0]` alone when
the caller did not flag row 0, exactly as theirs does.

## What is ours, and it is the same gap `reorder_one_bit.mojo` records

The device-wide decoupling: three kernels, block scan, scan of block
aggregates, add the carry. CUB does this internally and MAX ships no
device-wide scan (`VENDOR_LIBS.md` 3b/3c), so it is written out.

# =========================================================================
# DEVIATION BLOCK 1: the BLOCK-level scan is hand-written here, where
# `reorder_one_bit.mojo` calls `max.gpu.primitives.block.prefix_sum`.
#
# Theirs is `cub::DeviceScan::InclusiveScan` with a CUSTOM ASSOCIATIVE
# OPERATOR -- `TSegmentedSum` / `TNonNegativeSegmentedSum`, whose combine is
#
#     newFlag  = left.flag | right.flag
#     newValue = right.flag ? right.value : left.value + right.value
#
# `block.prefix_sum` takes no operator; it is addition only (docs, "GPU
# block and warp operations", the `prefix_sum` entry: inclusive or exclusive
# SUM, no op parameter). So there is no shipped counterpart to call and the
# in-block half is a Hillis-Steele over that operator in threadgroup memory,
# which is what `cub::BlockScan` is doing on their side.
#
# The obvious alternative was measured against on paper and REFUSED: a
# segmented scan can be faked from two unsegmented ones as
# `cum[i] - cum[start(i) - 1]`, which WOULD let `prefix_sum` do the work.
# It is a subtraction of two nearly equal floats. At 800k rows of unit
# weights `cum` reaches 8e5 while a segment sum is order 1, so float32
# leaves about 0.06 of absolute error on a quantity of size 1. Their
# operator accumulates FROM the segment start and has no such term. Trading
# the answer for a library call is not a port.
# =========================================================================

# =========================================================================
# DEVIATION BLOCK 2: phase 2 scans the per-block aggregates with ONE
# THREAD, mirroring `scan_block_sums_kernel` in `reorder_one_bit.mojo`
# rather than recursing. At `SEG_SCAN_BLOCK` = 768 an 800k-row array has
# 1042 aggregates; CUB decouples with a lookback instead. Recorded because
# it is a scheduling choice of ours, not of theirs.
# =========================================================================
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.memory import stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from mojo_only.kernel_matrix import TARGET_COLUMN, column_shared_limit


#: `TIndexWrapper::Index()` is `Idx & 0x3FFFFFFF` (`index_wrapper.cuh:22`).
#: THIRTY bits, not 31: bit 30 is masked off too. Theirs, copied.
comptime INDEX_MASK = 0x3FFFFFFF

#: `TIndexWrapper::UpdateMask` ORs `((ui32) isOnBorder) << 31`
#: (`index_wrapper.cuh:14`), and `ctr_calcers.h:182` passes the same
#: `1u << 31` as `SegmentedScanVector`'s `flagMask`.
comptime SEGMENT_START_MASK = 0x80000000

#: `GetScanBlockSize()` (`cuda_util/kernel/scan.cuh:7`), CatBoost's own
#: constant for the scan family. It caps the block below; the budget is what
#: bounds it.
comptime CATBOOST_SCAN_BLOCK = 768

#: Threadgroup bytes one thread of the segmented scan claims: one Float32
#: running value plus one Int32 running flag, which together are their
#: `TPair<ui32, T>`.
comptime SEG_SCAN_BYTES_PER_THREAD = 8


def seg_scan_block_size[column: Int]() -> Int:
    """The block, from the vendor's shared-memory budget, capped at theirs.

    NOT from what this laptop has. `column_shared_limit`
    (`mojo_only/kernel_matrix.mojo`) is the declared per-vendor threadgroup
    budget -- 32 KB Apple, 48 KB NVIDIA, 64 KB AMD -- and 8 bytes a thread
    puts the ceiling at 4096, 6144 and 8192. CatBoost's own 768 is below all
    three, so the cap binds on every vendor and the geometry is IDENTICAL
    across the three columns. That is the answer this file wants: no vendor
    row is needed, and none is added.
    """
    comptime limit = column_shared_limit(column) // SEG_SCAN_BYTES_PER_THREAD
    return limit if limit < CATBOOST_SCAN_BLOCK else CATBOOST_SCAN_BLOCK


comptime SEG_SCAN_BLOCK = seg_scan_block_size[TARGET_COLUMN]()

#: `ZeroSegmentStartsImpl` and every elementwise kernel in `ctr_calcers.cu`
#: launch at `const ui32 blockSize = 256` (`segmented_scan.cu:40`,
#: `ctr_calcers.cu:26`).
comptime SEG_EMIT_BLOCK = 256


def seg_block_scan_kernel[from_sign_bit: Bool](
    values: MutPointer[Float32, MutAnyOrigin],
    flags: MutPointer[UInt32, MutAnyOrigin],
    flag_mask_in: Int32,
    size_in: Int32,
    scanned: MutPointer[Float32, MutAnyOrigin],
    has_flag: MutPointer[UInt8, MutAnyOrigin],
    block_sums: MutPointer[Float32, MutAnyOrigin],
    block_flags: MutPointer[UInt8, MutAnyOrigin],
):
    """Phase 1: the segmented inclusive scan inside one block.

    `cub::BlockScan` under `TSegmentedSum` (`segmented_scan_helpers.cuh:24`)
    when `from_sign_bit` is False, and under `TNonNegativeSegmentedSum`
    (`:11`) when it is True. The two differ only in where the flag is read
    and in the `abs()` their non-negative form applies, which is done here
    at load: flag out of the sign bit, value as the magnitude. After that
    the combine is one operator, and it is theirs verbatim.

    Writes, per element, the running sum since the last segment start AT OR
    AFTER the block's first element, plus whether such a start was seen. The
    element that saw one takes no carry from earlier blocks; that is the
    reset, and phase 3 is where it lands.
    """
    var s_val = stack_allocation[
        SEG_SCAN_BLOCK,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var s_flg = stack_allocation[
        SEG_SCAN_BLOCK,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()

    var size = Int(size_in)
    var tid = Int(thread_idx.x)
    var i = Int(block_idx.x) * SEG_SCAN_BLOCK + tid

    # A thread past `size` loads (0, no flag). It cannot start a segment and
    # it adds nothing, so the last thread's accumulator is the block's
    # answer whether or not the block is full -- the same padding argument
    # `block_scan_flags_kernel` makes in `reorder_one_bit.mojo`.
    var v = Float32(0.0)
    var f = Int32(0)
    if i < size:
        var raw = values.unsafe_load(i)
        comptime if from_sign_bit:
            # `ExtractSignBit` (`kernel_helpers.cuh:20`) is
            # `(*reinterpret_cast<ui32*>(&val)) >> 31`. A BIT TEST. `-0.0f`
            # is a segment start with a zero weight and `raw < 0` would
            # miss it.
            f = Int32(1) if (raw.to_bits[DType.uint32]() >> 31) != 0 else (
                Int32(0)
            )
            # `abs(right)` / `abs(left)` in their operator (`:19`).
            v = abs(raw)
        else:
            v = raw
            # `bool flag = Flags[0] & FlagMask` (`segmented_scan_helpers.cuh
            # :362`, `TSegmentedScanInputIterator::operator*`).
            var bits = flags.unsafe_load(i) & UInt32(flag_mask_in)
            f = Int32(1) if bits != 0 else Int32(0)

    s_val.unsafe_store(tid, v)
    s_flg.unsafe_store(tid, f)
    barrier()

    # The operator, applied as a Hillis-Steele inclusive scan:
    #
    #     resultValue = rightFlag ? right : left + right
    #     newFlag     = leftFlag | rightFlag
    #
    # `right` is this thread, `left` is the neighbour `d` behind it. A
    # thread whose accumulated flag is already set has reached its segment's
    # start and stops absorbing, which is the reset. `d` is uniform across
    # the block, so every barrier is reached by every thread.
    var d = 1
    while d < SEG_SCAN_BLOCK:
        var lv = Float32(0.0)
        var lf = Int32(0)
        if tid >= d:
            lv = s_val.unsafe_load(tid - d)
            lf = s_flg.unsafe_load(tid - d)
        barrier()
        if tid >= d and s_flg.unsafe_load(tid) == Int32(0):
            s_val.unsafe_store(tid, s_val.unsafe_load(tid) + lv)
            s_flg.unsafe_store(tid, lf)
        barrier()
        d *= 2

    if i < size:
        scanned.unsafe_store(i, s_val.unsafe_load(tid))
        var seen = s_flg.unsafe_load(tid)
        has_flag.unsafe_store(i, UInt8(1) if seen != Int32(0) else UInt8(0))

    if tid == SEG_SCAN_BLOCK - 1:
        var b = Int(block_idx.x)
        block_sums.unsafe_store(b, s_val.unsafe_load(tid))
        var any = s_flg.unsafe_load(tid)
        block_flags.unsafe_store(b, UInt8(1) if any != Int32(0) else UInt8(0))


def seg_scan_block_sums_kernel(
    block_sums: MutPointer[Float32, MutAnyOrigin],
    block_flags: MutPointer[UInt8, MutAnyOrigin],
    n_blocks_in: Int32,
):
    """Phase 2: the same operator again, over the per-block aggregates.

    One thread, serial, exactly as `scan_block_sums_kernel`
    (`reorder_one_bit.mojo`) is. `block_sums[b]` leaves holding the carry
    INTO block `b`, and the carry dies at a block that contains a segment
    start -- past that start nothing earlier is in the segment any more.
    """
    if Int(thread_idx.x) != 0 or Int(block_idx.x) != 0:
        return
    var n = Int(n_blocks_in)
    var running = Float32(0.0)
    for b in range(n):
        var v = block_sums.unsafe_load(b)
        var f = block_flags.unsafe_load(b)
        block_sums.unsafe_store(b, running)
        if f != UInt8(0):
            running = v
        else:
            running = running + v


def seg_add_block_carry_kernel(
    scanned: MutPointer[Float32, MutAnyOrigin],
    has_flag: MutPointer[UInt8, MutAnyOrigin],
    block_sums: MutPointer[Float32, MutAnyOrigin],
    size_in: Int32,
):
    """Phase 3: add the block's carry, but ONLY to elements still open.

    An element that already saw a segment start inside its own block owns a
    complete sum and must not take anything from before the block. This is
    the whole difference from `add_block_carry_kernel`, which adds
    unconditionally.
    """
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(size_in):
        if has_flag.unsafe_load(i) == UInt8(0):
            scanned.unsafe_store(
                i,
                scanned.unsafe_load(i)
                + block_sums.unsafe_load(Int(block_idx.x)),
            )


def seg_shift_output_kernel(
    scanned: MutPointer[Float32, MutAnyOrigin],
    size_in: Int32,
    output: MutPointer[Float32, MutAnyOrigin],
):
    """`TSegmentedScanOutputIterator<T, false>::operator=`
    (`segmented_scan_helpers.cuh:202-211`): `if ((Ptr + 1) != End)
    Ptr[1] = val.Second`. Element `i` writes slot `i + 1`. Slot 0 is not
    written; see the header.
    """
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var size = Int(size_in)
    if i < size and (i + 1) < size:
        output.unsafe_store(i + 1, scanned.unsafe_load(i))


def seg_copy_output_kernel(
    scanned: MutPointer[Float32, MutAnyOrigin],
    size_in: Int32,
    output: MutPointer[Float32, MutAnyOrigin],
):
    """`TSegmentedScanOutputIterator<T, true>::operator=` (`:203-204`):
    `Ptr[0] = val.Second`, the inclusive arm."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(size_in):
        output.unsafe_store(i, scanned.unsafe_load(i))


def zero_segment_starts_kernel(
    flags: MutPointer[UInt32, MutAnyOrigin],
    flag_mask_in: Int32,
    size_in: Int32,
    output: MutPointer[Float32, MutAnyOrigin],
):
    """`ZeroSegmentStartsImpl` (`segmented_scan.cu:11-19`), transliterated.

    Runs AFTER the shift, and it is what makes the exclusive answer 0 at
    every segment start.
    """
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(size_in):
        var segment_start = flags.unsafe_load(i) & UInt32(flag_mask_in)
        if segment_start != UInt32(0):
            output.unsafe_store(i, Float32(0.0))


def seg_scatter_output_kernel[inclusive: Bool](
    scanned: MutPointer[Float32, MutAnyOrigin],
    indices: MutPointer[UInt32, MutAnyOrigin],
    size_in: Int32,
    output: MutPointer[Float32, MutAnyOrigin],
):
    """`TNonNegativeSegmentedScanOutputIterator::TReference::operator=`
    (`segmented_scan_helpers.cuh:62-76`), both arms.

        inclusive:  Ptr[TIndexWrapper(Index[0]).Index()] = abs(val)
        exclusive:  if ((Index + 1) != End)
                        w = TIndexWrapper(Index[1])
                        Ptr[w.Index()] = w.IsSegmentStart() ? 0 : abs(val)

    The scan runs in the SORTED order and the answer is scattered back to
    the row it belongs to, which is why this is one kernel and not a scan
    followed by a gather.
    """
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var size = Int(size_in)
    if i >= size:
        return
    comptime if inclusive:
        var w = indices.unsafe_load(i)
        output.unsafe_store(
            Int(w & UInt32(INDEX_MASK)), abs(scanned.unsafe_load(i))
        )
    else:
        if (i + 1) != size:
            var w = indices.unsafe_load(i + 1)
            var dst = Int(w & UInt32(INDEX_MASK))
            if (w & UInt32(SEGMENT_START_MASK)) != UInt32(0):
                output.unsafe_store(dst, Float32(0.0))
            else:
                output.unsafe_store(dst, abs(scanned.unsafe_load(i)))


def _run_segmented_scan[from_sign_bit: Bool](
    ctx: DeviceContext,
    size: Int,
    flag_mask: Int,
    mut values: DeviceBuffer[DType.float32],
    mut flags: DeviceBuffer[DType.uint32],
    mut scanned: DeviceBuffer[DType.float32],
    mut has_flag: DeviceBuffer[DType.uint8],
    mut block_sums: DeviceBuffer[DType.float32],
    mut block_flags: DeviceBuffer[DType.uint8],
) raises:
    """The three phases. `scanned` leaves holding the segmented INCLUSIVE
    scan of the whole array, which is what both of their entry points hand
    to their output iterator."""
    var n_blocks = (size + SEG_SCAN_BLOCK - 1) // SEG_SCAN_BLOCK

    ctx.enqueue_function[seg_block_scan_kernel[from_sign_bit]](
        values.unsafe_ptr(), flags.unsafe_ptr(), Int32(flag_mask),
        Int32(size), scanned.unsafe_ptr(), has_flag.unsafe_ptr(),
        block_sums.unsafe_ptr(), block_flags.unsafe_ptr(),
        grid_dim=n_blocks, block_dim=SEG_SCAN_BLOCK,
    )
    ctx.enqueue_function[seg_scan_block_sums_kernel](
        block_sums.unsafe_ptr(), block_flags.unsafe_ptr(), Int32(n_blocks),
        grid_dim=1, block_dim=1,
    )
    ctx.enqueue_function[seg_add_block_carry_kernel](
        scanned.unsafe_ptr(), has_flag.unsafe_ptr(), block_sums.unsafe_ptr(),
        Int32(size),
        grid_dim=n_blocks, block_dim=SEG_SCAN_BLOCK,
    )


def launch_segmented_scan_vector(
    ctx: DeviceContext,
    size: Int,
    inclusive: Bool,
    flag_mask: Int,
    mut values: DeviceBuffer[DType.float32],
    mut flags: DeviceBuffer[DType.uint32],
    mut output: DeviceBuffer[DType.float32],
    mut scanned: DeviceBuffer[DType.float32],
    mut has_flag: DeviceBuffer[DType.uint8],
    mut block_sums: DeviceBuffer[DType.float32],
    mut block_flags: DeviceBuffer[DType.uint8],
) raises:
    """`SegmentedScanVector` (`cuda_util/segmented_scan.h:8`) ->
    `SegmentedScanCub` (`segmented_scan.cu:22`).

    `ctr_calcers.h:182` is the CTR call: `inclusive=false`,
    `flagMask = 1u << 31`, flags carried in the permutation itself.

    The empty guard mirrors `if (numBlocks)` on their launches
    (`segmented_scan.cu:42`).
    """
    if size <= 0:
        return
    _run_segmented_scan[False](
        ctx, size, flag_mask, values, flags, scanned, has_flag,
        block_sums, block_flags,
    )

    var emit_blocks = (size + SEG_EMIT_BLOCK - 1) // SEG_EMIT_BLOCK
    if inclusive:
        ctx.enqueue_function[seg_copy_output_kernel](
            scanned.unsafe_ptr(), Int32(size), output.unsafe_ptr(),
            grid_dim=emit_blocks, block_dim=SEG_EMIT_BLOCK,
        )
    else:
        ctx.enqueue_function[seg_shift_output_kernel](
            scanned.unsafe_ptr(), Int32(size), output.unsafe_ptr(),
            grid_dim=emit_blocks, block_dim=SEG_EMIT_BLOCK,
        )
        ctx.enqueue_function[zero_segment_starts_kernel](
            flags.unsafe_ptr(), Int32(flag_mask), Int32(size),
            output.unsafe_ptr(),
            grid_dim=emit_blocks, block_dim=SEG_EMIT_BLOCK,
        )


def launch_segmented_scan_and_scatter_non_negative(
    ctx: DeviceContext,
    size: Int,
    inclusive: Bool,
    mut values: DeviceBuffer[DType.float32],
    mut indices: DeviceBuffer[DType.uint32],
    mut output: DeviceBuffer[DType.float32],
    mut scanned: DeviceBuffer[DType.float32],
    mut has_flag: DeviceBuffer[DType.uint8],
    mut block_sums: DeviceBuffer[DType.float32],
    mut block_flags: DeviceBuffer[DType.uint8],
) raises:
    """`SegmentedScanAndScatterNonNegativeVector` (`scan.cu:47`).

    Flag from the value's sign bit, answer scattered through `indices`.
    Called three times by `THistoryBasedCtrCalcer` (`ctr_calcers.h:75`,
    `:93`, `:137`), always with `inclusive=false`.

    `FillBuffer<T>((T*)output, 0, size, stream)` (`scan.cu:59`) is the
    exclusive arm's zero fill and it is load-bearing: the scatter writes
    `size - 1` of the `size` slots, so the row named by `indices[0]` keeps
    whatever the fill left. `output` is a `size`-length buffer, which is
    what their mapping hands the kernel too.
    """
    if size <= 0:
        return
    if not inclusive:
        ctx.enqueue_memset(output, Float32(0.0))

    _run_segmented_scan[True](
        ctx, size, SEGMENT_START_MASK, values, indices, scanned, has_flag,
        block_sums, block_flags,
    )

    var emit_blocks = (size + SEG_EMIT_BLOCK - 1) // SEG_EMIT_BLOCK
    if inclusive:
        ctx.enqueue_function[seg_scatter_output_kernel[True]](
            scanned.unsafe_ptr(), indices.unsafe_ptr(), Int32(size),
            output.unsafe_ptr(),
            grid_dim=emit_blocks, block_dim=SEG_EMIT_BLOCK,
        )
    else:
        ctx.enqueue_function[seg_scatter_output_kernel[False]](
            scanned.unsafe_ptr(), indices.unsafe_ptr(), Int32(size),
            output.unsafe_ptr(),
            grid_dim=emit_blocks, block_dim=SEG_EMIT_BLOCK,
        )
