# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`cub::DeviceSegmentedRadixSort::SortKeys`, the one call in
`quantiles.cuh` that has no shipped counterpart.

THIS DOES NOT FOLLOW A cuML FILE. It is the replacement for a CUB call
that `quantiles.cuh:244` and `:258` make, and it is filed under
`checks/` for exactly that reason: cuML's mirror address for this code
is CUB's, not theirs.

WHY IT IS HAND-WRITTEN RATHER THAN CALLED
------------------------------------------
Two facts, in this order:

  1. CUB is OPEN. `CONTRIBUTING.md` (Algorithms and references) (CUB is readable, so a question about it has an answer) and the
     lane's charter both say the vendor-substitution rule applies only
     where the incumbent calls a CLOSED library -- cuBLAS, cuSOLVER --
     because there is nothing to read. CUB is readable, so the correct
     move is to implement the kernel, not to swap in someone else's sort.
  2. MAX ships NO device sort and NO device scan (`archive/reference/VENDOR_LIBS.md`,
     re-checked 2026-08-20 by another lane and not re-measured here).
     So there is nothing to swap in even if the rule allowed it.

WHAT IS WRITTEN HERE IS NOT A FRESH DESIGN
-------------------------------------------
It is `gbdt/gpu_util/kernel/segmented_sort.mojo` with the value payload
removed. That file is this repository's implementation of CatBoost's
`NKernel::SegmentedRadixSort`, which is itself a
`cub::DeviceSegmentedRadixSort::SortPairs` call -- so the construction
below has already been through this repository's radix-sort check
(`checks/radix_sort_check.mojo`, `checks/segmented_scan_check.mojo`)
in its SortPairs form. It is DUPLICATED rather than imported because
`gbdt/` belongs to another session this round and the lane charter
forbids reaching into it. The duplication is a lane-ownership artifact
and should be collapsed at merge; see the note in the lane report.

Two things are dropped from that file, both because their call site does
not have them:

  * the VALUE payload. `quantiles.cuh` calls `SortKeys`, not
    `SortPairs`; nothing downstream of `:258` ever asks which row a
    sorted sample came from.
  * the per-segment `offsets`/`sizes` BUFFERS. Their segment bounds are
    a `thrust::make_transform_iterator` over a counting iterator
    (`quantiles.cuh:201-205`) returning `col * sample_count`, so every
    segment has exactly `sample_count` entries and the base is
    arithmetic, not a lookup. Carrying general offset buffers would be
    implementing a shape their dispatch does not take.

THE KEY TRANSFORM IS CUB'S, NOT OURS
-------------------------------------
A radix sort orders by unsigned integer value and IEEE-754 float bits do
not order that way: negatives run backwards and `-0.0` sorts below
`+0.0`. CUB does this in `NumericTraits<float>::TwiddleIn` --

    mask = (key & HIGH_BIT) ? UnsignedBits(-1) : HIGH_BIT;
    key ^= mask;

-- which is why their call site hands `float` keys to CUB and says
nothing about bit twiddling. `float_to_sortable` below is that function
and `sortable_to_float` inverts it. Their bit range is `0` to
`8 * sizeof(T)` (`quantiles.cuh:253`, `:267`), i.e. ALL THIRTY-TWO bits
of a float32 key, so no bits are dropped and the ordering is total.

`-0.0` and `+0.0` therefore land in a definite order (`-0.0` first) and
then compare EQUAL to `thrust::unique` in the kernel downstream. That is
their behaviour, it is reproduced here, and the check plants both.

PORTABILITY
------------
No warp or wavefront width appears in this file. The one block-level
primitive is `max.gpu.primitives.block.prefix_sum`, whose shared-memory
footprint at `SORT_BLOCK = 512` is 512 Int32 = 2 KB -- under the 16 KB
floor of every column declared in `checks/kernel_matrix.mojo`,
including the `spec-baseline` row, so no device query is needed to know
it fits. `SORT_BLOCK` is a fixed constant rather than a queried width
precisely so that the number of blocks, and hence the summation ORDER of
the block-sum scan, is the same on every vendor. It is a SCHEDULING knob
by `kernel_matrix.mojo`'s test only in the sense that no arithmetic
depends on it: a radix reorder moves values, it never adds them.

DEVIATION 111 covers this whole file; its text is in `quantiles.mojo`
beside the call it replaces.
"""

from core.launch_log import log_launch
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.primitives.block import prefix_sum
from std.gpu import block_dim, block_idx, thread_idx
from std.memory import bitcast, stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, NUMERIC_IDENTICAL
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator

#: Threads per block for every kernel here. See PORTABILITY above.
comptime SORT_BLOCK = 512

#: FAST on Apple (lane apple-fast-rfet-scan): the exclusive scan of each
#: segment's block totals runs as one 256-thread block per segment
#: (`seg_scan_block_sums_block_kernel`) instead of one thread per segment.
#: Int32 sums of the same values in a different association: the same
#: offsets bit for bit. `-D MOJOLEARN_SEG_SUMS_SERIAL` keeps the one-thread
#: scan (the A/B arm); IDENTICAL compiles the old path.
#: Default since the M3 A/B (b272364e4, n=2, identical output hashes): RF
#: istella 13,583 -> 10,116 ms (-25.5%), taxi neutral. GBDT's own copy
#: (gbdt/gpu_util/kernel/segmented_sort.mojo) keeps the one-thread scan: its
#: opt-in block arm MOJOLEARN_GBDT_SEG_SUMS_BLOCK was DROPPED-noise
#: (lane/apple-fast-rfet-scan @ 500168cfe).
comptime SEG_SUMS_BLOCK_SCAN = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and not is_defined["MOJOLEARN_SEG_SUMS_SERIAL"]()
)
comptime SEG_SUMS_TPB = 256

#: `8 * sizeof(T)` for `T = float`, i.e. their `end_bit`
#: (`quantiles.cuh:253`). Their `begin_bit` is `0`.
comptime FLOAT32_KEY_BITS = 32

#: K11 (IDENTICAL, lane ml-cluster-nbrs 2026-10-04): `segmented_sort_keys_f32`
#: sorts by 8-bit digits, 4 passes of 3 launches, instead of 32 one-bit passes
#: of 4 launches. A sort of keys has one answer, so the output words are the
#: same bit for bit on every vendor (and the host column, which does not run
#: this, is unchanged). Each pass is a stable counting sort: per (segment,
#: block) digit counts with no atomics, one exclusive Int32 scan per segment in
#: (digit, block) order, then a stable scatter whose in-block rank is counted
#: from a shared copy of the block's digits. The counts live in the caller's
#: `offsets` buffer, so the form runs only when R8_BINS * blocks_wide <=
#: seg_size (seg_size >= R8_BINS); smaller segments keep the one-bit passes.
#: Shared pages: SORT_BLOCK Int32 = 2 KB (under every column's 16 KB floor).
#: `-D MOJOLEARN_IDN_SEG_RADIX8_OFF` (or MOJOLEARN_IDN_ALL_OFF) restores the
#: one-bit passes. The one-bit kernels stay for ensemble/randomforest.mojo.
comptime IDN_SEG_RADIX8 = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not (
    is_defined["MOJOLEARN_IDN_SEG_RADIX8_OFF"]() or is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
comptime R8_BITS = 8
comptime R8_BINS = 256
comptime R8_PASSES = FLOAT32_KEY_BITS // R8_BITS


@always_inline
def float_to_sortable(bits: UInt32) -> UInt32:
    """`cub::NumericTraits<float>::TwiddleIn`."""
    if (bits & UInt32(0x80000000)) != UInt32(0):
        return ~bits
    return bits | UInt32(0x80000000)


@always_inline
def sortable_to_float(key: UInt32) -> UInt32:
    """`cub::NumericTraits<float>::TwiddleOut`."""
    if (key & UInt32(0x80000000)) != UInt32(0):
        return key & UInt32(0x7FFFFFFF)
    return ~key


def twiddle_in_kernel(
    src: MutPointer[Float32, MutAnyOrigin],
    size_in: Int32,
    keys: MutPointer[UInt32, MutAnyOrigin],
):
    """CUB's twiddle-in, which their call gets for free by passing floats.
    """
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(size_in):
        return
    keys.unsafe_store(
        i, float_to_sortable(bitcast[DType.uint32](src.unsafe_load(i)))
    )


def twiddle_out_kernel(
    keys: MutPointer[UInt32, MutAnyOrigin],
    size_in: Int32,
    dst: MutPointer[Float32, MutAnyOrigin],
):
    """CUB's twiddle-out, writing `d_keys_out` (`quantiles.cuh:247`)."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(size_in):
        return
    dst.unsafe_store(
        i, bitcast[DType.float32](sortable_to_float(keys.unsafe_load(i)))
    )


def seg_scan_key_bit_kernel(
    keys: MutPointer[UInt32, MutAnyOrigin],
    bit_in: Int32,
    seg_size_in: Int32,
    blocks_wide_in: Int32,
    offsets: MutPointer[Int32, MutAnyOrigin],
    block_sums: MutPointer[Int32, MutAnyOrigin],
):
    """Exclusive prefix count of ONES of bit `bit_in`, within each block.

    `block_idx.y` is the segment, i.e. the feature column. Threads past
    the segment end contribute a zero bit and store nothing, so the block
    total is right whether or not the segment fills its last block. Every
    thread reaches `prefix_sum` -- returning early from a block-wide
    primitive is how this construction breaks.
    """
    var seg = Int(block_idx.y)
    var seg_size = Int(seg_size_in)
    var base = seg * seg_size
    var tid = Int(thread_idx.x)
    var start = Int(block_idx.x) * SORT_BLOCK
    var bit = Int(bit_in)

    var v = Int32(0)
    if start + tid < seg_size:
        v = Int32((Int(keys.unsafe_load(base + start + tid)) >> bit) & 1)

    var exclusive = prefix_sum[block_size=SORT_BLOCK, exclusive=True](v)

    if start + tid < seg_size:
        offsets.unsafe_store(base + start + tid, exclusive)
    if tid == SORT_BLOCK - 1:
        block_sums.unsafe_store(
            seg * Int(blocks_wide_in) + Int(block_idx.x), exclusive + v
        )


def seg_scan_block_sums_kernel(
    block_sums: MutPointer[Int32, MutAnyOrigin],
    seg_size_in: Int32,
    blocks_wide_in: Int32,
):
    """The serial exclusive scan of one segment's block totals.

    One thread per segment. This is the decoupling step CUB does inside
    its own sort; it has no counterpart to implement because it IS the vendor
    call. Recorded as a gap, not presented as a choice.
    """
    var seg = Int(block_idx.x)
    var wide = Int(blocks_wide_in)
    var used = (Int(seg_size_in) + SORT_BLOCK - 1) // SORT_BLOCK
    var acc = Int32(0)
    for b in range(used):
        var v = block_sums.unsafe_load(seg * wide + b)
        block_sums.unsafe_store(seg * wide + b, acc)
        acc += v


def seg_scan_block_sums_block_kernel(
    block_sums: MutPointer[Int32, MutAnyOrigin],
    seg_size_in: Int32,
    blocks_wide_in: Int32,
):
    """`seg_scan_block_sums_kernel` with one SEG_SUMS_TPB-thread block per
    segment (SEG_SUMS_BLOCK_SCAN): chunks of SEG_SUMS_TPB block totals are
    scanned with the block prefix sum and carried. Exclusive and in place
    over the same `used` slots; Int32 adds, so the same values."""
    var seg = Int(block_idx.x)
    var wide = Int(blocks_wide_in)
    var used = (Int(seg_size_in) + SORT_BLOCK - 1) // SORT_BLOCK
    var tid = Int(thread_idx.x)
    var total = stack_allocation[
        1, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var carry = Int32(0)
    var c0 = 0
    while c0 < used:
        var b = c0 + tid
        var v = Int32(0)
        if b < used:
            v = block_sums.unsafe_load(seg * wide + b)
        var ex = prefix_sum[block_size=SEG_SUMS_TPB, exclusive=True](v)
        if b < used:
            block_sums.unsafe_store(seg * wide + b, carry + ex)
        if tid == SEG_SUMS_TPB - 1:
            total[0] = ex + v
        barrier()
        carry += total[0]
        barrier()
        c0 += SEG_SUMS_TPB


def enqueue_seg_scan_block_sums(
    ctx: DeviceContext,
    block_sums: MutPointer[Int32, MutAnyOrigin],
    seg_size: Int,
    blocks_wide: Int,
    n_segments: Int,
) raises:
    """Launch the per-segment exclusive scan of block totals: the block
    kernel under SEG_SUMS_BLOCK_SCAN, else the one-thread kernel."""
    comptime if SEG_SUMS_BLOCK_SCAN:
        ctx.enqueue_function[seg_scan_block_sums_block_kernel](
            block_sums,
            Int32(seg_size),
            Int32(blocks_wide),
            grid_dim=(n_segments, 1, 1),
            block_dim=(SEG_SUMS_TPB, 1, 1),
        )
    else:
        ctx.enqueue_function[seg_scan_block_sums_kernel](
            block_sums,
            Int32(seg_size),
            Int32(blocks_wide),
            grid_dim=(n_segments, 1, 1),
            block_dim=(1, 1, 1),
        )


def seg_add_block_carry_kernel(
    offsets: MutPointer[Int32, MutAnyOrigin],
    block_sums: MutPointer[Int32, MutAnyOrigin],
    seg_size_in: Int32,
    blocks_wide_in: Int32,
):
    """Turn the per-block exclusive counts into segment-wide ones."""
    var seg = Int(block_idx.y)
    var seg_size = Int(seg_size_in)
    var base = seg * seg_size
    var i = Int(block_idx.x) * SORT_BLOCK + Int(thread_idx.x)
    if i >= seg_size:
        return
    var carry = block_sums.unsafe_load(
        seg * Int(blocks_wide_in) + Int(block_idx.x)
    )
    offsets.unsafe_store(base + i, offsets.unsafe_load(base + i) + carry)


def seg_reorder_one_bit_kernel(
    temp_keys: MutPointer[UInt32, MutAnyOrigin],
    offsets: MutPointer[Int32, MutAnyOrigin],
    bit_in: Int32,
    seg_size_in: Int32,
    keys: MutPointer[UInt32, MutAnyOrigin],
):
    """One stable partition by bit `bit_in`, per segment.

        totalOnes    = offsets[size-1] + ((tempKeys[size-1] >> bit) & 1)
        totalZeros   = size - totalOnes
        onesBefore   = offsets[idx]
        zeroesBefore = idx - onesBefore
        isZero       = ((key >> bit) & 1) == 0
        offset       = isZero ? zeroesBefore : (totalZeros + onesBefore)

    THE STABILITY IS WHAT MAKES THE LSD LOOP A SORT: zero-bit entries
    keep their relative order at the front and one-bit entries keep
    theirs behind, so passing over ascending bits leaves each segment
    fully ordered. A "sorted" output that permuted ties would still be
    sorted and would still be wrong for a later pass.
    """
    var seg = Int(block_idx.y)
    var seg_size = Int(seg_size_in)
    if seg_size <= 0:
        return
    var base = seg * seg_size
    var bit = Int(bit_in)

    var last_flag = Int32(
        (Int(temp_keys.unsafe_load(base + seg_size - 1)) >> bit) & 1
    )
    var total_ones = offsets.unsafe_load(base + seg_size - 1) + last_flag
    var total_zeros = Int32(seg_size) - total_ones

    var i = Int(block_idx.x) * SORT_BLOCK + Int(thread_idx.x)
    if i >= seg_size:
        return
    var ones_before = offsets.unsafe_load(base + i)
    var key = temp_keys.unsafe_load(base + i)
    var zeroes_before = Int32(i) - ones_before

    var is_zero = ((Int(key) >> bit) & 1) == 0
    var dst = zeroes_before if is_zero else (total_zeros + ones_before)
    keys.unsafe_store(base + Int(dst), key)


def seg_r8_count_kernel(
    keys: MutPointer[UInt32, MutAnyOrigin],
    shift_in: Int32,
    seg_size_in: Int32,
    blocks_wide_in: Int32,
    cnt: MutPointer[Int32, MutAnyOrigin],
):
    """K11: cnt[seg*seg_size + digit*blocks_wide + block] = how many keys of
    this block (block_idx.x) of segment block_idx.y have that digit. Integer
    counts by a fixed scan of a shared copy of the digits: no atomics."""
    comptime assert SORT_BLOCK >= R8_BINS, "one thread per digit"
    var seg = Int(block_idx.y)
    var seg_size = Int(seg_size_in)
    var base = seg * seg_size
    var tid = Int(thread_idx.x)
    var blk = Int(block_idx.x)
    var start = blk * SORT_BLOCK
    var dg = stack_allocation[
        SORT_BLOCK, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var d = Int32(-1)
    if start + tid < seg_size:
        d = Int32((Int(keys.unsafe_load(base + start + tid)) >> Int(shift_in)) & 0xFF)
    dg[tid] = d
    barrier()
    if tid < R8_BINS:
        var c = Int32(0)
        var want = Int32(tid)
        for j in range(SORT_BLOCK):
            if dg[j] == want:
                c += 1
        cnt.unsafe_store(base + tid * Int(blocks_wide_in) + blk, c)


def seg_r8_scan_kernel(
    cnt: MutPointer[Int32, MutAnyOrigin],
    seg_size_in: Int32,
    blocks_wide_in: Int32,
):
    """K11: the exclusive scan, in place, of one segment's R8_BINS *
    blocks_wide counts in (digit, block) order: each entry becomes the
    position of that block's first key of that digit. One SORT_BLOCK-thread
    block per segment, chunks scanned by the block prefix sum and carried
    (Int32 adds: one answer)."""
    var seg = Int(block_idx.x)
    var base = seg * Int(seg_size_in)
    var used = R8_BINS * Int(blocks_wide_in)
    var tid = Int(thread_idx.x)
    var total = stack_allocation[
        1, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var carry = Int32(0)
    var c0 = 0
    while c0 < used:
        var b = c0 + tid
        var v = Int32(0)
        if b < used:
            v = cnt.unsafe_load(base + b)
        var ex = prefix_sum[block_size=SORT_BLOCK, exclusive=True](v)
        if b < used:
            cnt.unsafe_store(base + b, carry + ex)
        if tid == SORT_BLOCK - 1:
            total[0] = ex + v
        barrier()
        carry += total[0]
        barrier()
        c0 += SORT_BLOCK


def seg_r8_scatter_kernel(
    src_keys: MutPointer[UInt32, MutAnyOrigin],
    shift_in: Int32,
    seg_size_in: Int32,
    blocks_wide_in: Int32,
    cnt: MutPointer[Int32, MutAnyOrigin],
    dst_keys: MutPointer[UInt32, MutAnyOrigin],
):
    """K11: the stable scatter of one pass. A key goes to its (digit, block)
    start plus the keys of the same digit before it in its block, so equal
    digits keep their order (what makes the LSD loop a sort)."""
    var seg = Int(block_idx.y)
    var seg_size = Int(seg_size_in)
    var base = seg * seg_size
    var tid = Int(thread_idx.x)
    var blk = Int(block_idx.x)
    var i = blk * SORT_BLOCK + tid
    var dg = stack_allocation[
        SORT_BLOCK, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var key = UInt32(0)
    var d = Int32(-1)
    if i < seg_size:
        key = src_keys.unsafe_load(base + i)
        d = Int32((Int(key) >> Int(shift_in)) & 0xFF)
    dg[tid] = d
    barrier()
    if i >= seg_size:
        return
    var rank = 0
    for j in range(tid):
        if dg[j] == d:
            rank += 1
    var at = Int(cnt.unsafe_load(base + Int(d) * Int(blocks_wide_in) + blk))
    dst_keys.unsafe_store(base + at + rank, key)


def _seg_radix8_sort(
    ctx: DeviceContext,
    blocks_wide: Int,
    n_segments: Int,
    seg_size: Int,
    mut work_a: DeviceBuffer[DType.uint32],
    mut work_b: DeviceBuffer[DType.uint32],
    mut offsets: DeviceBuffer[DType.int32],
) raises:
    """K11: R8_PASSES (even) 8-bit passes, ping-pong a -> b -> a: the sorted
    keys end in work_a, as the one-bit loop leaves them."""
    comptime assert (R8_PASSES % 2) == 0, "an odd pass count would leave the answer in work_b"
    for p in range(R8_PASSES):
        var shift = p * R8_BITS
        # untracked: the ping-pong select merges both origins, which the
        # aliasing check reads as src == dst (box-run-2 compile fix).
        var src = (work_a.unsafe_ptr() if p % 2 == 0 else work_b.unsafe_ptr()).unsafe_origin_cast[MutUntrackedOrigin]()
        var dst = (work_b.unsafe_ptr() if p % 2 == 0 else work_a.unsafe_ptr()).unsafe_origin_cast[MutUntrackedOrigin]()
        log_launch("seg_sort_r8_count")
        ctx.enqueue_function[seg_r8_count_kernel](
            src,
            Int32(shift),
            Int32(seg_size),
            Int32(blocks_wide),
            offsets.unsafe_ptr(),
            grid_dim=(blocks_wide, n_segments, 1),
            block_dim=(SORT_BLOCK, 1, 1),
        )
        log_launch("seg_sort_r8_scan")
        ctx.enqueue_function[seg_r8_scan_kernel](
            offsets.unsafe_ptr(),
            Int32(seg_size),
            Int32(blocks_wide),
            grid_dim=(n_segments, 1, 1),
            block_dim=(SORT_BLOCK, 1, 1),
        )
        log_launch("seg_sort_r8_scatter")
        ctx.enqueue_function[seg_r8_scatter_kernel](
            src,
            Int32(shift),
            Int32(seg_size),
            Int32(blocks_wide),
            offsets.unsafe_ptr(),
            dst,
            grid_dim=(blocks_wide, n_segments, 1),
            block_dim=(SORT_BLOCK, 1, 1),
        )


def segmented_sort_keys_f32(
    ctx: DeviceContext,
    n_segments: Int,
    seg_size: Int,
    mut src: DeviceBuffer[DType.float32],
    mut dst: DeviceBuffer[DType.float32],
    mut work_a: DeviceBuffer[DType.uint32],
    mut work_b: DeviceBuffer[DType.uint32],
    mut offsets: DeviceBuffer[DType.int32],
    mut block_sums: DeviceBuffer[DType.int32],
) raises:
    """`cub::DeviceSegmentedRadixSort::SortKeys(d_temp, bytes, src, dst,
    n_segments * seg_size, n_segments, segment_offsets,
    segment_offsets + 1, 0, 8 * sizeof(T), stream)`
    (`quantiles.cuh:257-268`).

    `src` and `dst` are DISTINCT buffers, as theirs are
    (`sampled_columns` -> `sorted_samples`, `:246-247`); `src` is not
    written -- it is spelled `mut` only because a MAX device buffer
    hands out a mutable pointer or none at all, which is the same
    spelling-not-value note `dataset.mojo` records for its widths. The caller supplies every temporary, which is CUB's
    contract too -- `:241-255` is the sizing call and the allocation it
    feeds.

    `work_a`, `work_b` must hold `n_segments * seg_size` UInt32 each,
    `offsets` the same count of Int32, and `block_sums`
    `n_segments * ceil(seg_size / SORT_BLOCK)` Int32.

    THE PING-PONG PARITY IS NOT LEFT TO CHANCE: thirty-two passes is
    EVEN, so the answer is back in `work_a` where the twiddle-out reads
    it. `checks/radix_sort_check.mojo` records a green run on a
    sabotaged copy-back that this shape simply does not have -- there is
    no copy-back, because the pass count is a compile-time constant of
    the key width and not a caller's bit range.
    """
    if n_segments <= 0 or seg_size <= 0:
        return
    var total = n_segments * seg_size

    var blocks_wide = (seg_size + SORT_BLOCK - 1) // SORT_BLOCK
    var flat_blocks = (total + SORT_BLOCK - 1) // SORT_BLOCK

    log_launch("seg_sort_twiddle_in")
    ctx.enqueue_function[twiddle_in_kernel](
        src.unsafe_ptr(),
        Int32(total),
        work_a.unsafe_ptr(),
        grid_dim=(flat_blocks, 1, 1),
        block_dim=(SORT_BLOCK, 1, 1),
    )

    comptime assert (
        FLOAT32_KEY_BITS % 2
    ) == 0, "an odd pass count would leave the answer in work_b"

    var by_r8 = False
    comptime if IDN_SEG_RADIX8:
        if R8_BINS * blocks_wide <= seg_size:
            _seg_radix8_sort(
                ctx, blocks_wide, n_segments, seg_size, work_a, work_b, offsets
            )
            by_r8 = True

    var bit = 0
    var parity = 0
    while bit < FLOAT32_KEY_BITS and not by_r8:
        if parity == 0:
            _seg_radix_pass(
                ctx,
                bit,
                blocks_wide,
                n_segments,
                seg_size,
                work_a,
                work_b,
                offsets,
                block_sums,
            )
        else:
            _seg_radix_pass(
                ctx,
                bit,
                blocks_wide,
                n_segments,
                seg_size,
                work_b,
                work_a,
                offsets,
                block_sums,
            )
        parity = 1 - parity
        bit += 1

    log_launch("seg_sort_twiddle_out")
    ctx.enqueue_function[twiddle_out_kernel](
        work_a.unsafe_ptr(),
        Int32(total),
        dst.unsafe_ptr(),
        grid_dim=(flat_blocks, 1, 1),
        block_dim=(SORT_BLOCK, 1, 1),
    )


def _seg_radix_pass(
    ctx: DeviceContext,
    bit: Int,
    blocks_wide: Int,
    n_segments: Int,
    seg_size: Int,
    mut src_keys: DeviceBuffer[DType.uint32],
    mut dst_keys: DeviceBuffer[DType.uint32],
    mut offsets: DeviceBuffer[DType.int32],
    mut block_sums: DeviceBuffer[DType.int32],
) raises:
    """One bit, every segment at once."""
    log_launch("seg_sort_scan_key_bit")
    ctx.enqueue_function[seg_scan_key_bit_kernel](
        src_keys.unsafe_ptr(),
        Int32(bit),
        Int32(seg_size),
        Int32(blocks_wide),
        offsets.unsafe_ptr(),
        block_sums.unsafe_ptr(),
        grid_dim=(blocks_wide, n_segments, 1),
        block_dim=(SORT_BLOCK, 1, 1),
    )
    log_launch("seg_sort_scan_block_sums")
    enqueue_seg_scan_block_sums(
        ctx, block_sums.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
        seg_size, blocks_wide, n_segments,
    )
    log_launch("seg_sort_add_block_carry")
    ctx.enqueue_function[seg_add_block_carry_kernel](
        offsets.unsafe_ptr(),
        block_sums.unsafe_ptr(),
        Int32(seg_size),
        Int32(blocks_wide),
        grid_dim=(blocks_wide, n_segments, 1),
        block_dim=(SORT_BLOCK, 1, 1),
    )
    log_launch("seg_sort_reorder_one_bit")
    ctx.enqueue_function[seg_reorder_one_bit_kernel](
        src_keys.unsafe_ptr(),
        offsets.unsafe_ptr(),
        Int32(bit),
        Int32(seg_size),
        dst_keys.unsafe_ptr(),
        grid_dim=(blocks_wide, n_segments, 1),
        block_dim=(SORT_BLOCK, 1, 1),
    )
