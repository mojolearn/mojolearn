# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Radix top-k, one block per row.

PORT OF `raft/matrix/detail/select_radix.cuh` at RAFT `9aa17e5`. Partial.
Do not improve.

**THIS IS A `gbdt/` FILE WHOSE UPSTREAM IS RAFT, WHICH REFINES THE RULE IN
`cluster/README.md`.** That rule said a RAFT call is not a `gbdt/` file
because RAFT is a general library this tree does not mirror. That is still
right for a call we merely STAND IN FOR, like `raft::linalg::norm`. It is
wrong for a file we actually READ AND TRANSLITERATE, which is what this is,
and which makes it a derivative work of RAFT with the attribution duty that
follows. The refined rule:

    a RAFT call we stand in for   ->  original/, naming the call
    a RAFT file we transliterate  ->  gbdt/,  with raft as its upstream

WHY THIS ONE FIRST, AND THE OTHER ONE IS **NOT** RULED OUT
----------------------------------------------------------
RAFT ships two top-k families. This file ported the radix one.

**CORRECTED 2026-08-19.** This section used to say that
`matrix/detail/select_warpsort.cuh`, the FAISS WarpSelect design, was **not
expressible** because "Mojo 1.0 has no warp primitives at all, only `block`
and `barrier()`". **That claim was false.** Warp primitives exist and are
under `std.gpu.primitives.warp`: `shuffle_down`, `shuffle_idx`,
`shuffle_xor`, `lane_id`, `prefix_sum`, `reduce`, `sum`, `max`, `broadcast`,
with `syncwarp` in `max.gpu.sync`. The earlier searches looked under
`std.gpu`, `max.gpu`, `std.gpu.block` and `max.gpu.block` and missed the
`primitives` level in all four. `VENDOR_LIBRARIES.md` opens by retracting the
claim; this file was one of the places it was asserted.

So the 14 occurrences of `__shfl` and `laneId()` in `select_warpsort.cuh` are
not a wall. **Warpsort is an OPEN item, not a blocked one**, and it is not a
small one: `select_k-inl.cuh:38` routes `k > 256` to radix and `2 < k <= 256`
to the warp family, so every k a k-NN user actually asks for (10, 50, 100)
goes to warpsort in RAFT's own dispatch and radix is their second choice
across the entire practical range. Nothing here has measured the two against
each other, and warpsort IS now ported (`select_warpsort.mojo`) but cannot yet be instantiated at a launch site without crashing the compiler; see UNWIRED.md.

What remains TRUE about the choice made here: `select_radix.cuh` has **ZERO**
warp intrinsics. Counted, not assumed. It synchronizes with
`__syncthreads()` and counts with CUB block collectives, which is the pair
Mojo has shipped all along, so it was the cheaper of the two to port and it
went first for that reason and no longer for the false one. The bar is not
WarpSelect on an NVIDIA card, because that card cannot run here at all. The
bar is `argpartition` on a CPU.

A FINDING ABOUT THEIR TIE HANDLING, WHICH IS THE OPPOSITE OF WHAT I EXPECTED
----------------------------------------------------------------------------
Radix select was described to me as MORE deterministic than warp select
because bit-twiddled keys plus an index tie-break give a fixed answer. **The
index tie-break does not exist.** `last_filter` places every output with
`atomicAdd`:

    bits <  kth  ->  pos      = atomicAdd(p_out_cnt, 1)
    bits == kth  ->  back_pos = atomicAdd(p_out_back_cnt, 1)
                     kept only if back_pos < num_of_kth_needed

So the multiset of returned VALUES is deterministic, and neither the returned
INDICES nor their positions are, whenever more elements tie at the k-th value
than there are slots left. Two runs on one device can return different
neighbors of equal distance.

That is a real property of the upstream and it is NOT fixed here, because
fixing it is an improvement on RAFT and improvements do not belong in
`derived/`. What it costs was worth stating and used to end: "an `IDENTICAL`
column cannot cover k-NN indices without an index tie-break that RAFT does
not have, and adding one is cheap. Recorded, not done."

**DONE 2026-08-23, BESIDE THIS FILE RATHER THAN IN IT.**
`neighbors/original/select_radix_identical.mojo` runs the same passes over
a 64-bit `(twiddle_in(distance) << 32) | index` composite key -- a total
order in which the tie class does not exist -- and rewrites every output
slot from a rank instead of an atomic arrival (DEVIATIONS 500 and 501,
IDENTITY_PATHS row 11). `tiled_brute_force_knn` dispatches to it under
`NUMERIC_IDENTICAL` and to THIS kernel, unchanged, under `NUMERIC_FAST`.
The two files are meant to be read together: this one is what RAFT does,
that one is what an identical column needs.

DEVIATIONS
----------
1. `BitsPerPass = 8`, so 256 buckets and 4 passes over a 32-bit key. RAFT's
   tuned setting is 11 bits (2048 buckets, 3 passes). Eight keeps the
   histogram at 1 KB against Metal's 32 KB threadgroup budget
   (`PORTING.md 1`) and makes the block scan exactly one element per thread.
   A pass costs a full sweep of the survivors, so this trades one extra pass
   for a much smaller scan. Measure before changing it.
2. `vectorized_process` is not ported. It is a 16-byte-load optimization
   whose only effect is bandwidth, and their own comment says they avoid it
   in two of the three branches because it costs registers.

WHAT USED TO BE DEVIATION 2 AND IS NOT A DEVIATION ANY MORE
-----------------------------------------------------------
The bucket scan was a hand-written Hillis-Steele loop in shared memory, and
it was listed here as a deviation. **It now calls
`max.gpu.primitives.block.prefix_sum[block_size=SELECT_BLOCK]`, Modular's
block collective, which is the direct counterpart of the
`cub::BlockScan<IdxT, BlockSize>` at `select_radix.cuh:341` and its
`.InclusiveSum` at `:353,:364`.** `SELECT_BLOCK == NUM_BUCKETS`, so we land
in their `items_per_thread == 1` branch: one count per thread, one call.

What that bought: the loop ran 2 barriers x 8 rounds = **16 barriers per
radix pass per row**, against one collective call. The counts are integers,
so the result is bit-for-bit the same sequence the loop produced and there is
no fidelity cost anywhere in it. The scan is now the same KIND of thing
upstream's is, so it is ordinary ported code, not a substitution to declare.
See `VENDOR_LIBRARIES.md`.

`Atomic.fetch_add` on the SHARED histogram is NOT in that category and stays.
It is theirs: `select_radix.cuh:1002,1016,1035` in
`filter_and_histogram_for_one_block` bump a shared `histogram` with
`atomicAdd` exactly as we do, and these are INTEGER atomics, which Metal has.
"""

from std.atomic import Atomic
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.math import sqrt
from std.memory import bitcast, stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.primitives.block import prefix_sum as block_prefix_sum
from max.gpu.sync import barrier


comptime BITS_PER_PASS = 8
comptime NUM_BUCKETS = 1 << BITS_PER_PASS
comptime NUM_PASSES = (32 + BITS_PER_PASS - 1) // BITS_PER_PASS
comptime SELECT_BLOCK = NUM_BUCKETS

# Counter slots, their `Counter<T, IdxT>` fields by name.
comptime CTR_K = 0
comptime CTR_LEN = 1
comptime CTR_PREVIOUS_LEN = 2
comptime CTR_OUT_CNT = 3
comptime CTR_OUT_BACK_CNT = 4
comptime CTR_FILTER_CNT = 5
comptime CTR_SLOTS = 6


def calc_start_bit(pass_id: Int) -> Int:
    """`calc_start_bit`. Most significant bits first, so passes can be skipped.
    """
    var start_bit = 32 - (pass_id + 1) * BITS_PER_PASS
    if start_bit < 0:
        start_bit = 0
    return start_bit


def calc_mask(pass_id: Int) -> UInt32:
    """`calc_mask`, which is `calc_start_bit(pass-1) - calc_start_bit(pass)`."""
    var num_bits = calc_start_bit(pass_id - 1) - calc_start_bit(pass_id)
    return (UInt32(1) << UInt32(num_bits)) - 1


def twiddle_in(value: Float32, select_min: Bool) -> UInt32:
    """`cub::Traits<float>::TwiddleIn`, which makes float bits order-comparable.

    Flip the sign bit for positives and every bit for negatives, so that the
    unsigned ordering of the result matches the float ordering of the input.
    Then invert for a max-select. Copied because a radix pass compares BITS
    and a plain reinterpret would order negatives backwards.
    """
    var bits = bitcast[DType.uint32](value)
    if (bits & UInt32(0x80000000)) != 0:
        bits = bits ^ UInt32(0xFFFFFFFF)
    else:
        bits = bits ^ UInt32(0x80000000)
    if not select_min:
        bits = bits ^ UInt32(0xFFFFFFFF)
    return bits


def radix_topk_one_block_kernel(
    in_val: MutPointer[Float32, MutAnyOrigin],
    out_val: MutPointer[Float32, MutAnyOrigin],
    out_idx: MutPointer[UInt32, MutAnyOrigin],
    buf_val: MutPointer[Float32, MutAnyOrigin],
    buf_idx: MutPointer[UInt32, MutAnyOrigin],
    len_in: Int32,
    k_in: Int32,
    buf_len_in: Int32,
    select_min_in: Int32,
):
    """`radix_topk_one_block_kernel`, one block per row of the batch.

    One block per row is what makes the counters and the histogram fit in
    shared memory and lets every synchronization be a `barrier()` instead of
    a grid-wide one. Their multi-block `radix_topk` exists for a single very
    long row and is not ported.
    """
    var length = Int(len_in)
    var k = Int(k_in)
    var buf_len = Int(buf_len_in)
    var select_min = select_min_in != 0
    var tid = Int(thread_idx.x)
    var batch = Int(block_idx.x)

    var in_base = in_val.unsafe_offset(batch * length)
    var o_val = out_val.unsafe_offset(batch * k)
    var o_idx = out_idx.unsafe_offset(batch * k)
    var b_val = buf_val.unsafe_offset(batch * 2 * buf_len)
    var b_idx = buf_idx.unsafe_offset(batch * 2 * buf_len)

    var hist = stack_allocation[
        NUM_BUCKETS,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var ctr = stack_allocation[
        CTR_SLOTS,
        Scalar[DType.int32],
        address_space = AddressSpace.SHARED,
    ]()
    var kth_bits = stack_allocation[
        1,
        Scalar[DType.uint32],
        address_space = AddressSpace.SHARED,
    ]()

    if tid == 0:
        ctr[CTR_K] = Int32(k)
        ctr[CTR_LEN] = Int32(length)
        ctr[CTR_PREVIOUS_LEN] = Int32(length)
        ctr[CTR_OUT_CNT] = Int32(0)
        ctr[CTR_OUT_BACK_CNT] = Int32(0)
        kth_bits[0] = UInt32(0)
    barrier()

    for pass_id in range(NUM_PASSES):
        # `set_buf_pointers`. Pass 0 histograms the input and writes no
        # buffer; from pass 1 the survivors ping-pong between the two halves.
        var read_from_input = pass_id <= 1
        var in_ptr = b_val
        var in_ip = b_idx
        var out_ptr = b_val
        var out_ip = b_idx
        if pass_id == 0 or pass_id == 1:
            in_ptr = in_base
            out_ptr = b_val
            out_ip = b_idx
        elif pass_id % 2 == 0:
            in_ptr = b_val
            in_ip = b_idx
            out_ptr = b_val.unsafe_offset(buf_len)
            out_ip = b_idx.unsafe_offset(buf_len)
        else:
            in_ptr = b_val.unsafe_offset(buf_len)
            in_ip = b_idx.unsafe_offset(buf_len)
            out_ptr = b_val
            out_ip = b_idx

        var current_len = Int(ctr[CTR_LEN])
        var current_k = Int(ctr[CTR_K])
        var previous_len = Int(ctr[CTR_PREVIOUS_LEN])

        # Their two overflow branches. If the survivors did not fit the
        # buffer last pass, re-read the original and re-filter; if they will
        # not fit this pass, histogram without writing.
        if previous_len > buf_len:
            in_ptr = in_base
            read_from_input = True
            previous_len = length
        var writes_buffer = current_len <= buf_len and pass_id > 0

        # --- filter_and_histogram_for_one_block --------------------------
        hist[tid] = Int32(0)
        if tid == 0:
            ctr[CTR_FILTER_CNT] = Int32(0)
        barrier()

        var start_bit = calc_start_bit(pass_id)
        var mask = calc_mask(pass_id)
        var kth = kth_bits[0]
        var prev_start_bit = calc_start_bit(pass_id - 1)

        var i = tid
        while i < previous_len:
            var value = in_ptr.unsafe_load(i)
            var tw = twiddle_in(value, select_min)
            if pass_id == 0:
                var bucket = Int((tw >> UInt32(start_bit)) & mask)
                _ = Atomic.fetch_add(hist.unsafe_offset(bucket), Int32(1))
            else:
                var previous_bits = (
                    tw >> UInt32(prev_start_bit)
                ) << UInt32(prev_start_bit)
                if previous_bits == kth:
                    if writes_buffer:
                        var pos = Int(
                            Atomic.fetch_add(
                                ctr.unsafe_offset(CTR_FILTER_CNT), Int32(1)
                            )
                        )
                        out_ptr.unsafe_store(pos, value)
                        var src = UInt32(i)
                        if not read_from_input:
                            src = in_ip.unsafe_load(i)
                        out_ip.unsafe_store(pos, src)
                    var bucket = Int((tw >> UInt32(start_bit)) & mask)
                    _ = Atomic.fetch_add(hist.unsafe_offset(bucket), Int32(1))
                elif previous_bits < kth and writes_buffer:
                    # Already known to be inside the top k. Emit it now and
                    # never look at it again.
                    var pos = Int(
                        Atomic.fetch_add(
                            ctr.unsafe_offset(CTR_OUT_CNT), Int32(1)
                        )
                    )
                    var src = UInt32(i)
                    if not read_from_input:
                        src = in_ip.unsafe_load(i)
                    if pos < k:
                        o_val.unsafe_store(pos, value)
                        o_idx.unsafe_store(pos, src)
            i += SELECT_BLOCK
        barrier()

        # --- scan: inclusive prefix over the bucket counts ---------------
        # `cub::BlockScan<IdxT, BlockSize>` (`select_radix.cuh:341`) and its
        # `.InclusiveSum` (`:353,:364`), as `max.gpu.primitives.block`'s
        # counterpart. INCLUSIVE, because `choose_bucket` below reads
        # `hist[i - 1]` and `hist[i]` as an inclusive prefix, which is what
        # their `scan()` writes back over the histogram.
        #
        # The hand-written Hillis-Steele loop this replaced is gone: it cost
        # 2 barriers x 8 rounds = 16 barriers per radix pass per row against
        # one collective call, and the counts are integers so the arithmetic
        # is identical. `SELECT_BLOCK == NUM_BUCKETS`, one count per thread,
        # so this is their `items_per_thread == 1` branch. See
        # VENDOR_LIBRARIES.md.
        var scanned = block_prefix_sum[block_size=SELECT_BLOCK](hist[tid])
        hist[tid] = scanned
        barrier()

        # --- choose_bucket -----------------------------------------------
        var prev_count = Int32(0)
        if tid > 0:
            prev_count = hist[tid - 1]
        var cur_count = hist[tid]
        if Int(prev_count) < current_k and Int(cur_count) >= current_k:
            # One and only one thread satisfies this, which is why the
            # counter needs no atomic here. Theirs, and worth not "fixing".
            ctr[CTR_K] = Int32(current_k - Int(prev_count))
            ctr[CTR_LEN] = Int32(Int(cur_count) - Int(prev_count))
            kth_bits[0] = kth_bits[0] | (UInt32(tid) << UInt32(start_bit))
        barrier()
        if tid == 0:
            ctr[CTR_PREVIOUS_LEN] = Int32(current_len)
        barrier()


        if Int(ctr[CTR_LEN]) == Int(ctr[CTR_K]) or pass_id == NUM_PASSES - 1:
            # --- last_filter ---------------------------------------------
            var lf_kth = kth_bits[0]
            var lf_start = calc_start_bit(pass_id)
            var needed = Int(ctr[CTR_K])
            var lf_len = length
            var lf_ptr = in_base
            var lf_has_idx = False
            if writes_buffer:
                lf_len = current_len
                lf_ptr = out_ptr
                lf_has_idx = True
            barrier()

            var j = tid
            while j < lf_len:
                var value = lf_ptr.unsafe_load(j)
                var bits = (
                    twiddle_in(value, select_min) >> UInt32(lf_start)
                ) << UInt32(lf_start)
                var src = UInt32(j)
                if lf_has_idx:
                    src = out_ip.unsafe_load(j)
                if bits < lf_kth:
                    var pos = Int(
                        Atomic.fetch_add(
                            ctr.unsafe_offset(CTR_OUT_CNT), Int32(1)
                        )
                    )
                    if pos < k:
                        o_val.unsafe_store(pos, value)
                        o_idx.unsafe_store(pos, src)
                elif bits == lf_kth:
                    var back_pos = Int(
                        Atomic.fetch_add(
                            ctr.unsafe_offset(CTR_OUT_BACK_CNT), Int32(1)
                        )
                    )
                    if back_pos < needed:
                        # Ties fill from the BACK, `select_radix.cuh:429`.
                        # Which tied element lands where is atomic order and
                        # is therefore not reproducible. See the module
                        # docstring.
                        var pos = k - 1 - back_pos
                        if pos >= 0 and pos < k:
                            o_val.unsafe_store(pos, value)
                            o_idx.unsafe_store(pos, src)
                j += SELECT_BLOCK
            barrier()
            break
