"""Radix top-k, one block per row.

PORT OF `raft/matrix/detail/select_radix.cuh` at RAFT `9aa17e5`. Partial.
Do not improve.

**THIS IS A `ported/` FILE WHOSE UPSTREAM IS RAFT, WHICH REFINES THE RULE IN
`cluster/README.md`.** That rule said a RAFT call is not a `ported/` file
because RAFT is a general library this tree does not mirror. That is still
right for a call we merely STAND IN FOR, like `raft::linalg::norm`. It is
wrong for a file we actually READ AND TRANSLITERATE, which is what this is,
and which makes it a derivative work of RAFT with the attribution duty that
follows. The refined rule:

    a RAFT call we stand in for   ->  mojo_only/, naming the call
    a RAFT file we transliterate  ->  ported/,  with raft as its upstream

WHY THIS ONE AND NOT THE OTHER ONE
----------------------------------
RAFT ships two top-k families and only one of them can exist here.

`matrix/detail/select_warpsort.cuh` is the FAISS WarpSelect design and is
saturated with warp intrinsics: 14 occurrences of `__shfl` and `laneId()`.
Mojo 1.0 has no warp primitives at all, only `block` and `barrier()`, so it
is not expressible.

`select_radix.cuh` has **ZERO** warp intrinsics. Counted, not assumed. It
synchronizes with `__syncthreads()` and counts with CUB block collectives,
which is exactly the pair Mojo provides. The algorithm ports; only the
primitives underneath change, and that is the correct thing to substitute.

**Their own dispatch prefers the one we cannot have.** `select_k-inl.cuh:38`
routes `k > 256` to radix and `2 < k <= 256` to the warp family, so every k a
k-NN user actually asks for (10, 50, 100) goes to warpsort and radix is
RAFT's second choice across the practical range. It is our only choice, and
the bar is not WarpSelect on an NVIDIA card, because that card cannot run
here at all. The bar is `argpartition` on a CPU.

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
fixing it is an improvement on RAFT and this tree's rule is that improvements
wait for the measurement. What it costs is worth stating: an `IDENTICAL`
column cannot cover k-NN indices without an index tie-break that RAFT does
not have, and adding one is cheap. Recorded, not done.

DEVIATIONS
----------
1. `BitsPerPass = 8`, so 256 buckets and 4 passes over a 32-bit key. RAFT's
   tuned setting is 11 bits (2048 buckets, 3 passes). Eight keeps the
   histogram at 1 KB against Metal's 32 KB threadgroup budget
   (`PORTING.md 1`) and makes the block scan exactly one element per thread.
   A pass costs a full sweep of the survivors, so this trades one extra pass
   for a much smaller scan. Measure before changing it.
2. `cub::BlockScan` becomes a Hillis-Steele scan in shared memory, the same
   substitution as `PORTING.md 14`. It is a scan over COUNTS, which are
   integers, so unlike the histogram scan of `PORTING.md 8` it is exact and
   order-independent.
3. `vectorized_process` is not ported. It is a 16-byte-load optimization
   whose only effect is bandwidth, and their own comment says they avoid it
   in two of the three branches because it costs registers.
"""

from std.atomic import Atomic
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.math import sqrt
from std.memory import bitcast, stack_allocation
from max.gpu.memory import AddressSpace
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
        # DEVIATION 2. Integer counts, so exact and order-independent.
        var offset = 1
        while offset < NUM_BUCKETS:
            var addend = Int32(0)
            if tid >= offset:
                addend = hist[tid - offset]
            barrier()
            hist[tid] = hist[tid] + addend
            barrier()
            offset *= 2

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
