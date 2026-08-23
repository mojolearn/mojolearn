"""Radix top-k over a COMPOSITE (distance, index) key: row 11's closure.

DEVIATIONS 500 and 501 (IDENTITY_PATHS row 11). Reached only under
`NUMERIC_IDENTICAL`.

NOT A PORT, and it is deliberately not one. `neighbors/ported/matrix/detail/
select_radix.mojo` is RAFT's `select_radix.cuh` transliterated, tie handling
included, and its module docstring records what that handling costs:

    bits <  kth  ->  pos      = atomicAdd(p_out_cnt, 1)
    bits == kth  ->  back_pos = atomicAdd(p_out_back_cnt, 1)
                     kept only if back_pos < num_of_kth_needed

so when more elements tie at the k-th distance than there are slots left,
WHICH of them is returned is decided by atomic arrival order, and WHERE each
returned element lands is decided by it too. Two runs on one device can
return different neighbours; two vendors certainly can. That is a real
property of the upstream, it is why IDENTITY_PATHS row 11 was a REFUSE, and
the ported file must keep it -- fixing a thing upstream does not do is an
improvement, and improvements live outside `ported/` (PORTING_RULES).

This file is that improvement, reached only under `NUMERIC_IDENTICAL`.

WHAT CHANGES, IN TWO MOVES
---------------------------
**DEVIATION 500 -- the key is 64 bits and no two elements share one.**
The radix passes run over

    key = (twiddle_in(distance) << 32) | row_index

which is a TOTAL ORDER on (distance, index): equal distances are separated
by the index, which is unique within a row by construction. The tie class
the upstream resolves with an atomic therefore does not exist -- there is
exactly one k-th element, `num_of_kth_needed` is exactly one, and the
SELECTED SET is a pure function of the input. Eight passes of eight bits
instead of four; the extra passes walk survivors, not the row, so the cost
is not four more sweeps of `n`.

**DEVIATION 501 -- the placement is a RANK, not an arrival order.**
A deterministic set is not yet a deterministic answer: `last_filter` still
hands out slots with `atomicAdd`, so the same k elements can land in k!
different arrangements. After selection this kernel ranks the k winners
against each other by the same composite key -- one thread per slot,
counting how many of the k are smaller -- and writes each to its rank. The
output is therefore ASCENDING by (distance, index), which is also what the
fused arm's `WarpSelect::reduce()` produces, so the two arms become
comparable slot for slot instead of as multisets.

Rank-by-counting rather than a sort: k is at most `SELECT_BLOCK`, the
comparison is a total order so no two ranks collide, and it needs no
network, no shared scratch beyond the staged pairs, and no second barrier
pattern to get wrong. `ball_cover.mojo` already uses the same argument.

WHAT IS STILL REFUSED HERE
---------------------------
`k > SELECT_BLOCK`. The rank pass gives one thread to each output slot; a
larger k needs a loop, which is easy and is not written until something
asks for it. The refusal is at the launcher, not silent.

THIS SELECTOR IS NOT THE WHOLE TILED ARM. Its distances arrive from
`core/gemm.mojo::gemm_nt`, which is MAX's `linalg.matmul` -- a closed vendor
library whose k-split and tile shape are per-vendor, and therefore not
identical across GPUs no matter what the selector does. That is why the
IDENTICAL build routes the tiled arm's distances through
`neighbors/mojo_only/pinned_distance_tile.mojo` instead. Selector and
distances are two separate closures and row 11 needed both.
"""

from std.atomic import Atomic
from std.gpu import block_idx, thread_idx
from std.memory import bitcast, stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.primitives.block import prefix_sum as block_prefix_sum
from max.gpu.sync import barrier

from neighbors.ported.matrix.detail.select_radix import (
    SELECT_BLOCK,
    twiddle_in,
)


comptime BITS_PER_PASS_64 = 8
comptime NUM_BUCKETS_64 = 1 << BITS_PER_PASS_64
comptime NUM_PASSES_64 = 64 // BITS_PER_PASS_64

# `Counter<T, IdxT>`'s fields, the same slots the ported kernel uses. The
# back-fill counter is gone with the tie class it served.
comptime CTR_K = 0
comptime CTR_LEN = 1
comptime CTR_PREVIOUS_LEN = 2
comptime CTR_OUT_CNT = 3
comptime CTR_FILTER_CNT = 4
comptime CTR_SLOTS = 5


@always_inline
def composite_key(value: Float32, index: UInt32, select_min: Bool) -> UInt64:
    """The total order: distance in the high half, index in the low half.

    `twiddle_in` is the ported `cub::Traits<float>::TwiddleIn` -- the same
    function the upstream selector uses -- so the high half orders exactly
    as RAFT's 32-bit key does and this key is a REFINEMENT of theirs, never
    a different ordering. The low half breaks what the high half leaves
    equal, always toward the LOWER INDEX, which is the tie-break
    `raft::argmin_op` uses in the unfused distance path and the one the
    fused k-means kernel was already given.

    The index half is NOT twiddled: it is an unsigned count, and unsigned
    ordering is already its ordering.
    """
    return (UInt64(twiddle_in(value, select_min)) << UInt64(32)) | UInt64(
        index
    )


@always_inline
def calc_start_bit_64(pass_id: Int) -> Int:
    """Most significant bits first, so a pass can be skipped."""
    var start_bit = 64 - (pass_id + 1) * BITS_PER_PASS_64
    if start_bit < 0:
        start_bit = 0
    return start_bit


@always_inline
def calc_mask_64(pass_id: Int) -> UInt64:
    var num_bits = calc_start_bit_64(pass_id - 1) - calc_start_bit_64(pass_id)
    return (UInt64(1) << UInt64(num_bits)) - 1


def radix_topk_identical_kernel(
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
    """One block per row, eight passes over the composite key, ranked output.

    The structure is the ported kernel's: a per-pass histogram over one byte
    of the key, a block scan over the buckets, `choose_bucket`, and a final
    filter. What differs is the key width (64), the absence of the tie
    back-fill, and the rank pass at the end.

    THE BUFFER CONTRACT IS THE PORTED KERNEL'S. `buf_val` / `buf_idx` hold
    `2 * buf_len` pairs per row and the survivors ping-pong between the two
    halves. Pass 0 and pass 1 read the original row; from pass 2 the
    survivors carry their ORIGINAL indices with them, which is what makes
    the composite key computable at every pass.
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
        NUM_BUCKETS_64,
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
        Scalar[DType.uint64],
        address_space = AddressSpace.SHARED,
    ]()
    # The rank pass's staging. `SELECT_BLOCK` pairs, which is also the
    # largest k this kernel accepts; the launcher refuses anything larger.
    var s_val = stack_allocation[
        SELECT_BLOCK,
        Scalar[DType.float32],
        address_space = AddressSpace.SHARED,
    ]()
    var s_idx = stack_allocation[
        SELECT_BLOCK,
        Scalar[DType.uint32],
        address_space = AddressSpace.SHARED,
    ]()

    if tid == 0:
        ctr[CTR_K] = Int32(k)
        ctr[CTR_LEN] = Int32(length)
        ctr[CTR_PREVIOUS_LEN] = Int32(length)
        ctr[CTR_OUT_CNT] = Int32(0)
        kth_bits[0] = UInt64(0)
    barrier()

    for pass_id in range(NUM_PASSES_64):
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

        # Their two overflow branches, unchanged in meaning.
        if previous_len > buf_len:
            in_ptr = in_base
            in_ip = b_idx
            read_from_input = True
            previous_len = length
        var writes_buffer = current_len <= buf_len and pass_id > 0

        hist[tid] = Int32(0)
        if tid == 0:
            ctr[CTR_FILTER_CNT] = Int32(0)
        barrier()

        var start_bit = calc_start_bit_64(pass_id)
        var mask = calc_mask_64(pass_id)
        var kth = kth_bits[0]
        var prev_start_bit = calc_start_bit_64(pass_id - 1)

        var i = tid
        while i < previous_len:
            var value = in_ptr.unsafe_load(i)
            var src = UInt32(i)
            if not read_from_input:
                src = in_ip.unsafe_load(i)
            var key = composite_key(value, src, select_min)
            if pass_id == 0:
                var bucket = Int((key >> UInt64(start_bit)) & mask)
                _ = Atomic.fetch_add(hist.unsafe_offset(bucket), Int32(1))
            else:
                var previous_bits = (
                    key >> UInt64(prev_start_bit)
                ) << UInt64(prev_start_bit)
                if previous_bits == kth:
                    if writes_buffer:
                        var pos = Int(
                            Atomic.fetch_add(
                                ctr.unsafe_offset(CTR_FILTER_CNT), Int32(1)
                            )
                        )
                        out_ptr.unsafe_store(pos, value)
                        out_ip.unsafe_store(pos, src)
                    var bucket = Int((key >> UInt64(start_bit)) & mask)
                    _ = Atomic.fetch_add(hist.unsafe_offset(bucket), Int32(1))
                elif previous_bits < kth and writes_buffer:
                    # Already inside the top k. The SLOT it takes here is
                    # still an atomic arrival order -- and it no longer
                    # matters, because the rank pass below rewrites every
                    # slot from the key. That is the whole of DEVIATION 501.
                    var pos = Int(
                        Atomic.fetch_add(
                            ctr.unsafe_offset(CTR_OUT_CNT), Int32(1)
                        )
                    )
                    if pos < k:
                        o_val.unsafe_store(pos, value)
                        o_idx.unsafe_store(pos, src)
            i += SELECT_BLOCK
        barrier()

        var scanned = block_prefix_sum[block_size=SELECT_BLOCK](hist[tid])
        hist[tid] = scanned
        barrier()

        var prev_count = Int32(0)
        if tid > 0:
            prev_count = hist[tid - 1]
        var cur_count = hist[tid]
        if Int(prev_count) < current_k and Int(cur_count) >= current_k:
            ctr[CTR_K] = Int32(current_k - Int(prev_count))
            ctr[CTR_LEN] = Int32(Int(cur_count) - Int(prev_count))
            kth_bits[0] = kth_bits[0] | (
                UInt64(tid) << UInt64(start_bit)
            )
        barrier()
        if tid == 0:
            ctr[CTR_PREVIOUS_LEN] = Int32(current_len)
        barrier()

        if Int(ctr[CTR_LEN]) == Int(ctr[CTR_K]) or pass_id == NUM_PASSES_64 - 1:
            # --- last_filter, WITHOUT the back-fill ----------------------
            # Every key is unique, so `bits == kth` holds for exactly one
            # element and `needed` is exactly one. The `<` and `==` arms
            # can therefore share one path: take everything at or below the
            # k-th key. No `out_back_cnt`, no "kept only if", no tie class.
            var lf_kth = kth_bits[0]
            var lf_start = calc_start_bit_64(pass_id)
            var lf_len = length
            var lf_ptr = in_base
            var lf_ip = b_idx
            var lf_has_idx = False
            if writes_buffer:
                lf_len = current_len
                lf_ptr = out_ptr
                lf_ip = out_ip
                lf_has_idx = True
            barrier()

            var j = tid
            while j < lf_len:
                var value = lf_ptr.unsafe_load(j)
                var src = UInt32(j)
                if lf_has_idx:
                    src = lf_ip.unsafe_load(j)
                var bits = (
                    composite_key(value, src, select_min) >> UInt64(lf_start)
                ) << UInt64(lf_start)
                if bits <= lf_kth:
                    var pos = Int(
                        Atomic.fetch_add(
                            ctr.unsafe_offset(CTR_OUT_CNT), Int32(1)
                        )
                    )
                    if pos < k:
                        o_val.unsafe_store(pos, value)
                        o_idx.unsafe_store(pos, src)
                j += SELECT_BLOCK
            barrier()
            break

    # ---- DEVIATION 501: the rank pass -----------------------------------
    # The k winners are in `o_val` / `o_idx` in an order no one chose. Stage
    # them, rank each against the others under the same total order, and
    # write each to its rank. Distinct keys means distinct ranks, so the
    # permutation is exact and every slot is written exactly once.
    if tid < k:
        s_val[tid] = o_val.unsafe_load(tid)
        s_idx[tid] = o_idx.unsafe_load(tid)
    barrier()
    if tid < k:
        var key_t = composite_key(s_val[tid], s_idx[tid], select_min)
        var rank = 0
        for j in range(k):
            if composite_key(s_val[j], s_idx[j], select_min) < key_t:
                rank += 1
        o_val.unsafe_store(rank, s_val[tid])
        o_idx.unsafe_store(rank, s_idx[tid])
    barrier()
