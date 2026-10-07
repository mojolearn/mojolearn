"""IDENTICAL batched IVF-Flat list scan (Apple; lane/apple-identical-neural,
2026-09-26).

The pinned search serves one query at a time from the host (gather the
probed lists into a candidate row, upload, `pinned_distance_tile_kernel`,
the identical radix selector, download): at 1M rows that is minutes. This
kernel does the same scoring and selection for every query in one launch,
one SIMD group per query, and returns the same k (distance, id) pairs in
the same order:

  * each candidate's squared distance is `pinned_distance_tile_kernel`'s
    arithmetic term for term: the dot `acc = ftz(identical_mul_add(ftz(q_f),
    ftz(y_f), acc))` over f ascending from +0.0, then
    `ftz(identical_mul_add(-2, acc, ftz(ftz(q_norm) + ftz(y_norm))))`,
    clamped to +0.0 when <= 0 (the norms are the ones the pinned path
    computes, `compute_row_norms`);
  * the pinned selector keys on `(twiddle_in(distance) << 32) | position in
    the candidate row`, and that row is ordered by original index
    (DEVIATION 1786), so the key is `(distance bits, original index)`; this
    kernel keeps each lane's ascending top-KM by that same key and merges
    the 32 lane lists, so the selected set and its order do not depend on
    which lane saw a candidate or in what probe order.
"""
from core.classical_distance import direct_distance_step
from experiments.classical_identical_ideas.graph_controls import IVF_DIRECT_DISTANCE


from std.gpu import WARP_SIZE, block_dim, block_idx, thread_idx, lane_id
from std.gpu.primitives.warp import shuffle_xor
from std.memory import bitcast, stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from checks.numerics import ftz, identical_mul_add

comptime IIVF_QPB = 4
comptime IIVF_MAX_DIM = 256


@always_inline
def _key(d: Float32) -> UInt32:
    """`twiddle_in(d, select_min=True)`: unsigned order == float order."""
    var bits = bitcast[DType.uint32](d)
    if (bits & UInt32(0x80000000)) != 0:
        return bits ^ UInt32(0xFFFFFFFF)
    return bits ^ UInt32(0x80000000)


@always_inline
def _kless(a: UInt32, ai: UInt32, b: UInt32, bi: UInt32) -> Bool:
    return a < b or (a == b and ai < bi)


def identical_ivf_scan_kernel[KM: Int](
    queries: MutPointer[Float32, MutAnyOrigin],
    q_norm: MutPointer[Float32, MutAnyOrigin],
    list_data: MutPointer[Float32, MutAnyOrigin],
    list_norm: MutPointer[Float32, MutAnyOrigin],
    list_offsets: MutPointer[Int32, MutAnyOrigin],
    list_indices: MutPointer[UInt32, MutAnyOrigin],
    probe_idx: MutPointer[UInt32, MutAnyOrigin],
    out_dist: MutPointer[Float32, MutAnyOrigin],
    out_idx: MutPointer[UInt32, MutAnyOrigin],
    n_queries: Int32,
    dim_in: Int32,
    n_probes_in: Int32,
    k_in: Int32,
):
    var warp = Int(thread_idx.x) // WARP_SIZE
    var lane = Int(lane_id())
    var q = Int(block_idx.x) * IIVF_QPB + warp
    var dim = Int(dim_in)
    var s_q = stack_allocation[IIVF_QPB * IIVF_MAX_DIM, Float32, address_space=AddressSpace.SHARED]()
    var qv = s_q + warp * IIVF_MAX_DIM
    # lane/no-dim-idn: the query is staged in shared memory when it fits
    # IIVF_MAX_DIM, else read from global memory in the same order.
    var staged_dim = dim if dim <= IIVF_MAX_DIM else 0
    var active = q < Int(n_queries)
    if active:
        var j = lane
        while j < staged_dim:
            qv[j] = ftz(queries[q * dim + j])
            j += WARP_SIZE
    barrier()
    if not active:
        return
    var qn = ftz(q_norm[q])
    var tk = InlineArray[UInt32, KM](fill=UInt32.MAX)
    var td = InlineArray[Float32, KM](fill=Float32(0))
    var ti = InlineArray[UInt32, KM](fill=UInt32.MAX)
    var n_probes = Int(n_probes_in)
    for p in range(n_probes):
        var l = Int(probe_idx[q * n_probes + p])
        var s = Int(list_offsets[l])
        var e = Int(list_offsets[l + 1])
        var pos = s + lane
        while pos < e:
            var row = list_data + pos * dim
            var acc = Float32(0.0)
            for f in range(dim):
                acc = direct_distance_step[1](acc,(qv[f] if staged_dim > 0 else ftz(queries[q * dim + f])),ftz(row[f])) if IVF_DIRECT_DISTANCE else ftz(identical_mul_add((qv[f] if staged_dim > 0 else ftz(queries[q * dim + f])), ftz(row[f]), acc))
            var d = acc if IVF_DIRECT_DISTANCE else ftz(identical_mul_add(Float32(-2.0), acc, ftz(qn + ftz(list_norm[pos]))))
            if d <= Float32(0.0):
                d = Float32(0.0)
            var key = _key(d)
            var id = list_indices[pos]
            if _kless(key, id, tk[KM - 1], ti[KM - 1]):
                var ck = key
                var cd = d
                var ci = id
                comptime for j in range(KM):
                    if _kless(ck, ci, tk[j], ti[j]):
                        var sk = tk[j]
                        var sd = td[j]
                        var si = ti[j]
                        tk[j] = ck
                        td[j] = cd
                        ti[j] = ci
                        ck = sk
                        cd = sd
                        ci = si
            pos += WARP_SIZE
    var k = Int(k_in)
    for r in range(k):
        var bk = tk[0]
        var bd = td[0]
        var bi = ti[0]
        # Every lane of the warp, 64 on AMD (lane/neural-net-experiment).
        comptime for sh in [32, 16, 8, 4, 2, 1]:
            comptime if sh < WARP_SIZE:
                var ok = shuffle_xor(bk, UInt32(sh))
                var od = shuffle_xor(bd, UInt32(sh))
                var oi = shuffle_xor(bi, UInt32(sh))
                if _kless(ok, oi, bk, bi):
                    bk = ok
                    bd = od
                    bi = oi
        if tk[0] == bk and ti[0] == bi:
            comptime for j in range(KM - 1):
                tk[j] = tk[j + 1]
                td[j] = td[j + 1]
                ti[j] = ti[j + 1]
            tk[KM - 1] = UInt32.MAX
            ti[KM - 1] = UInt32.MAX
        if lane == 0:
            out_dist[q * k + r] = bd
            out_idx[q * k + r] = bi


#: The staged scan's dim chunk and its padded row stride (lane neural-pass35)
comptime IIVF_CH = 32
comptime IIVF_CHP = IIVF_CH + 1


def identical_ivf_scan_staged_kernel[KM: Int](
    queries: MutPointer[Float32, MutAnyOrigin],
    q_norm: MutPointer[Float32, MutAnyOrigin],
    list_data: MutPointer[Float32, MutAnyOrigin],
    list_norm: MutPointer[Float32, MutAnyOrigin],
    list_offsets: MutPointer[Int32, MutAnyOrigin],
    list_indices: MutPointer[UInt32, MutAnyOrigin],
    probe_idx: MutPointer[UInt32, MutAnyOrigin],
    out_dist: MutPointer[Float32, MutAnyOrigin],
    out_idx: MutPointer[UInt32, MutAnyOrigin],
    n_queries: Int32,
    dim_in: Int32,
    n_probes_in: Int32,
    k_in: Int32,
):
    """`identical_ivf_scan_kernel` with the candidate rows staged through
    threadgroup memory (lane neural-pass35, 2026-10-01). Lane l still scores
    candidates pos = s + l, s + l + WARP_SIZE, ... of each probed list in
    that order with the same chain over f ascending from +0.0, the same
    distance epilogue, the same per-lane top-KM insertion and the same lane
    merge, so the bits are that kernel's. What changes is where the row's
    words are read from: the warp loads its WARP_SIZE candidate rows a
    chunk of IIVF_CH dims at a time with consecutive lanes reading
    consecutive words of one row (one coalesced transaction per row chunk,
    where the plain kernel's lanes each walked their own row, WARP_SIZE
    scattered rows per load), into a tile with a padded stride (no bank
    conflicts on the lane's read-back). The block's warps walk the same
    number of steps (the block's longest candidate walk; a warp past its own
    end only keeps the barriers), so every barrier is block-uniform."""
    var warp = Int(thread_idx.x) // WARP_SIZE
    var lane = Int(lane_id())
    var q = Int(block_idx.x) * IIVF_QPB + warp
    var dim = Int(dim_in)
    var n_probes = Int(n_probes_in)
    var s_q = stack_allocation[IIVF_QPB * IIVF_MAX_DIM, Float32, address_space=AddressSpace.SHARED]()
    var s_tile = stack_allocation[IIVF_QPB * WARP_SIZE * IIVF_CHP, Float32, address_space=AddressSpace.SHARED]()
    var s_steps = stack_allocation[IIVF_QPB, Int32, address_space=AddressSpace.SHARED]()
    var qv = s_q + warp * IIVF_MAX_DIM
    # lane/no-dim-idn: the query is staged in shared memory when it fits
    # IIVF_MAX_DIM, else read from global memory in the same order.
    var staged_dim = dim if dim <= IIVF_MAX_DIM else 0
    var tile = s_tile + warp * WARP_SIZE * IIVF_CHP
    var active = q < Int(n_queries)
    var my_steps = 0
    if active:
        var j = lane
        while j < staged_dim:
            qv[j] = ftz(queries[q * dim + j])
            j += WARP_SIZE
        # this query's steps: ceil(list size / WARP_SIZE) summed over its probes
        for p in range(n_probes):
            var l = Int(probe_idx[q * n_probes + p])
            var sz = Int(list_offsets[l + 1]) - Int(list_offsets[l])
            my_steps += (sz + WARP_SIZE - 1) // WARP_SIZE
    if lane == 0:
        s_steps[warp] = Int32(my_steps)
    barrier()
    var block_steps = 0
    for w in range(IIVF_QPB):
        var v = Int(s_steps[w])
        if v > block_steps:
            block_steps = v
    var qn = Float32(0.0)
    if active:
        qn = ftz(q_norm[q])
    var tk = InlineArray[UInt32, KM](fill=UInt32.MAX)
    var td = InlineArray[Float32, KM](fill=Float32(0))
    var ti = InlineArray[UInt32, KM](fill=UInt32.MAX)
    # the warp's cursor over (probe, position)
    var p = 0
    var s = 0
    var e = 0
    var base = 0
    if active and n_probes > 0:
        var l = Int(probe_idx[q * n_probes])
        s = Int(list_offsets[l])
        e = Int(list_offsets[l + 1])
        base = s
    var n_chunks = (dim + IIVF_CH - 1) // IIVF_CH
    for _step in range(block_steps):
        # advance past exhausted probes (a warp past its last probe idles)
        var have = active and p < n_probes
        while have and base >= e:
            p += 1
            if p < n_probes:
                var l2 = Int(probe_idx[q * n_probes + p])
                s = Int(list_offsets[l2])
                e = Int(list_offsets[l2 + 1])
                base = s
            else:
                have = False
        var pos = base + lane
        var mine = have and pos < e
        var acc = Float32(0.0)
        for c in range(n_chunks):
            var c0 = c * IIVF_CH
            var cnt = min(IIVF_CH, dim - c0)
            # the tile: WARP_SIZE rows x cnt words, consecutive lanes on
            # consecutive words of one row
            if have:
                var idx = lane
                while idx < WARP_SIZE * IIVF_CH:
                    var r = idx // IIVF_CH
                    var f = idx - r * IIVF_CH
                    var rp = base + r
                    if f < cnt and rp < e:
                        tile[r * IIVF_CHP + f] = list_data[rp * dim + c0 + f]
                    idx += WARP_SIZE
            barrier()
            if mine:
                var trow = tile + lane * IIVF_CHP
                for f in range(cnt):
                    acc = direct_distance_step[1](acc,(qv[c0 + f] if staged_dim > 0 else ftz(queries[q * dim + c0 + f])),ftz(trow[f])) if IVF_DIRECT_DISTANCE else ftz(identical_mul_add((qv[c0 + f] if staged_dim > 0 else ftz(queries[q * dim + c0 + f])), ftz(trow[f]), acc))
            barrier()
        if mine:
            var d = acc if IVF_DIRECT_DISTANCE else ftz(identical_mul_add(Float32(-2.0), acc, ftz(qn + ftz(list_norm[pos]))))
            if d <= Float32(0.0):
                d = Float32(0.0)
            var key = _key(d)
            var id = list_indices[pos]
            if _kless(key, id, tk[KM - 1], ti[KM - 1]):
                var ck = key
                var cd = d
                var ci = id
                comptime for j in range(KM):
                    if _kless(ck, ci, tk[j], ti[j]):
                        var sk = tk[j]
                        var sd = td[j]
                        var si = ti[j]
                        tk[j] = ck
                        td[j] = cd
                        ti[j] = ci
                        ck = sk
                        cd = sd
                        ci = si
        if have:
            base += WARP_SIZE
    if not active:
        return
    var k = Int(k_in)
    for r in range(k):
        var bk = tk[0]
        var bd = td[0]
        var bi = ti[0]
        comptime for sh in [32, 16, 8, 4, 2, 1]:
            comptime if sh < WARP_SIZE:
                var ok = shuffle_xor(bk, UInt32(sh))
                var od = shuffle_xor(bd, UInt32(sh))
                var oi = shuffle_xor(bi, UInt32(sh))
                if _kless(ok, oi, bk, bi):
                    bk = ok
                    bd = od
                    bi = oi
        if tk[0] == bk and ti[0] == bi:
            comptime for j in range(KM - 1):
                tk[j] = tk[j + 1]
                td[j] = td[j + 1]
                ti[j] = ti[j + 1]
            tk[KM - 1] = UInt32.MAX
            ti[KM - 1] = UInt32.MAX
        if lane == 0:
            out_dist[q * k + r] = bd
            out_idx[q * k + r] = bi


#: The list-grouped scan (lane neural-pass42, 2026-10-01): a block holds ONE
#: probed list and GQPB queries that probe it, so the list's rows are read
#: from device memory once per block instead of once per query (at the
#: board's shape every list is probed by ~125 queries: 44 GB of row reads
#: per search, which is where the L40S's 646 ms goes). Each warp is one
#: query: lane l scores the list's candidates pos = s + l, + WARP_SIZE, ...
#: in that order with the same chain over the dims from +0.0, the same
#: distance epilogue, the same per-lane top-KM insertion and the same lane
#: merge as `identical_ivf_scan_kernel`, but over this list alone; the
#: result, the list's k best by the key (distance bits, original index), is
#: written as that (query, probe)'s partial. `identical_ivf_merge_kernel`
#: then folds a query's probes' partials by the same key into the top-k.
#: The top-k by that key is a SET function of the candidates (the scan's
#: own docstring: lane and probe order do not matter), every candidate's
#: distance is its own chain, and a list's k best contain every member of
#: the global top-k that the list holds, so the bits are the scan's.
comptime GQPB = 8


def identical_ivf_scan_grouped_kernel[KM: Int](
    queries: MutPointer[Float32, MutAnyOrigin],
    q_norm: MutPointer[Float32, MutAnyOrigin],
    list_data: MutPointer[Float32, MutAnyOrigin],
    list_norm: MutPointer[Float32, MutAnyOrigin],
    list_offsets: MutPointer[Int32, MutAnyOrigin],
    list_indices: MutPointer[UInt32, MutAnyOrigin],
    grp_off: MutPointer[Int32, MutAnyOrigin],
    grp_q: MutPointer[Int32, MutAnyOrigin],
    grp_p: MutPointer[Int32, MutAnyOrigin],
    blk_list: MutPointer[Int32, MutAnyOrigin],
    blk_start: MutPointer[Int32, MutAnyOrigin],
    part_dist: MutPointer[Float32, MutAnyOrigin],
    part_idx: MutPointer[UInt32, MutAnyOrigin],
    keep: MutPointer[Int32, MutAnyOrigin],
    keep_len: Int32,
    dim_in: Int32,
    n_probes_in: Int32,
    k_in: Int32,
):
    """`keep` (lane ivf-filter-fix): the sample filter, one int32 per
    original row, 0 removing the row; `keep_len` 0 is no filter. A removed
    candidate is never scored, selected or counted (`filter_candidate_slots`
    drops it before the per-query path scores it), so a filtered search is
    this kernel over the kept set, whose top-k by the key is the oracle's
    over the index with the removed rows deleted, and an all-ones filter is
    the unfiltered search bit for bit."""
    var warp = Int(thread_idx.x) // WARP_SIZE
    var lane = Int(lane_id())
    var tid = Int(thread_idx.x)
    var b = Int(block_idx.x)
    var l = Int(blk_list[b])
    var j = Int(blk_start[b]) + warp
    var active = j < Int(grp_off[l + 1])
    var dim = Int(dim_in)
    var n_probes = Int(n_probes_in)
    var k = Int(k_in)
    var q = 0
    var p = 0
    if active:
        q = Int(grp_q[j])
        p = Int(grp_p[j])
    var s_q = stack_allocation[GQPB * IIVF_MAX_DIM, Float32, address_space=AddressSpace.SHARED]()
    var tile = stack_allocation[WARP_SIZE * IIVF_CHP, Float32, address_space=AddressSpace.SHARED]()
    var qv = s_q + warp * IIVF_MAX_DIM
    # lane/no-dim-idn: the query is staged in shared memory when it fits
    # IIVF_MAX_DIM, else read from global memory in the same order.
    var staged_dim = dim if dim <= IIVF_MAX_DIM else 0
    if active:
        var f = lane
        while f < staged_dim:
            qv[f] = ftz(queries[q * dim + f])
            f += WARP_SIZE
    barrier()
    var qn = Float32(0.0)
    if active:
        qn = ftz(q_norm[q])
    var s = Int(list_offsets[l])
    var e = Int(list_offsets[l + 1])
    var tk = InlineArray[UInt32, KM](fill=UInt32.MAX)
    var td = InlineArray[Float32, KM](fill=Float32(0))
    var ti = InlineArray[UInt32, KM](fill=UInt32.MAX)
    var n_chunks = (dim + IIVF_CH - 1) // IIVF_CH
    var base = s
    while base < e:
        var pos = base + lane
        var mine = active and pos < e
        if mine and keep_len != 0 and keep[Int(list_indices[pos])] == 0:
            mine = False
        var acc = Float32(0.0)
        for c in range(n_chunks):
            var c0 = c * IIVF_CH
            var cnt = min(IIVF_CH, dim - c0)
            var idx = tid
            while idx < WARP_SIZE * IIVF_CH:
                var r = idx // IIVF_CH
                var f = idx - r * IIVF_CH
                var rp = base + r
                if f < cnt and rp < e:
                    tile[r * IIVF_CHP + f] = list_data[rp * dim + c0 + f]
                idx += GQPB * WARP_SIZE
            barrier()
            if mine:
                var trow = tile + lane * IIVF_CHP
                for f in range(cnt):
                    acc = direct_distance_step[1](acc,(qv[c0 + f] if staged_dim > 0 else ftz(queries[q * dim + c0 + f])),ftz(trow[f])) if IVF_DIRECT_DISTANCE else ftz(identical_mul_add((qv[c0 + f] if staged_dim > 0 else ftz(queries[q * dim + c0 + f])), ftz(trow[f]), acc))
            barrier()
        if mine:
            var d = acc if IVF_DIRECT_DISTANCE else ftz(identical_mul_add(Float32(-2.0), acc, ftz(qn + ftz(list_norm[pos]))))
            if d <= Float32(0.0):
                d = Float32(0.0)
            var key = _key(d)
            var id = list_indices[pos]
            if _kless(key, id, tk[KM - 1], ti[KM - 1]):
                var ck = key
                var cd = d
                var ci = id
                comptime for jj in range(KM):
                    if _kless(ck, ci, tk[jj], ti[jj]):
                        var sk = tk[jj]
                        var sd = td[jj]
                        var si = ti[jj]
                        tk[jj] = ck
                        td[jj] = cd
                        ti[jj] = ci
                        ck = sk
                        cd = sd
                        ci = si
        base += WARP_SIZE
    if not active:
        return
    var out = (q * n_probes + p) * KM
    for r in range(KM):
        var bk = tk[0]
        var bd = td[0]
        var bi = ti[0]
        comptime for sh in [32, 16, 8, 4, 2, 1]:
            comptime if sh < WARP_SIZE:
                var ok = shuffle_xor(bk, UInt32(sh))
                var od = shuffle_xor(bd, UInt32(sh))
                var oi = shuffle_xor(bi, UInt32(sh))
                if _kless(ok, oi, bk, bi):
                    bk = ok
                    bd = od
                    bi = oi
        if tk[0] == bk and ti[0] == bi:
            comptime for jj in range(KM - 1):
                tk[jj] = tk[jj + 1]
                td[jj] = td[jj + 1]
                ti[jj] = ti[jj + 1]
            tk[KM - 1] = UInt32.MAX
            ti[KM - 1] = UInt32.MAX
        if lane == 0:
            part_dist[out + r] = bd
            part_idx[out + r] = bi


def identical_ivf_merge_kernel[KM: Int](
    part_dist: MutPointer[Float32, MutAnyOrigin],
    part_idx: MutPointer[UInt32, MutAnyOrigin],
    out_dist: MutPointer[Float32, MutAnyOrigin],
    out_idx: MutPointer[UInt32, MutAnyOrigin],
    n_queries_in: Int32,
    n_probes_in: Int32,
    k_in: Int32,
):
    """One thread per query: its probes' partials (each a list's KM best by
    the key, ascending) folded into the top-k by the same key, written in
    ascending key order as the scan writes them."""
    var q = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if q >= Int(n_queries_in):
        return
    var n_probes = Int(n_probes_in)
    var k = Int(k_in)
    var tk = InlineArray[UInt32, KM](fill=UInt32.MAX)
    var td = InlineArray[Float32, KM](fill=Float32(0))
    var ti = InlineArray[UInt32, KM](fill=UInt32.MAX)
    for p in range(n_probes):
        var base = (q * n_probes + p) * KM
        for r in range(KM):
            var id = part_idx[base + r]
            if id == UInt32.MAX:
                break
            var d = part_dist[base + r]
            var key = _key(d)
            if _kless(key, id, tk[KM - 1], ti[KM - 1]):
                var ck = key
                var cd = d
                var ci = id
                comptime for jj in range(KM):
                    if _kless(ck, ci, tk[jj], ti[jj]):
                        var sk = tk[jj]
                        var sd = td[jj]
                        var si = ti[jj]
                        tk[jj] = ck
                        td[jj] = cd
                        ti[jj] = ci
                        ck = sk
                        cd = sd
                        ci = si
    for r in range(k):
        out_dist[q * k + r] = td[r]
        out_idx[q * k + r] = ti[r]
