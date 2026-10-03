# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The IVF per-query search path on the device, every query at once (lane
cgr5-owed, 2026-10-03). It replaced the host loop over queries (each
query's probed lists merged on the host, its candidate vectors and norms
gathered and uploaded, one distance launch, one selection launch, a
download and a host sort per query) that served k > 32, dim > FIVF_MAX_DIM,
traced, partial-storage and short-query searches.

THE ANSWER IS THE ONE THE HOST LOOP GAVE, BIT FOR BIT.

  * The candidates of query q are the slots of its probed lists (the kept
    ones under a filter), ordered by ORIGINAL index (`merge_probed_lists`'
    order, DEVIATION 1786). Each list's slice ascends in the original id,
    so a slot's position in q's merged row is the count of q's candidates
    with a smaller id: a binary search per probed list (`_cand_rank_kernel`,
    one block per (query, probe), threads over the list). Integers only.
  * Each (query, candidate) distance is `pinned_distance_tile_kernel`'s
    cell, statement for statement (one ascending `identical_mul_add` chain
    over d, the norm sum, the `-2` fma, the clamp at zero), reading the
    candidate row and its norm straight from the device list data and list
    norms the host gather copied from.
  * The k nearest of q are the k smallest of its row under the total order
    (distance, original index): the host path's radix selection keyed on
    (distance, position) with positions in original-id order, then its
    (distance, index) insertion sort. Here two stable device radix sorts
    (by the distance's sortable bits, then by query) put every query's
    candidates in exactly that order, and the first min(k, count) of each
    are the answer. The clamp makes every distance >= +0.0, so the sortable
    bits order as the floats do.

The queries go in batches whose candidate total is capped (a function of
the counts and the index size only); a query's answer does not depend on
its batch.
"""

from std.gpu import block_dim, block_idx, thread_idx
from std.memory import bitcast
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.numerics import ftz, identical_mul_add
from core.device_fold import device_exclusive_scan_total_from
from core.fast_radix_sort import fast_radix_sort_pairs_u32, frs_counts_len
from core.segmented_sort import float_to_sortable

comptime _TPB = 256
comptime _I32P = MutPointer[Int32, MutAnyOrigin]
comptime _U32P = MutPointer[UInt32, MutAnyOrigin]
comptime _F32P = MutPointer[Float32, MutAnyOrigin]

#: the per-batch candidate cap floor (the cap is at least the index size, the
#: most one query can hold)
comptime IVF_QUERY_BATCH_CANDIDATES = 1 << 22


def _grid(n: Int) -> Int:
    return max((n + _TPB - 1) // _TPB, 1)


@always_inline
def _lower_bound_u32(a: _U32P, lo_in: Int, hi_in: Int, v: UInt32) -> Int:
    """The first index i in [lo, hi) with a[i] >= v (a ascending there)."""
    var lo = lo_in
    var hi = hi_in
    while lo < hi:
        var mid = (lo + hi) // 2
        if a[mid] < v:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _kept_flags_kernel(ind: _U32P, keep: _I32P, n_slots_in: Int32, flags: _I32P):
    """flags[s] = 1 when slot s's original row passes the filter."""
    var s = Int(block_idx.x) * _TPB + Int(thread_idx.x)
    if s < Int(n_slots_in):
        flags[s] = Int32(1) if keep[Int(ind[s])] != 0 else Int32(0)


def _cand_rank_kernel(
    probe: _U32P, off: _I32P, ind: _U32P, kp: _I32P, filtered: Int32,
    q0_in: Int32, n_probes_in: Int32, coff: _I32P,
    cslot: _I32P, corig: _U32P, cq: _U32P,
):
    """Block (qq, p): the slots of query q0 + qq's p-th probed list, each
    written to its position in q's candidate row (the count of q's
    candidates with a smaller original id), offset by coff[qq]."""
    var n_probes = Int(n_probes_in)
    var pair = Int(block_idx.x)
    var qq = pair // n_probes
    var p = pair - qq * n_probes
    var q = Int(q0_in) + qq
    var l = Int(probe[q * n_probes + p])
    var s0 = Int(off[l])
    var s1 = Int(off[l + 1])
    var base = Int(coff[qq])
    var s = s0 + Int(thread_idx.x)
    while s < s1:
        var orig = ind[s]
        var live = True
        if filtered != 0:
            live = kp[s + 1] != kp[s]
        if live:
            var rank = 0
            for pp in range(n_probes):
                var l2 = Int(probe[q * n_probes + pp])
                var a = Int(off[l2])
                var b = Int(off[l2 + 1])
                var t = _lower_bound_u32(ind, a, b, orig)
                if filtered != 0:
                    rank += Int(kp[t]) - Int(kp[a])
                else:
                    rank += t - a
            var g = base + rank
            cslot[g] = Int32(s)
            corig[g] = orig
            cq[g] = UInt32(qq)
        s += Int(block_dim.x)


def _cand_dist_kernel(
    cslot: _I32P, cq: _U32P, q0_in: Int32, total_in: Int32, dim_in: Int32,
    xq: _F32P, xq_norm: _F32P, ldata: _F32P, lnorm: _F32P,
    cdist: _F32P, keys: _U32P, vals: _U32P,
):
    """Candidate g: `pinned_distance_tile_kernel`'s cell for (its query, its
    slot), into cdist[g]; the sort pair (sortable distance bits, g)."""
    var g = Int(block_idx.x) * _TPB + Int(thread_idx.x)
    if g >= Int(total_in):
        return
    var d = Int(dim_in)
    var row = Int(q0_in) + Int(cq[g])
    var col = Int(cslot[g])
    var acc = Float32(0.0)
    for f in range(d):
        var qv = ftz(xq.unsafe_load(row * d + f))
        var yv = ftz(ldata.unsafe_load(col * d + f))
        acc = ftz(identical_mul_add(qv, yv, acc))
    var dist = ftz(
        identical_mul_add(
            Float32(-2.0),
            acc,
            ftz(ftz(xq_norm.unsafe_load(row)) + ftz(lnorm.unsafe_load(col))),
        )
    )
    if dist <= Float32(0.0):
        dist = Float32(0.0)
    cdist[g] = dist
    keys[g] = float_to_sortable(bitcast[DType.uint32](dist))
    vals[g] = UInt32(g)


def _query_keys_kernel(vals: _U32P, cq: _U32P, total_in: Int32, keys: _U32P):
    var i = Int(block_idx.x) * _TPB + Int(thread_idx.x)
    if i < Int(total_in):
        keys[i] = cq[Int(vals[i])]


def _emit_kernel(
    vals: _U32P, coff: _I32P, cdist: _F32P, corig: _U32P,
    q0_in: Int32, nqb_in: Int32, k_in: Int32, out_d: _F32P, out_i: _U32P,
):
    """Cell (qq, i): the i-th nearest of query q0 + qq (its sorted run
    starts at coff[qq]), or 0 past its candidate count."""
    var t = Int(block_idx.x) * _TPB + Int(thread_idx.x)
    var k = Int(k_in)
    if t >= Int(nqb_in) * k:
        return
    var qq = t // k
    var i = t - qq * k
    var a = Int(coff[qq])
    var c = Int(coff[qq + 1]) - a
    var o = (Int(q0_in) + qq) * k + i
    if i < c:
        var g = Int(vals[a + i])
        out_d[o] = cdist[g]
        out_i[o] = corig[g]
    else:
        out_d[o] = Float32(0.0)
        out_i[o] = UInt32(0)


def ivf_query_batches(counts: List[Int32], n_queries: Int, cap: Int) -> List[Int]:
    """Batch starts (plus n_queries at the end): consecutive queries while
    their candidate total stays within `cap` (a query alone may reach it)."""
    var starts = List[Int]()
    starts.append(0)
    var acc = 0
    for q in range(n_queries):
        var c = Int(counts[q])
        if acc > 0 and acc + c > cap:
            starts.append(q)
            acc = 0
        acc += c
    starts.append(n_queries)
    return starts^


struct IvfQueryBatch(Movable):
    """One batch's device arrays, kept for the trace (corig, cdist in
    candidate order) after the answer is emitted."""

    var total: Int
    var corig: DeviceBuffer[DType.uint32]
    var cdist: DeviceBuffer[DType.float32]

    def __init__(out self, total: Int, var corig: DeviceBuffer[DType.uint32], var cdist: DeviceBuffer[DType.float32]):
        self.total = total
        self.corig = corig^
        self.cdist = cdist^


def ivf_kept_prefix_device(
    ctx: DeviceContext,
    mut ind: DeviceBuffer[DType.uint32],
    mut keep: DeviceBuffer[DType.int32],
    n_slots: Int,
) raises -> DeviceBuffer[DType.int32]:
    """kp[0 .. n_slots]: the kept slots before each slot (an exclusive scan
    of the filter over the slot order)."""
    var flags = ctx.enqueue_create_buffer[DType.int32](max(n_slots, 1))
    var kp = ctx.enqueue_create_buffer[DType.int32](n_slots + 1)
    if n_slots > 0:
        ctx.enqueue_function[_kept_flags_kernel](
            ind.unsafe_ptr(), keep.unsafe_ptr(), Int32(n_slots), flags.unsafe_ptr(),
            grid_dim=_grid(n_slots), block_dim=_TPB,
        )
    device_exclusive_scan_total_from(ctx, flags, kp, n_slots)
    ctx.synchronize()
    _ = flags^
    return kp^


def ivf_query_batch_device(
    ctx: DeviceContext,
    mut probe: DeviceBuffer[DType.uint32],
    mut off: DeviceBuffer[DType.int32],
    mut ind: DeviceBuffer[DType.uint32],
    mut kp: DeviceBuffer[DType.int32],
    filtered: Bool,
    mut counts: DeviceBuffer[DType.int32],
    mut xq: DeviceBuffer[DType.float32],
    mut xq_norm: DeviceBuffer[DType.float32],
    mut ldata: DeviceBuffer[DType.float32],
    mut lnorm: DeviceBuffer[DType.float32],
    q0: Int,
    q1: Int,
    total: Int,
    n_probes: Int,
    dim: Int,
    k: Int,
    mut out_d: DeviceBuffer[DType.float32],
    mut out_i: DeviceBuffer[DType.uint32],
) raises -> IvfQueryBatch:
    """Queries [q0, q1) (their candidates `total` in all): the candidate
    rows, the distances, the two stable sorts and the answer cells of
    `out_d` / `out_i` (n_queries x k). Returns the batch's candidate ids and
    distances in candidate order (the trace's)."""
    var nqb = q1 - q0
    var n = max(total, 1)
    var coff = ctx.enqueue_create_buffer[DType.int32](nqb + 1)
    var csub = counts.create_sub_buffer[DType.int32](q0, nqb)
    device_exclusive_scan_total_from(ctx, csub, coff, nqb)
    var cslot = ctx.enqueue_create_buffer[DType.int32](n)
    var corig = ctx.enqueue_create_buffer[DType.uint32](n)
    var cq = ctx.enqueue_create_buffer[DType.uint32](n)
    var cdist = ctx.enqueue_create_buffer[DType.float32](n)
    var keys = ctx.enqueue_create_buffer[DType.uint32](n)
    var vals = ctx.enqueue_create_buffer[DType.uint32](n)
    var tk = ctx.enqueue_create_buffer[DType.uint32](n)
    var tv = ctx.enqueue_create_buffer[DType.uint32](n)
    var cnt = ctx.enqueue_create_buffer[DType.int32](max(frs_counts_len(total), 1))
    if total > 0:
        ctx.enqueue_function[_cand_rank_kernel](
            probe.unsafe_ptr(), off.unsafe_ptr(), ind.unsafe_ptr(), kp.unsafe_ptr(),
            Int32(1) if filtered else Int32(0), Int32(q0), Int32(n_probes), coff.unsafe_ptr(),
            cslot.unsafe_ptr(), corig.unsafe_ptr(), cq.unsafe_ptr(),
            grid_dim=nqb * n_probes, block_dim=_TPB,
        )
        ctx.enqueue_function[_cand_dist_kernel](
            cslot.unsafe_ptr(), cq.unsafe_ptr(), Int32(q0), Int32(total), Int32(dim),
            xq.unsafe_ptr(), xq_norm.unsafe_ptr(), ldata.unsafe_ptr(), lnorm.unsafe_ptr(),
            cdist.unsafe_ptr(), keys.unsafe_ptr(), vals.unsafe_ptr(),
            grid_dim=_grid(total), block_dim=_TPB,
        )
        # by distance (stable: original-id order among equal distances),
        # then by query (stable: that order within a query)
        fast_radix_sort_pairs_u32(ctx, total, keys, vals, tk, tv, cnt)
        ctx.enqueue_function[_query_keys_kernel](
            vals.unsafe_ptr(), cq.unsafe_ptr(), Int32(total), keys.unsafe_ptr(),
            grid_dim=_grid(total), block_dim=_TPB,
        )
        fast_radix_sort_pairs_u32(ctx, total, keys, vals, tk, tv, cnt)
    if nqb > 0 and k > 0:
        ctx.enqueue_function[_emit_kernel](
            vals.unsafe_ptr(), coff.unsafe_ptr(), cdist.unsafe_ptr(), corig.unsafe_ptr(),
            Int32(q0), Int32(nqb), Int32(k), out_d.unsafe_ptr(), out_i.unsafe_ptr(),
            grid_dim=_grid(nqb * k), block_dim=_TPB,
        )
    ctx.synchronize()
    _ = csub^
    _ = coff^
    _ = cslot^
    _ = cq^
    _ = keys^
    _ = vals^
    _ = tk^
    _ = tv^
    _ = cnt^
    return IvfQueryBatch(total, corig^, cdist^)
