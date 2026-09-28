# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE PARALLEL IVF SEARCH (PQ, SQ, RaBitQ) ON THE DEVICE, SAME BITS
(lane/algos-ann-speed, phase C, 2026-09-28).

The pass-1 search kernels ran one thread per query: the probe walk
recomputed every coarse distance once per probe (`pq_next_probe`), and the
IVF-PQ scan recomputed every lookup-table entry once per ROW
(`pq_lut_entry` inside the code sum), so a 1M-row search did pq_len times
the arithmetic of a lookup table and used one thread where a GPU has
thousands. This file restates the same search as four launches whose every
value is computed by the SAME core function as before:

  1. `coarse_kernel`: one thread per (query, list): `pq_coarse_dist`, the
     value `pq_next_probe` computes, stored once.
  2. `probes_kernel`: one thread per query: `pq_select_probes` walks the
     stored values with `pq_next_probe`'s rule (the next list after
     (prev_d, prev_l) in (distance, list id)), so the probe order is the
     same list sequence.
  3. a scan kernel, ONE BLOCK PER QUERY: the rows of each probed list are
     split over the block's threads; each row's score is formed exactly as
     the per-query cell forms it (the IVF-PQ code sum in ascending subspace
     order from the lookup-table entry `pq_lut_entry` returns, now read from
     a table the block fills once per probe; SQ and RaBitQ inline, unchanged);
     each thread keeps its own top-k through `pq_insert`'s order (`pq_better`).
  4. the block merge: k rounds, each taking the least head under (distance,
     row id) over every thread's list.

WHY THE BITS DO NOT MOVE. Every score is the same instruction sequence on
the same operands. The selected set and its order are the k least
(distance, id) pairs of the scored rows under a TOTAL order (row ids are
unique), which does not depend on which thread scored a row or in what
order (IDENTITY_PATHS rows 11 and 23: the property `pq_insert` already
relies on). The candidate count is an integer sum. Short answers keep the
(+inf, -1) fill.

THE LOOKUP TABLE IN TILES. Metal gives a threadgroup 32 KB, so the table
lives in shared memory LUT_TILE floats at a time: subspaces [j0, j1) per
tile, and a row's running code sum is carried between tiles in a per-block
scratch slot that only the thread owning that row reads and writes (the
same thread owns the same row in every tile: slot = start + tid + t*TPB).
The sum is still `ftz(total + entry)` over j ascending from +0.0.

k larger than `SCAN_MAX_K` keeps the pass-1 per-query kernels.
"""

from std.gpu import block_idx, block_dim, thread_idx
from std.memory import stack_allocation, bitcast
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from max.gpu.host import DeviceBuffer, DeviceContext

from checks.numerics import ftz, identical_mul_add
from x_ann.ivf_pq_core import (
    F32P, I32P, pq_better, pq_coarse_dist, pq_inf, pq_lut_entry, pq_row_removed, pq_select_probes,
)
from x_ann.ivf_sq_core import sq_row_score
from x_ann.ivf_rabitq_core import rq_rotate, rq_row_estimate

comptime SCAN_TPB = 128
comptime LUT_TILE = 4096
comptime SCAN_MAX_K = 256


def _gid() -> Int:
    return Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)


def coarse_kernel(count: Int32, q0: Int32, queries: F32P, dim: Int32, centers: F32P, n_lists: Int32, dist: F32P):
    """dist[(q - q0) * n_lists + l] = pq_coarse_dist(query q, list l)."""
    var e = _gid()
    if e < Int(count):
        var nl = Int(n_lists)
        var q = Int(q0) + e // nl
        dist.unsafe_store(e, pq_coarse_dist(queries, q * Int(dim), centers, e % nl, Int(dim)))


def probes_kernel(count: Int32, dist: F32P, n_lists: Int32, n_probes: Int32, probes: I32P):
    var e = _gid()
    if e < Int(count):
        pq_select_probes(dist, e * Int(n_lists), Int(n_lists), Int(n_probes), probes, e * Int(n_probes))


@always_inline
def _tk_insert[KM: Int](k: Int, d: Float32, id: Int32, mut td: InlineArray[Float32, KM], mut ti: InlineArray[Int32, KM]):
    """`pq_insert` on a thread's own list: the same `pq_better` order (an
    empty slot, id -1, loses to any row)."""
    if not pq_better(d, id, td[k - 1], ti[k - 1]):
        return
    var s = k - 1
    while s > 0 and pq_better(d, id, td[s - 1], ti[s - 1]):
        td[s] = td[s - 1]
        ti[s] = ti[s - 1]
        s -= 1
    td[s] = d
    ti[s] = id


@always_inline
def _block_merge[KM: Int](
    k: Int, qi: Int, n_cand: Int, mut td: InlineArray[Float32, KM], mut ti: InlineArray[Int32, KM],
    out_d: F32P, out_i: I32P, out_n: I32P,
):
    """k rounds: the least head over every thread's sorted list under
    (distance, id); the owner pops it. Then the candidate count."""
    var tid = Int(thread_idx.x)
    var hd = stack_allocation[SCAN_TPB, Float32, address_space=AddressSpace.SHARED]()
    var hi = stack_allocation[SCAN_TPB, Int32, address_space=AddressSpace.SHARED]()
    var hc = stack_allocation[SCAN_TPB, Int32, address_space=AddressSpace.SHARED]()
    var win = stack_allocation[1, Int32, address_space=AddressSpace.SHARED]()
    hc[tid] = Int32(n_cand)
    var head = 0
    for r in range(k):
        barrier()
        if head < k:
            hd[tid] = td[head]
            hi[tid] = ti[head]
        else:
            hd[tid] = pq_inf()
            hi[tid] = Int32(-1)
        barrier()
        if tid == 0:
            var bt = -1
            for t in range(SCAN_TPB):
                if hi[t] >= 0 and (bt < 0 or pq_better(hd[t], hi[t], hd[bt], hi[bt])):
                    bt = t
            win[0] = Int32(bt)
            if bt < 0:
                out_d.unsafe_store(qi * k + r, pq_inf())
                out_i.unsafe_store(qi * k + r, Int32(-1))
            else:
                out_d.unsafe_store(qi * k + r, hd[bt])
                out_i.unsafe_store(qi * k + r, hi[bt])
        barrier()
        if Int(win[0]) == tid:
            head += 1
    if tid == 0:
        var total = 0
        for t in range(SCAN_TPB):
            total += Int(hc[t])
        out_n.unsafe_store(qi, Int32(total))


def pq_scan_kernel[KM: Int](
    q0: Int32, queries: F32P, dim: Int32, centers: F32P, offsets: I32P, list_indices: I32P, codes: I32P,
    cb: F32P, pq_dim: Int32, pq_len: Int32, n_codes: Int32, k: Int32, n_probes: Int32, probes: I32P,
    mask: I32P, partial: F32P, max_list: Int32, out_d: F32P, out_i: I32P, out_n: I32P,
):
    """One block per query: the IVF-PQ code sum of `pq_search_cell`, the
    lookup table filled once per probe (per tile) instead of once per row."""
    var tid = Int(thread_idx.x)
    var qi = Int(q0) + Int(block_idx.x)
    var d = Int(dim)
    var pqd = Int(pq_dim)
    var pl = Int(pq_len)
    var nc = Int(n_codes)
    var kk = Int(k)
    var np = Int(n_probes)
    var jt = LUT_TILE // nc
    var lut = stack_allocation[LUT_TILE, Float32, address_space=AddressSpace.SHARED]()
    var pbase = Int(block_idx.x) * Int(max_list)
    var td = InlineArray[Float32, KM](fill=pq_inf())
    var ti = InlineArray[Int32, KM](fill=Int32(-1))
    var n_cand = 0
    for p in range(np):
        var l = Int(probes.unsafe_load(qi * np + p))
        if l < 0:
            break
        var start = Int(offsets.unsafe_load(l))
        var stop = Int(offsets.unsafe_load(l + 1))
        var j0 = 0
        while j0 < pqd:
            var j1 = j0 + jt if j0 + jt < pqd else pqd
            barrier()
            var e = tid
            while e < (j1 - j0) * nc:
                lut[e] = pq_lut_entry(queries, qi * d, centers, l, d, cb, j0 + e // nc, e % nc, pl, nc)
                e += SCAN_TPB
            barrier()
            var slot = start + tid
            while slot < stop:
                var row = Int(list_indices.unsafe_load(slot))
                if not pq_row_removed(mask, row):
                    var total = Float32(0.0)
                    if j0 > 0:
                        total = partial.unsafe_load(pbase + slot - start)
                    for j in range(j0, j1):
                        var code = Int(codes.unsafe_load(row * pqd + j))
                        total = ftz(total + lut[(j - j0) * nc + code])
                    if j1 == pqd:
                        _tk_insert[KM](kk, total, Int32(row), td, ti)
                        n_cand += 1
                    else:
                        partial.unsafe_store(pbase + slot - start, total)
                slot += SCAN_TPB
            j0 = j1
    _block_merge[KM](kk, qi, n_cand, td, ti, out_d, out_i, out_n)


def sq_scan_kernel[KM: Int](
    q0: Int32, queries: F32P, dim: Int32, centers: F32P, offsets: I32P, list_indices: I32P, codes: I32P,
    vmin: F32P, delta: F32P, k: Int32, n_probes: Int32, probes: I32P, mask: I32P,
    out_d: F32P, out_i: I32P, out_n: I32P,
):
    """One block per query: `sq_search_cell`'s row score, term for term."""
    var tid = Int(thread_idx.x)
    var qi = Int(q0) + Int(block_idx.x)
    var d = Int(dim)
    var kk = Int(k)
    var np = Int(n_probes)
    var q_off = qi * d
    var td = InlineArray[Float32, KM](fill=pq_inf())
    var ti = InlineArray[Int32, KM](fill=Int32(-1))
    var n_cand = 0
    for p in range(np):
        var l = Int(probes.unsafe_load(qi * np + p))
        if l < 0:
            break
        var slot = Int(offsets.unsafe_load(l)) + tid
        var stop = Int(offsets.unsafe_load(l + 1))
        while slot < stop:
            var row = Int(list_indices.unsafe_load(slot))
            if not pq_row_removed(mask, row):
                _tk_insert[KM](kk, sq_row_score(queries, q_off, centers, l, d, codes, row, vmin, delta),
                               Int32(row), td, ti)
                n_cand += 1
            slot += SCAN_TPB
    _block_merge[KM](kk, qi, n_cand, td, ti, out_d, out_i, out_n)


def rq_prep_kernel(
    count: Int32, q0: Int32, queries: F32P, dim: Int32, centers: F32P, n_probes: Int32, probes: I32P,
    D: Int32, seed: Int32, scale: Float32, ws: F32P, qn2: F32P,
):
    """One thread per (query, probe): `rq_search_cell`'s rotated query
    residual and its squared norm, stored for the scan."""
    var e = _gid()
    if e < Int(count):
        var np = Int(n_probes)
        var q = Int(q0) + e // np
        var l = Int(probes.unsafe_load(q * np + e % np))
        if l < 0:
            return
        var DD = Int(D)
        rq_rotate(queries, q * Int(dim), centers, l * Int(dim), Int(dim), DD, Int(seed), scale, ws, e * DD)
        var s = Float32(0.0)
        for j in range(DD):
            var v = ws.unsafe_load(e * DD + j)
            s = ftz(identical_mul_add(v, v, s))
        qn2.unsafe_store(e, s)


def rq_scan_kernel[KM: Int](
    q0: Int32, offsets: I32P, list_indices: I32P, codes: I32P, norms: F32P, ips: F32P, D: Int32, words: Int32,
    scale: Float32, k: Int32, n_probes: Int32, probes: I32P, mask: I32P, ws: F32P, qn2: F32P,
    out_d: F32P, out_i: I32P, out_n: I32P,
):
    """One block per query: `rq_search_cell`'s estimate, term for term."""
    var tid = Int(thread_idx.x)
    var ql = Int(block_idx.x)
    var qi = Int(q0) + ql
    var DD = Int(D)
    var w = Int(words)
    var kk = Int(k)
    var np = Int(n_probes)
    var td = InlineArray[Float32, KM](fill=pq_inf())
    var ti = InlineArray[Int32, KM](fill=Int32(-1))
    var n_cand = 0
    for p in range(np):
        var l = Int(probes.unsafe_load(qi * np + p))
        if l < 0:
            break
        var wb = (ql * np + p) * DD
        var q2 = qn2.unsafe_load(ql * np + p)
        var slot = Int(offsets.unsafe_load(l)) + tid
        var stop = Int(offsets.unsafe_load(l + 1))
        while slot < stop:
            var row = Int(list_indices.unsafe_load(slot))
            if not pq_row_removed(mask, row):
                _tk_insert[KM](kk, rq_row_estimate(ws, wb, q2, codes, row, w, DD, scale, norms, ips),
                               Int32(row), td, ti)
                n_cand += 1
            slot += SCAN_TPB
    _block_merge[KM](kk, qi, n_cand, td, ti, out_d, out_i, out_n)


def _grid(count: Int) -> Int:
    return (count + SCAN_TPB - 1) // SCAN_TPB


def scan_probes(
    ctx: DeviceContext, dq: DeviceBuffer[DType.float32], dc: DeviceBuffer[DType.float32], m: Int, dim: Int,
    n_lists: Int, n_probes: Int, mut dprobes: DeviceBuffer[DType.int32],
) raises:
    """Launches 1 and 2 over query chunks: dprobes[q * n_probes + p]."""
    var chunk = (1 << 24) // n_lists if n_lists < (1 << 24) else 1
    if chunk < 1:
        chunk = 1
    if chunk > m:
        chunk = m
    var ddist = ctx.enqueue_create_buffer[DType.float32](chunk * n_lists)
    var q0 = 0
    while q0 < m:
        var c = chunk if q0 + chunk <= m else m - q0
        ctx.enqueue_function[coarse_kernel](
            Int32(c * n_lists), Int32(q0), dq.unsafe_ptr(), Int32(dim), dc.unsafe_ptr(), Int32(n_lists),
            ddist.unsafe_ptr(), grid_dim=_grid(c * n_lists), block_dim=SCAN_TPB,
        )
        var sub = dprobes.create_sub_buffer[DType.int32](q0 * n_probes, c * n_probes)
        ctx.enqueue_function[probes_kernel](
            Int32(c), ddist.unsafe_ptr(), Int32(n_lists), Int32(n_probes), sub.unsafe_ptr(),
            grid_dim=_grid(c), block_dim=SCAN_TPB,
        )
        q0 += c
    ctx.synchronize()
    _ = ddist^


def max_list_len(offsets: List[Int32], n_lists: Int) -> Int:
    var mx = 1
    for l in range(n_lists):
        var s = Int(offsets[l + 1]) - Int(offsets[l])
        if s > mx:
            mx = s
    return mx


def _km_launch_pq[KM: Int](
    ctx: DeviceContext, c: Int, q0: Int, queries: F32P, dim: Int, centers: F32P, offsets: I32P, list_indices: I32P,
    codes: I32P, cb: F32P, pq_dim: Int, pq_len: Int, n_codes: Int, k: Int, n_probes: Int, probes: I32P, mask: I32P,
    partial: F32P, max_list: Int, out_d: F32P, out_i: I32P, out_n: I32P,
) raises:
    ctx.enqueue_function[pq_scan_kernel[KM]](
        Int32(q0), queries, Int32(dim), centers, offsets, list_indices, codes, cb, Int32(pq_dim), Int32(pq_len),
        Int32(n_codes), Int32(k), Int32(n_probes), probes, mask, partial, Int32(max_list), out_d, out_i, out_n,
        grid_dim=c, block_dim=SCAN_TPB,
    )


def pq_scan(
    ctx: DeviceContext, m: Int, queries: F32P, dim: Int, centers: F32P, n_lists: Int, offsets: List[Int32],
    doffsets: I32P, list_indices: I32P, codes: I32P, cb: F32P, pq_dim: Int, pq_len: Int, n_codes: Int, k: Int,
    n_probes: Int, dq: DeviceBuffer[DType.float32], dc: DeviceBuffer[DType.float32], mask: I32P,
    out_d: F32P, out_i: I32P, out_n: I32P,
) raises:
    """Launches 1-4 for IVF-PQ (k <= SCAN_MAX_K, m > 0)."""
    var dprobes = ctx.enqueue_create_buffer[DType.int32](m * n_probes)
    scan_probes(ctx, dq, dc, m, dim, n_lists, n_probes, dprobes)
    var ml = max_list_len(offsets, n_lists)
    var chunk = (1 << 26) // ml
    if chunk < 1:
        chunk = 1
    if chunk > m:
        chunk = m
    var dpart = ctx.enqueue_create_buffer[DType.float32](chunk * ml)
    var q0 = 0
    while q0 < m:
        var c = chunk if q0 + chunk <= m else m - q0
        if k <= 16:
            _km_launch_pq[16](ctx, c, q0, queries, dim, centers, doffsets, list_indices, codes, cb, pq_dim, pq_len,
                              n_codes, k, n_probes, dprobes.unsafe_ptr(), mask, dpart.unsafe_ptr(), ml, out_d, out_i, out_n)
        elif k <= 64:
            _km_launch_pq[64](ctx, c, q0, queries, dim, centers, doffsets, list_indices, codes, cb, pq_dim, pq_len,
                              n_codes, k, n_probes, dprobes.unsafe_ptr(), mask, dpart.unsafe_ptr(), ml, out_d, out_i, out_n)
        else:
            _km_launch_pq[SCAN_MAX_K](ctx, c, q0, queries, dim, centers, doffsets, list_indices, codes, cb, pq_dim,
                                      pq_len, n_codes, k, n_probes, dprobes.unsafe_ptr(), mask, dpart.unsafe_ptr(), ml,
                                      out_d, out_i, out_n)
        q0 += c
    ctx.synchronize()
    _ = dpart^
    _ = dprobes^


def _km_launch_sq[KM: Int](
    ctx: DeviceContext, m: Int, queries: F32P, dim: Int, centers: F32P, offsets: I32P, list_indices: I32P,
    codes: I32P, vmin: F32P, delta: F32P, k: Int, n_probes: Int, probes: I32P, mask: I32P,
    out_d: F32P, out_i: I32P, out_n: I32P,
) raises:
    ctx.enqueue_function[sq_scan_kernel[KM]](
        Int32(0), queries, Int32(dim), centers, offsets, list_indices, codes, vmin, delta, Int32(k), Int32(n_probes),
        probes, mask, out_d, out_i, out_n, grid_dim=m, block_dim=SCAN_TPB,
    )


def sq_scan(
    ctx: DeviceContext, m: Int, queries: F32P, dim: Int, centers: F32P, n_lists: Int, offsets: I32P,
    list_indices: I32P, codes: I32P, vmin: F32P, delta: F32P, k: Int, n_probes: Int,
    dq: DeviceBuffer[DType.float32], dc: DeviceBuffer[DType.float32], mask: I32P,
    out_d: F32P, out_i: I32P, out_n: I32P,
) raises:
    """Launches 1-4 for IVF-SQ (k <= SCAN_MAX_K, m > 0)."""
    var dprobes = ctx.enqueue_create_buffer[DType.int32](m * n_probes)
    scan_probes(ctx, dq, dc, m, dim, n_lists, n_probes, dprobes)
    var pr = dprobes.unsafe_ptr()
    if k <= 16:
        _km_launch_sq[16](ctx, m, queries, dim, centers, offsets, list_indices, codes, vmin, delta, k, n_probes, pr,
                          mask, out_d, out_i, out_n)
    elif k <= 64:
        _km_launch_sq[64](ctx, m, queries, dim, centers, offsets, list_indices, codes, vmin, delta, k, n_probes, pr,
                          mask, out_d, out_i, out_n)
    else:
        _km_launch_sq[SCAN_MAX_K](ctx, m, queries, dim, centers, offsets, list_indices, codes, vmin, delta, k,
                                  n_probes, pr, mask, out_d, out_i, out_n)
    ctx.synchronize()
    _ = dprobes^


def _km_launch_rq[KM: Int](
    ctx: DeviceContext, c: Int, q0: Int, offsets: I32P, list_indices: I32P, codes: I32P, norms: F32P, ips: F32P,
    D: Int, words: Int, scale: Float32, k: Int, n_probes: Int, probes: I32P, mask: I32P, ws: F32P, qn2: F32P,
    out_d: F32P, out_i: I32P, out_n: I32P,
) raises:
    ctx.enqueue_function[rq_scan_kernel[KM]](
        Int32(q0), offsets, list_indices, codes, norms, ips, Int32(D), Int32(words), scale, Int32(k), Int32(n_probes),
        probes, mask, ws, qn2, out_d, out_i, out_n, grid_dim=c, block_dim=SCAN_TPB,
    )


def rq_scan(
    ctx: DeviceContext, m: Int, queries: F32P, dim: Int, centers: F32P, n_lists: Int, offsets: I32P,
    list_indices: I32P, codes: I32P, norms: F32P, ips: F32P, D: Int, words: Int, seed: Int, scale: Float32,
    k: Int, n_probes: Int, dq: DeviceBuffer[DType.float32], dc: DeviceBuffer[DType.float32], mask: I32P,
    out_d: F32P, out_i: I32P, out_n: I32P,
) raises:
    """Launches 1, 2, the rotation prep and the scan for IVF-RaBitQ."""
    var dprobes = ctx.enqueue_create_buffer[DType.int32](m * n_probes)
    scan_probes(ctx, dq, dc, m, dim, n_lists, n_probes, dprobes)
    var per_q = n_probes * D
    var chunk = (1 << 26) // per_q
    if chunk < 1:
        chunk = 1
    if chunk > m:
        chunk = m
    var dws = ctx.enqueue_create_buffer[DType.float32](chunk * per_q)
    var dqn = ctx.enqueue_create_buffer[DType.float32](chunk * n_probes)
    var pr = dprobes.unsafe_ptr()
    var q0 = 0
    while q0 < m:
        var c = chunk if q0 + chunk <= m else m - q0
        ctx.enqueue_function[rq_prep_kernel](
            Int32(c * n_probes), Int32(q0), queries, Int32(dim), centers, Int32(n_probes), pr, Int32(D), Int32(seed),
            scale, dws.unsafe_ptr(), dqn.unsafe_ptr(), grid_dim=_grid(c * n_probes), block_dim=SCAN_TPB,
        )
        if k <= 16:
            _km_launch_rq[16](ctx, c, q0, offsets, list_indices, codes, norms, ips, D, words, scale, k, n_probes, pr,
                              mask, dws.unsafe_ptr(), dqn.unsafe_ptr(), out_d, out_i, out_n)
        elif k <= 64:
            _km_launch_rq[64](ctx, c, q0, offsets, list_indices, codes, norms, ips, D, words, scale, k, n_probes, pr,
                              mask, dws.unsafe_ptr(), dqn.unsafe_ptr(), out_d, out_i, out_n)
        else:
            _km_launch_rq[SCAN_MAX_K](ctx, c, q0, offsets, list_indices, codes, norms, ips, D, words, scale, k,
                                      n_probes, pr, mask, dws.unsafe_ptr(), dqn.unsafe_ptr(), out_d, out_i, out_n)
        q0 += c
    ctx.synchronize()
    _ = dqn^
    _ = dws^
    _ = dprobes^
