# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The IVF-PQ / IVF-SQ / IVF-RaBitQ device search, split into launches that
fill the GPU (lane ann-apple, 2026-09-28). SAME BITS as one thread per query
running `pq_search_cell` / `sq_search_cell` / `rq_search_cell`:

  1. `coarse_kernel`: one thread per (query, list), `pq_coarse_dist`, the
     cell `pq_next_probe` evaluates, stored once instead of once per probe.
  2. `probe_kernel`: one thread per query, `pq_next_probe`'s walk over those
     stored values (the same comparisons on the same words, so the same
     lists in the same order; the host twin is `_probe_walk`).
  3. a score kernel, one threadgroup per (query, probe), one thread per list
     slot: each candidate's distance by the cell's own statements (the PQ
     lookup entries `pq_lut_entry` computed once per (query, probe) into
     threadgroup memory -- the same function of the same inputs, so the same
     words -- and summed over subspaces ascending as before).
  4. `select_kernel`: one thread per query, the cell's `pq_insert` over the
     stored distances in the cell's order (probe order, then slot order,
     masked rows skipped), so the top-k, NaN handling included, and the
     candidate count are the cell's.

The old kernels ran ~1000 threads (one per query) that recomputed every
coarse distance for every probe; this runs one thread per candidate.
Queries go in chunks so the candidate buffer stays under CAND_BUDGET floats."""

from std.gpu import block_idx, block_dim, thread_idx
from std.memory import stack_allocation
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.numerics import ftz, identical_mul_add
from x_ann.ivf_pq_core import (
    F32P, I32P, ivf_row_removed, pq_coarse_dist, pq_inf, pq_insert, pq_lut_entry, pq_probe_takes,
)
from x_ann.ivf_sq_core import sq_candidate_dist
from x_ann.ivf_rabitq_core import rq_candidate_est, rq_rotate

comptime TPB = 128
#: threads per (query, probe) threadgroup
comptime STPB = 128
#: PQ lookup entries held in threadgroup memory (16 KB); a larger table
#: evaluates each entry where it is read, as the cell does
comptime LUT_MAX = 4096
#: candidate distances per chunk of queries (64 MB)
comptime CAND_BUDGET = 16 * 1024 * 1024


def _tid() -> Int:
    return Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)


def _grid(count: Int) -> Int:
    return (count + TPB - 1) // TPB


def coarse_kernel(count: Int32, q0: Int32, queries: F32P, dim: Int32, centers: F32P, n_lists: Int32, cd: F32P):
    var e = _tid()
    if e < Int(count):
        var lq = e // Int(n_lists)
        var l = e % Int(n_lists)
        cd.unsafe_store(e, pq_coarse_dist(queries, (Int(q0) + lq) * Int(dim), centers, l, Int(dim)))


def probe_kernel(mc: Int32, cd: F32P, n_lists: Int32, n_probes: Int32, offsets: I32P, probes: I32P, pstart: I32P):
    """`pq_next_probe`'s walk over the stored coarse distances; an unused
    probe is -1. pstart = the probe's first position in the query's
    candidate row (probes in walk order, slots in list order)."""
    var lq = _tid()
    if lq < Int(mc):
        var nl = Int(n_lists)
        var np = Int(n_probes)
        var row = lq * nl
        var prev_d = Float32(0.0)
        var prev_l = -1
        var pos = 0
        var p = 0
        while p < np:
            var best_d = Float32(0.0)
            var best_l = -1
            for l in range(nl):
                var d = cd.unsafe_load(row + l)
                if pq_probe_takes(d, l, prev_d, prev_l, best_d, best_l):
                    best_l = l
                    best_d = d
            if best_l < 0:
                break
            probes.unsafe_store(lq * np + p, Int32(best_l))
            pstart.unsafe_store(lq * np + p, Int32(pos))
            pos += Int(offsets.unsafe_load(best_l + 1)) - Int(offsets.unsafe_load(best_l))
            prev_l = best_l
            prev_d = best_d
            p += 1
        while p < np:
            probes.unsafe_store(lq * np + p, Int32(-1))
            pstart.unsafe_store(lq * np + p, Int32(pos))
            p += 1


def pq_score_kernel(
    q0: Int32, n_probes: Int32, queries: F32P, dim: Int32, centers: F32P, offsets: I32P, list_indices: I32P,
    codes: I32P, cb: F32P, pq_dim: Int32, pq_len: Int32, n_codes: Int32, use_lut: Int32, probes: I32P,
    pstart: I32P, stride: Int32, mask: I32P, cand: F32P,
):
    var b = Int(block_idx.x)
    var t = Int(thread_idx.x)
    var lq = b // Int(n_probes)
    var l = Int(probes.unsafe_load(b))
    var lut = stack_allocation[LUT_MAX, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    if l < 0:
        return
    var q_off = (Int(q0) + lq) * Int(dim)
    var pd = Int(pq_dim)
    var nc = Int(n_codes)
    if use_lut != 0:
        for e in range(t, pd * nc, STPB):
            lut[e] = pq_lut_entry(queries, q_off, centers, l, Int(dim), cb, e // nc, e % nc, Int(pq_len), nc)
    barrier()
    var start = Int(offsets.unsafe_load(l))
    var stop = Int(offsets.unsafe_load(l + 1))
    var base = lq * Int(stride) + Int(pstart.unsafe_load(b)) - start
    for slot in range(start + t, stop, STPB):
        var row = Int(list_indices.unsafe_load(slot))
        if ivf_row_removed(mask, row):
            continue
        var total = Float32(0.0)
        for j in range(pd):
            var code = Int(codes.unsafe_load(row * pd + j))
            var v: Float32
            if use_lut != 0:
                v = lut[j * nc + code]
            else:
                v = pq_lut_entry(queries, q_off, centers, l, Int(dim), cb, j, code, Int(pq_len), nc)
            total = ftz(total + v)
        cand.unsafe_store(base + slot, total)


def sq_score_kernel(
    q0: Int32, n_probes: Int32, queries: F32P, dim: Int32, centers: F32P, offsets: I32P, list_indices: I32P,
    codes: I32P, vmin: F32P, delta: F32P, probes: I32P, pstart: I32P, stride: Int32, mask: I32P, cand: F32P,
):
    var b = Int(block_idx.x)
    var t = Int(thread_idx.x)
    var lq = b // Int(n_probes)
    var l = Int(probes.unsafe_load(b))
    if l < 0:
        return
    var d = Int(dim)
    var q_off = (Int(q0) + lq) * d
    var start = Int(offsets.unsafe_load(l))
    var stop = Int(offsets.unsafe_load(l + 1))
    var base = lq * Int(stride) + Int(pstart.unsafe_load(b)) - start
    for slot in range(start + t, stop, STPB):
        var row = Int(list_indices.unsafe_load(slot))
        if ivf_row_removed(mask, row):
            continue
        cand.unsafe_store(base + slot, sq_candidate_dist(queries, q_off, centers, l, d, codes, row, vmin, delta))


def rq_rotate_kernel(
    count: Int32, q0: Int32, n_probes: Int32, queries: F32P, dim: Int32, centers: F32P, probes: I32P, D: Int32,
    seed: Int32, scale: Float32, ws: F32P, qn: F32P,
):
    """The query residual's rotation and squared norm per (query, probe), as
    `rq_search_cell` forms them at the top of each probe."""
    var b = _tid()
    if b < Int(count):
        var l = Int(probes.unsafe_load(b))
        if l < 0:
            return
        var lq = b // Int(n_probes)
        var dd = Int(D)
        rq_rotate(queries, (Int(q0) + lq) * Int(dim), centers, l * Int(dim), Int(dim), dd, Int(seed), scale, ws,
                  b * dd)
        var qn2 = Float32(0.0)
        for j in range(dd):
            var v = ws.unsafe_load(b * dd + j)
            qn2 = ftz(identical_mul_add(v, v, qn2))
        qn.unsafe_store(b, qn2)


def rq_score_kernel(
    n_probes: Int32, offsets: I32P, list_indices: I32P, codes: I32P, norms: F32P, ips: F32P, D: Int32,
    words: Int32, scale: Float32, probes: I32P, pstart: I32P, stride: Int32, mask: I32P, ws: F32P, qn: F32P,
    cand: F32P,
):
    var b = Int(block_idx.x)
    var t = Int(thread_idx.x)
    var lq = b // Int(n_probes)
    var l = Int(probes.unsafe_load(b))
    if l < 0:
        return
    var dd = Int(D)
    var w = Int(words)
    var qn2 = qn.unsafe_load(b)
    var start = Int(offsets.unsafe_load(l))
    var stop = Int(offsets.unsafe_load(l + 1))
    var base = lq * Int(stride) + Int(pstart.unsafe_load(b)) - start
    for slot in range(start + t, stop, STPB):
        var row = Int(list_indices.unsafe_load(slot))
        if ivf_row_removed(mask, row):
            continue
        cand.unsafe_store(base + slot, rq_candidate_est(ws, b * dd, qn2, codes, row, w, dd, norms, ips, scale))


def select_kernel(
    mc: Int32, q0: Int32, n_probes: Int32, offsets: I32P, list_indices: I32P, mask: I32P, probes: I32P,
    pstart: I32P, stride: Int32, cand: F32P, k: Int32, out_d: F32P, out_i: I32P, out_n: I32P,
):
    """The cell's top-k: `pq_insert` over the stored distances in the cell's
    order, masked rows skipped and not counted."""
    var lq = _tid()
    if lq < Int(mc):
        var qi = Int(q0) + lq
        var kk = Int(k)
        var np = Int(n_probes)
        var base = qi * kk
        for s in range(kk):
            out_d.unsafe_store(base + s, pq_inf())
            out_i.unsafe_store(base + s, Int32(-1))
        var n_cand = 0
        for p in range(np):
            var l = Int(probes.unsafe_load(lq * np + p))
            if l < 0:
                break
            var start = Int(offsets.unsafe_load(l))
            var stop = Int(offsets.unsafe_load(l + 1))
            var cbase = lq * Int(stride) + Int(pstart.unsafe_load(lq * np + p)) - start
            for slot in range(start, stop):
                var row = Int(list_indices.unsafe_load(slot))
                if ivf_row_removed(mask, row):
                    continue
                pq_insert(kk, base, cand.unsafe_load(cbase + slot), Int32(row), out_d, out_i)
                n_cand += 1
        out_n.unsafe_store(qi, Int32(n_cand))


def scan_stride(offsets: List[Int32], n_lists: Int, n_probes: Int) -> Int:
    """The longest candidate row a query can have: the n_probes longest
    lists, summed (at least 1)."""
    var lens = List[Int](capacity=n_lists)
    for l in range(n_lists):
        lens.append(Int(offsets[l + 1]) - Int(offsets[l]))
    var total = 0
    var taken = List[Bool](length=n_lists, fill=False)
    var np = n_probes if n_probes < n_lists else n_lists
    for _ in range(np):
        var best = -1
        for l in range(n_lists):
            if not taken[l] and (best < 0 or lens[l] > lens[best]):
                best = l
        taken[best] = True
        total += lens[best]
    return total if total > 0 else 1


def scan_chunk(m: Int, stride: Int) -> Int:
    var c = CAND_BUDGET // stride
    if c < 1:
        c = 1
    return c if c < m else m


# KIND: 0 = IVF-PQ, 1 = IVF-SQ, 2 = IVF-RaBitQ
def ivf_scan_search[KIND: Int](
    ctx: DeviceContext, dq: DeviceBuffer[DType.float32], dc: DeviceBuffer[DType.float32],
    doff: DeviceBuffer[DType.int32], dli: DeviceBuffer[DType.int32], dcodes: DeviceBuffer[DType.int32],
    dmask: DeviceBuffer[DType.int32], fa: DeviceBuffer[DType.float32], fb: DeviceBuffer[DType.float32],
    offsets: List[Int32], n_lists: Int, dim: Int, m: Int, k: Int, n_probes: Int,
    pq_dim: Int, pq_len: Int, n_codes: Int, D: Int, words: Int, seed: Int, scale: Float32,
    dd: DeviceBuffer[DType.float32], di: DeviceBuffer[DType.int32], dn: DeviceBuffer[DType.int32],
) raises:
    """Enqueue the whole search (no sync). fa/fb: PQ codebooks (fb unused);
    SQ vmin, delta; RaBitQ norms, ips."""
    var np = n_probes if n_probes < n_lists else n_lists
    var stride = scan_stride(offsets, n_lists, np)
    var mc = scan_chunk(m, stride)
    var dcd = ctx.enqueue_create_buffer[DType.float32](mc * n_lists)
    var dprobes = ctx.enqueue_create_buffer[DType.int32](mc * np)
    var dpstart = ctx.enqueue_create_buffer[DType.int32](mc * np)
    var dcand = ctx.enqueue_create_buffer[DType.float32](mc * stride)
    var dws = ctx.enqueue_create_buffer[DType.float32]((mc * np * D) if KIND == 2 else 1)
    var dqn = ctx.enqueue_create_buffer[DType.float32]((mc * np) if KIND == 2 else 1)
    var use_lut = 1 if pq_dim * n_codes <= LUT_MAX else 0
    var q0 = 0
    while q0 < m:
        var c = mc if m - q0 > mc else m - q0
        ctx.enqueue_function[coarse_kernel](
            Int32(c * n_lists), Int32(q0), dq.unsafe_ptr(), Int32(dim), dc.unsafe_ptr(), Int32(n_lists),
            dcd.unsafe_ptr(), grid_dim=_grid(c * n_lists), block_dim=TPB,
        )
        ctx.enqueue_function[probe_kernel](
            Int32(c), dcd.unsafe_ptr(), Int32(n_lists), Int32(np), doff.unsafe_ptr(), dprobes.unsafe_ptr(),
            dpstart.unsafe_ptr(), grid_dim=_grid(c), block_dim=TPB,
        )
        comptime if KIND == 0:
            ctx.enqueue_function[pq_score_kernel](
                Int32(q0), Int32(np), dq.unsafe_ptr(), Int32(dim), dc.unsafe_ptr(), doff.unsafe_ptr(),
                dli.unsafe_ptr(), dcodes.unsafe_ptr(), fa.unsafe_ptr(), Int32(pq_dim), Int32(pq_len),
                Int32(n_codes), Int32(use_lut), dprobes.unsafe_ptr(), dpstart.unsafe_ptr(), Int32(stride),
                dmask.unsafe_ptr(), dcand.unsafe_ptr(), grid_dim=c * np, block_dim=STPB,
            )
        elif KIND == 1:
            ctx.enqueue_function[sq_score_kernel](
                Int32(q0), Int32(np), dq.unsafe_ptr(), Int32(dim), dc.unsafe_ptr(), doff.unsafe_ptr(),
                dli.unsafe_ptr(), dcodes.unsafe_ptr(), fa.unsafe_ptr(), fb.unsafe_ptr(), dprobes.unsafe_ptr(),
                dpstart.unsafe_ptr(), Int32(stride), dmask.unsafe_ptr(), dcand.unsafe_ptr(),
                grid_dim=c * np, block_dim=STPB,
            )
        else:
            ctx.enqueue_function[rq_rotate_kernel](
                Int32(c * np), Int32(q0), Int32(np), dq.unsafe_ptr(), Int32(dim), dc.unsafe_ptr(),
                dprobes.unsafe_ptr(), Int32(D), Int32(seed), scale, dws.unsafe_ptr(), dqn.unsafe_ptr(),
                grid_dim=_grid(c * np), block_dim=TPB,
            )
            ctx.enqueue_function[rq_score_kernel](
                Int32(np), doff.unsafe_ptr(), dli.unsafe_ptr(), dcodes.unsafe_ptr(), fa.unsafe_ptr(),
                fb.unsafe_ptr(), Int32(D), Int32(words), scale, dprobes.unsafe_ptr(), dpstart.unsafe_ptr(),
                Int32(stride), dmask.unsafe_ptr(), dws.unsafe_ptr(), dqn.unsafe_ptr(), dcand.unsafe_ptr(),
                grid_dim=c * np, block_dim=STPB,
            )
        ctx.enqueue_function[select_kernel](
            Int32(c), Int32(q0), Int32(np), doff.unsafe_ptr(), dli.unsafe_ptr(), dmask.unsafe_ptr(),
            dprobes.unsafe_ptr(), dpstart.unsafe_ptr(), Int32(stride), dcand.unsafe_ptr(), Int32(k),
            dd.unsafe_ptr(), di.unsafe_ptr(), dn.unsafe_ptr(), grid_dim=_grid(c), block_dim=TPB,
        )
        q0 += c
    ctx.synchronize()
    _ = dqn^
    _ = dws^
    _ = dcand^
    _ = dpstart^
    _ = dprobes^
    _ = dcd^
