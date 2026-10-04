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
from std.memory import bitcast, stack_allocation
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, ftz, identical_mul_add
from std.sys.info import has_apple_gpu_accelerator
from x_ann.switches import ANN3_SCAN_SELECT
from x_ann.ivf_pq_core import (
    F32P, I32P, ivf_row_removed, pq_better, pq_coarse_dist, pq_inf, pq_insert, pq_lut_entry, pq_probe_takes,
)
from x_ann.ivf_sq_core import sq_candidate_dist
from x_ann.ivf_rabitq_core import rq_candidate_est, rq_est_tail, rq_rotate
from x_ann.tsne_core import ts_ftz_nonneg
from x_ann.stage_timer import AnnStages

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


@always_inline
def _probe_walk_seq_kern(lq: Int, cd: F32P, nl: Int, np: Int, offsets: I32P, probes: I32P, pstart: I32P):
    """`pq_next_probe`'s walk for query lq over its stored coarse distances,
    one thread (the cell's comparisons in the cell's order)."""
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


def probe_kernel(mc: Int32, cd: F32P, n_lists: Int32, n_probes: Int32, offsets: I32P, probes: I32P, pstart: I32P):
    """`pq_next_probe`'s walk over the stored coarse distances; an unused
    probe is -1. pstart = the probe's first position in the query's
    candidate row (probes in walk order, slots in list order)."""
    var lq = _tid()
    if lq < Int(mc):
        _probe_walk_seq_kern(lq, cd, Int(n_lists), Int(n_probes), offsets, probes, pstart)


@always_inline
def _nan_bits(v: Float32) -> Bool:
    """NaN by its bits (a float compare may be folded under fast math)."""
    var b = bitcast[DType.uint32](v)
    return (b & UInt32(0x7F800000)) == UInt32(0x7F800000) and (b & UInt32(0x007FFFFF)) != UInt32(0)


#: threads per query in the threadgroup probe walk (a power of two)
comptime PTPB = 256


def probe_group_kernel(cd: F32P, n_lists: Int32, n_probes: Int32, offsets: I32P, probes: I32P, pstart: I32P):
    """`probe_kernel` with one threadgroup per query (lane ann-apple2). Each
    probe is the minimum under the TOTAL order (distance, list id) of the
    lists after the previous probe: each thread takes the minimum of its
    lists (l = t, t + PTPB, ...) through `pq_probe_takes`, then a tree over
    the threads keeps the lesser pair. With no NaN distance the order is
    total, so the minimum is the one element the one-thread walk finds, and
    its own words are stored: the same bits. A NaN anywhere in the query's
    row sends the query to the one-thread walk itself (`_probe_walk_seq_kern`),
    since NaN makes the walk's result depend on list order."""
    var lq = Int(block_idx.x)
    var t = Int(thread_idx.x)
    var nl = Int(n_lists)
    var np = Int(n_probes)
    var row = lq * nl
    var sd = stack_allocation[PTPB, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var sl = stack_allocation[PTPB, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var has_nan = False
    for l in range(t, nl, PTPB):
        var d = cd.unsafe_load(row + l)
        if _nan_bits(d):
            has_nan = True
    sl[t] = Int32(1) if has_nan else Int32(0)
    barrier()
    var s = PTPB // 2
    while s > 0:
        if t < s:
            sl[t] = sl[t] | sl[t + s]
        barrier()
        s //= 2
    var any_nan = sl[0] != 0
    barrier()
    if any_nan:
        if t == 0:
            _probe_walk_seq_kern(lq, cd, nl, np, offsets, probes, pstart)
        return
    var prev_d = Float32(0.0)
    var prev_l = -1
    var pos = 0
    var p = 0
    while p < np:
        var best_d = Float32(0.0)
        var best_l = -1
        for l in range(t, nl, PTPB):
            var d = cd.unsafe_load(row + l)
            if pq_probe_takes(d, l, prev_d, prev_l, best_d, best_l):
                best_l = l
                best_d = d
        sd[t] = best_d
        sl[t] = Int32(best_l)
        barrier()
        s = PTPB // 2
        while s > 0:
            if t < s:
                var od = sd[t + s]
                var ol = Int(sl[t + s])
                # the cell's own comparison (the other entry is after the
                # previous probe by construction), so a change to it reaches
                # this path too (sabotage 5804)
                if ol >= 0 and pq_probe_takes(od, ol, prev_d, prev_l, sd[t], Int(sl[t])):
                    sd[t] = od
                    sl[t] = Int32(ol)
            barrier()
            s //= 2
        best_d = sd[0]
        best_l = Int(sl[0])
        barrier()
        if best_l < 0:
            break
        if t == 0:
            probes.unsafe_store(lq * np + p, Int32(best_l))
            pstart.unsafe_store(lq * np + p, Int32(pos))
        pos += Int(offsets.unsafe_load(best_l + 1)) - Int(offsets.unsafe_load(best_l))
        prev_l = best_l
        prev_d = best_d
        p += 1
    if t == 0:
        while p < np:
            probes.unsafe_store(lq * np + p, Int32(-1))
            pstart.unsafe_store(lq * np + p, Int32(pos))
            p += 1


def gather_i32_kernel(count: Int32, width: Int32, src: I32P, list_indices: I32P, dst: I32P):
    """dst[slot, c] = src[list_indices[slot], c] (lane ann-apple2): the
    per-row arrays laid out in list order, so a score threadgroup reads its
    list's rows contiguously. Plain copies."""
    var e = _tid()
    if e < Int(count):
        var w = Int(width)
        var slot = e // w
        dst.unsafe_store(e, src.unsafe_load(Int(list_indices.unsafe_load(slot)) * w + e % w))


def gather_f32_kernel(count: Int32, src: F32P, list_indices: I32P, dst: F32P):
    var e = _tid()
    if e < Int(count):
        dst.unsafe_store(e, src.unsafe_load(Int(list_indices.unsafe_load(e))))


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
        if ivf_row_removed(mask, slot):
            continue
        var total = Float32(0.0)
        for j in range(pd):
            var code = Int(codes.unsafe_load(slot * pd + j))
            var v: Float32
            if use_lut != 0:
                v = lut[j * nc + code]
            else:
                v = pq_lut_entry(queries, q_off, centers, l, Int(dim), cb, j, code, Int(pq_len), nc)
            # lane ann-apple2: entries are sums of squares from +0, so the
            # running sum is +0, positive or NaN: `ts_ftz_nonneg` (same word)
            total = ts_ftz_nonneg(total + v)
        cand.unsafe_store(base + slot, total)


#: widest row the staged SQ / RaBitQ score keeps in threadgroup memory
comptime SCORE_DIM_MAX = 512


def sq_score_kernel(
    q0: Int32, n_probes: Int32, queries: F32P, dim: Int32, centers: F32P, offsets: I32P, list_indices: I32P,
    codes: I32P, vmin: F32P, delta: F32P, probes: I32P, pstart: I32P, stride: Int32, mask: I32P, cand: F32P,
):
    """`sq_candidate_dist` per candidate. Lane ann-apple2: when dim <=
    SCORE_DIM_MAX the query residual `ftz(ftz(q) - ftz(center))`, delta and
    vmin are formed once per threadgroup into threadgroup memory (the same
    expressions on the same words), and each candidate runs the cell's decode,
    difference and fused square sum over them (the sum of squares from +0 is
    flushed by `ts_ftz_nonneg`, the same word): the same distance."""
    var b = Int(block_idx.x)
    var t = Int(thread_idx.x)
    var lq = b // Int(n_probes)
    var l = Int(probes.unsafe_load(b))
    var qr = stack_allocation[SCORE_DIM_MAX, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var dl = stack_allocation[SCORE_DIM_MAX, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var vm = stack_allocation[SCORE_DIM_MAX, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    if l < 0:
        return
    var d = Int(dim)
    var q_off = (Int(q0) + lq) * d
    var staged = d <= SCORE_DIM_MAX
    if staged:
        for c in range(t, d, STPB):
            qr[c] = ftz(ftz(queries.unsafe_load(q_off + c)) - ftz(centers.unsafe_load(l * d + c)))
            dl[c] = delta.unsafe_load(c)
            vm[c] = vmin.unsafe_load(c)
    barrier()
    var start = Int(offsets.unsafe_load(l))
    var stop = Int(offsets.unsafe_load(l + 1))
    var base = lq * Int(stride) + Int(pstart.unsafe_load(b)) - start
    for slot in range(start + t, stop, STPB):
        if ivf_row_removed(mask, slot):
            continue
        if staged:
            var acc = Float32(0.0)
            for c in range(d):
                var dec = ftz(identical_mul_add(Float32(Int(codes.unsafe_load(slot * d + c))), dl[c], vm[c]))
                var diff = ftz(qr[c] - dec)
                acc = ts_ftz_nonneg(identical_mul_add(diff, diff, acc))
            cand.unsafe_store(base + slot, acc)
        else:
            cand.unsafe_store(base + slot, sq_candidate_dist(queries, q_off, centers, l, d, codes, slot, vmin, delta))


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
    """`rq_candidate_est` per candidate. Lane ann-apple2: when D <=
    SCORE_DIM_MAX the rotated query residual is staged in threadgroup memory
    and each code word is loaded once for its 32 bits; the dot is the cell's
    fold (j ascending, `ftz(dot +- v)`) on the same words and the estimate
    is the cell's own tail (`rq_est_tail`): the same distance."""
    var b = Int(block_idx.x)
    var t = Int(thread_idx.x)
    var lq = b // Int(n_probes)
    var l = Int(probes.unsafe_load(b))
    var wv = stack_allocation[SCORE_DIM_MAX, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    if l < 0:
        return
    var dd = Int(D)
    var w = Int(words)
    var staged = dd <= SCORE_DIM_MAX
    if staged:
        for j in range(t, dd, STPB):
            wv[j] = ws.unsafe_load(b * dd + j)
    barrier()
    var qn2 = qn.unsafe_load(b)
    var start = Int(offsets.unsafe_load(l))
    var stop = Int(offsets.unsafe_load(l + 1))
    var base = lq * Int(stride) + Int(pstart.unsafe_load(b)) - start
    for slot in range(start + t, stop, STPB):
        if ivf_row_removed(mask, slot):
            continue
        if staged:
            var est = qn2
            var ip = ips.unsafe_load(slot)
            if ip > Float32(0.0):
                var dot = Float32(0.0)
                for wi in range(w):
                    var cw = codes.unsafe_load(slot * w + wi)
                    var j0 = wi * 32
                    var jn = dd - j0 if dd - j0 < 32 else 32
                    for u in range(jn):
                        var v = wv[j0 + u]
                        var bit = (cw >> Int32(u)) & Int32(1)
                        dot = ftz(dot + (v if bit != 0 else -v))
                est = rq_est_tail(dot, qn2, norms.unsafe_load(slot), ip, scale)
            cand.unsafe_store(base + slot, est)
        else:
            cand.unsafe_store(base + slot, rq_candidate_est(ws, b * dd, qn2, codes, slot, w, dd, norms, ips, scale))


@always_inline
def _select_seq_kern(
    lq: Int, qi: Int, np: Int, offsets: I32P, list_indices: I32P, mask: I32P, probes: I32P, pstart: I32P,
    stride: Int, cand: F32P, kk: Int, out_d: F32P, out_i: I32P, out_n: I32P,
):
    """The cell's top-k for query qi (chunk row lq), one thread: `pq_insert`
    over the stored distances in the cell's order, masked rows skipped and
    not counted."""
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
        var cbase = lq * stride + Int(pstart.unsafe_load(lq * np + p)) - start
        for slot in range(start, stop):
            var row = Int(list_indices.unsafe_load(slot))
            if ivf_row_removed(mask, row):
                continue
            pq_insert(kk, base, cand.unsafe_load(cbase + slot), Int32(row), out_d, out_i)
            n_cand += 1
    out_n.unsafe_store(qi, Int32(n_cand))


def select_kernel(
    mc: Int32, q0: Int32, n_probes: Int32, offsets: I32P, list_indices: I32P, mask: I32P, probes: I32P,
    pstart: I32P, stride: Int32, cand: F32P, k: Int32, out_d: F32P, out_i: I32P, out_n: I32P,
):
    """The cell's top-k: `pq_insert` over the stored distances in the cell's
    order, masked rows skipped and not counted."""
    var lq = _tid()
    if lq < Int(mc):
        _select_seq_kern(lq, Int(q0) + lq, Int(n_probes), offsets, list_indices, mask, probes, pstart, Int(stride),
                    cand, Int(k), out_d, out_i, out_n)


#: threads per query in the split top-k (lane ann-apple2)
comptime SEL_T = 128


def select_part_kernel(
    n_probes: Int32, offsets: I32P, list_indices: I32P, mask: I32P, probes: I32P, pstart: I32P, stride: Int32,
    cand: F32P, k: Int32, part_d: F32P, part_i: I32P, part_n: I32P,
):
    """Lane ann-apple2: one threadgroup per query, thread t keeps the top-k
    of the query's candidates at slots t, t + SEL_T, ... of every probe
    (`pq_insert` into its own k words of part_d / part_i) and counts them;
    part_n = the count, or -1 when a candidate distance is NaN."""
    var lq = Int(block_idx.x)
    var t = Int(thread_idx.x)
    var np = Int(n_probes)
    var kk = Int(k)
    var pb = (lq * SEL_T + t) * kk
    for s in range(kk):
        part_d.unsafe_store(pb + s, pq_inf())
        part_i.unsafe_store(pb + s, Int32(-1))
    var wd = pq_inf()
    var wi = Int32(-1)
    var n_cand = 0
    var has_nan = False
    for p in range(np):
        var l = Int(probes.unsafe_load(lq * np + p))
        if l < 0:
            break
        var start = Int(offsets.unsafe_load(l))
        var stop = Int(offsets.unsafe_load(l + 1))
        var cbase = lq * Int(stride) + Int(pstart.unsafe_load(lq * np + p)) - start
        for slot in range(start + t, stop, SEL_T):
            var row = Int(list_indices.unsafe_load(slot))
            if ivf_row_removed(mask, row):
                continue
            var v = cand.unsafe_load(cbase + slot)
            if _nan_bits(v):
                has_nan = True
            n_cand += 1
            # pq_insert's own first test, against a register copy of the last slot
            if pq_better(v, Int32(row), wd, wi):
                pq_insert(kk, pb, v, Int32(row), part_d, part_i)
                wd = part_d.unsafe_load(pb + kk - 1)
                wi = part_i.unsafe_load(pb + kk - 1)
    part_n.unsafe_store(lq * SEL_T + t, Int32(-1) if has_nan else Int32(n_cand))


def select_pair_kernel(count: Int32, pairs: Int32, k: Int32, ad: F32P, ai: I32P, bd: F32P, bi: I32P):
    """Lane ann-apple2: one level of a tree join of the partial top-k lists:
    thread (query lq, pair p) merges lists 2p and 2p + 1 of `a` (each
    ascending under `pq_better`, empty slots (+inf, -1) last) into list p of
    `b`, keeping the k first. Row ids are distinct, so `pq_better` is a
    strict total order on the entries and the k first of the union are the
    k least; the list after the last level is the k least of all
    candidates, the list `pq_insert` keeps. (A query with a NaN candidate
    does not use it: `select_merge_kernel` redoes that query sequentially.)"""
    var e = _tid()
    if e < Int(count):
        var np2 = Int(pairs)
        var kk = Int(k)
        var lq = e // np2
        var p = e % np2
        var xa = (lq * SEL_T + 2 * p) * kk
        var xb = xa + kk
        var o = (lq * SEL_T + p) * kk
        var u = 0
        var v = 0
        for s in range(kk):
            var ud = ad.unsafe_load(xa + u)
            var ui = ai.unsafe_load(xa + u)
            var vd = ad.unsafe_load(xb + v)
            var vi = ai.unsafe_load(xb + v)
            # take u's entry when it is first (an empty slot is last)
            if ui >= 0 and (vi < 0 or pq_better(ud, ui, vd, vi)):
                bd.unsafe_store(o + s, ud)
                bi.unsafe_store(o + s, ui)
                u += 1
            else:
                bd.unsafe_store(o + s, vd)
                bi.unsafe_store(o + s, vi)
                v += 1


def select_merge_kernel(
    mc: Int32, q0: Int32, n_probes: Int32, offsets: I32P, list_indices: I32P, mask: I32P, probes: I32P,
    pstart: I32P, stride: Int32, cand: F32P, k: Int32, part_d: F32P, part_i: I32P, part_n: I32P,
    out_d: F32P, out_i: I32P, out_n: I32P, lists: Int32,
):
    """Lane ann-apple2: one thread per query joins the SEL_T partial top-k
    lists with `pq_insert`. With no NaN candidate the order (distance, row
    id) is total (row ids are distinct), so the k least of the partial
    lists are the k least of all candidates, the list the cell's sequential
    insertion keeps: the same words (an empty slot stays (+inf, -1)). A NaN
    candidate makes the cell's result depend on insertion order, so that
    query runs the cell's own sequential insertion (`_select_seq_kern`). The
    count is the sum of the partial counts (integers)."""
    var lq = _tid()
    if lq < Int(mc):
        var qi = Int(q0) + lq
        var kk = Int(k)
        var any_nan = False
        var n_cand = 0
        for u in range(SEL_T):
            var c = Int(part_n.unsafe_load(lq * SEL_T + u))
            if c < 0:
                any_nan = True
            else:
                n_cand += c
        if any_nan:
            _select_seq_kern(lq, qi, Int(n_probes), offsets, list_indices, mask, probes, pstart, Int(stride), cand, kk,
                        out_d, out_i, out_n)
            return
        var base = qi * kk
        for s in range(kk):
            out_d.unsafe_store(base + s, pq_inf())
            out_i.unsafe_store(base + s, Int32(-1))
        for u in range(Int(lists)):
            var pb = (lq * SEL_T + u) * kk
            for s in range(kk):
                var id = part_i.unsafe_load(pb + s)
                if id < 0:
                    break
                pq_insert(kk, base, part_d.unsafe_load(pb + s), id, out_d, out_i)
        out_n.unsafe_store(qi, Int32(n_cand))


#: the widest k of the one-launch top-k (16 KB of threadgroup memory)
comptime SEL_KM = 16
comptime SCAN_SELECT_GROUP = (
    ANN3_SCAN_SELECT and GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator()
)


def select_group_kernel(
    q0: Int32, n_probes: Int32, offsets: I32P, list_indices: I32P, mask: I32P, probes: I32P, pstart: I32P,
    stride: Int32, cand: F32P, k: Int32, out_d: F32P, out_i: I32P, out_n: I32P,
):
    """FAST on Apple, OPT-IN (lane ann-apple3): `select_part_kernel`, the
    tree join and `select_merge_kernel` in ONE launch, for k <= SEL_KM. One
    threadgroup per query. Thread t keeps the top-k of the candidates at
    slots t, t + SEL_T, ... of every probe in registers (`pq_insert`'s
    statements on a register list); the lists go to threadgroup memory; each
    level of the join merges lists 2p and 2p + 1 into list p
    (`select_pair_kernel`'s statements: a thread reads its two lists, the
    threadgroup meets at a barrier, the thread writes the merged list);
    thread 0 writes list 0 and the summed count. Row ids are distinct, so
    `pq_better` is a strict total order and the k least are the entries the
    cell keeps. A NaN candidate sends the query to the cell's sequential
    insertion (`_select_seq_kern`), as `select_merge_kernel` does."""
    var lq = Int(block_idx.x)
    var t = Int(thread_idx.x)
    var np = Int(n_probes)
    var kk = Int(k)
    var sd = stack_allocation[SEL_T * SEL_KM, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    var si = stack_allocation[SEL_T * SEL_KM, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var sn = stack_allocation[SEL_T, Scalar[DType.int32], address_space=AddressSpace.SHARED]()
    var ld = InlineArray[Float32, SEL_KM](fill=pq_inf())
    var li = InlineArray[Int32, SEL_KM](fill=Int32(-1))
    var n_cand = 0
    var has_nan = False
    for p in range(np):
        var l = Int(probes.unsafe_load(lq * np + p))
        if l < 0:
            break
        var start = Int(offsets.unsafe_load(l))
        var stop = Int(offsets.unsafe_load(l + 1))
        var cbase = lq * Int(stride) + Int(pstart.unsafe_load(lq * np + p)) - start
        for slot in range(start + t, stop, SEL_T):
            var row = Int(list_indices.unsafe_load(slot))
            if ivf_row_removed(mask, row):
                continue
            var v = cand.unsafe_load(cbase + slot)
            if _nan_bits(v):
                has_nan = True
            n_cand += 1
            if pq_better(v, Int32(row), ld[kk - 1], li[kk - 1]):
                var s = kk - 1
                while s > 0 and pq_better(v, Int32(row), ld[s - 1], li[s - 1]):
                    ld[s] = ld[s - 1]
                    li[s] = li[s - 1]
                    s -= 1
                ld[s] = v
                li[s] = Int32(row)
    for s in range(kk):
        sd[t * SEL_KM + s] = ld[s]
        si[t * SEL_KM + s] = li[s]
    sn[t] = Int32(-1) if has_nan else Int32(n_cand)
    barrier()
    var lists = SEL_T
    while lists > 1:
        var pairs = lists // 2
        if t < pairs:
            var xa = 2 * t * SEL_KM
            var xb = xa + SEL_KM
            var u = 0
            var w = 0
            for s in range(kk):
                var ud = sd[xa + u]
                var ui = si[xa + u]
                var vd = sd[xb + w]
                var vi = si[xb + w]
                # take u's entry when it is first (an empty slot is last)
                if ui >= 0 and (vi < 0 or pq_better(ud, ui, vd, vi)):
                    ld[s] = ud
                    li[s] = ui
                    u += 1
                else:
                    ld[s] = vd
                    li[s] = vi
                    w += 1
        barrier()
        if t < pairs:
            for s in range(kk):
                sd[t * SEL_KM + s] = ld[s]
                si[t * SEL_KM + s] = li[s]
        barrier()
        lists = pairs
    if t == 0:
        var qi = Int(q0) + lq
        var any_nan = False
        var total = 0
        for u in range(SEL_T):
            var c = Int(sn[u])
            if c < 0:
                any_nan = True
            else:
                total += c
        if any_nan:
            _select_seq_kern(lq, qi, np, offsets, list_indices, mask, probes, pstart, Int(stride), cand, kk,
                        out_d, out_i, out_n)
        else:
            for s in range(kk):
                out_d.unsafe_store(qi * kk + s, sd[s])
                out_i.unsafe_store(qi * kk + s, si[s])
            out_n.unsafe_store(qi, Int32(total))


def scan_stride(offsets: List[Int32], n_lists: Int, n_probes: Int) -> Int:
    """The longest candidate row a query can have: the n_probes longest
    lists, summed (at least 1)."""
    var lens = List[Int](capacity=n_lists)
    for l in range(n_lists):  # small-loop(n_lists: one count per IVF list): sizes the candidate buffer, a shape not data
        lens.append(Int(offsets[l + 1]) - Int(offsets[l]))
    var total = 0
    var taken = List[Bool](length=n_lists, fill=False)
    var np = n_probes if n_probes < n_lists else n_lists
    for _ in range(np):
        var best = -1
        for l in range(n_lists):  # small-loop(n_lists: one count per IVF list): picks the longest lists for the buffer shape
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


def scan_gather_i32(
    ctx: DeviceContext, src: I32P, dli: I32P, n_slots: Int, width: Int,
) raises -> DeviceBuffer[DType.int32]:
    """A per-row int32 array of `width` words laid out in list order (the
    launch `ivf_scan_search` makes per search; a resident index makes it
    once, lane ann-apple3). Enqueued, not drained."""
    var out = ctx.enqueue_create_buffer[DType.int32](max(n_slots * width, 1))
    if n_slots > 0:
        ctx.enqueue_function[gather_i32_kernel](Int32(n_slots * width), Int32(width), src, dli, out.unsafe_ptr(),
                                                grid_dim=_grid(n_slots * width), block_dim=TPB)
    return out^


def scan_gather_f32(ctx: DeviceContext, src: F32P, dli: I32P, n_slots: Int) raises -> DeviceBuffer[DType.float32]:
    var out = ctx.enqueue_create_buffer[DType.float32](max(n_slots, 1))
    if n_slots > 0:
        ctx.enqueue_function[gather_f32_kernel](Int32(n_slots), src, dli, out.unsafe_ptr(),
                                                grid_dim=_grid(n_slots), block_dim=TPB)
    return out^


# KIND: 0 = IVF-PQ, 1 = IVF-SQ, 2 = IVF-RaBitQ
def ivf_scan_search[KIND: Int](
    ctx: DeviceContext, dq: F32P, dc: F32P, doff: I32P, dli: I32P, dcodes: I32P, dmask: I32P, fa: F32P, fb: F32P,
    offsets: List[Int32], n_lists: Int, dim: Int, m: Int, k: Int, n_probes: Int,
    pq_dim: Int, pq_len: Int, n_codes: Int, D: Int, words: Int, seed: Int, scale: Float32,
    dd: F32P, di: I32P, dn: I32P,
    have_pre: Bool, pre_codes: I32P, pre_a: F32P, pre_b: F32P, mask_pre: Bool, pre_mask: I32P,
) raises:
    """Run the whole search and synchronize. The pointers are device
    buffers'. fa/fb: PQ codebooks (fb unused); SQ vmin, delta; RaBitQ norms,
    ips.

    `have_pre` (lane ann-apple3, the resident index): `pre_codes` (and, for
    RaBitQ, `pre_a` / `pre_b`) are the list-order arrays `scan_gather_i32` /
    `scan_gather_f32` made from dcodes (fa, fb) when the index was prepared,
    so this search does not gather them again; `mask_pre` says `pre_mask` is
    the mask in list order too (the all-ones filter, the same words in any
    order). Without `have_pre` every array is gathered here, as before. The
    score kernels read the same words either way."""
    var np = n_probes if n_probes < n_lists else n_lists
    var stride = scan_stride(offsets, n_lists, np)
    var mc = scan_chunk(m, stride)
    var dcd = ctx.enqueue_create_buffer[DType.float32](mc * n_lists)
    var dprobes = ctx.enqueue_create_buffer[DType.int32](mc * np)
    var dpstart = ctx.enqueue_create_buffer[DType.int32](mc * np)
    var dcand = ctx.enqueue_create_buffer[DType.float32](mc * stride)
    var dws = ctx.enqueue_create_buffer[DType.float32]((mc * np * D) if KIND == 2 else 1)
    var dqn = ctx.enqueue_create_buffer[DType.float32]((mc * np) if KIND == 2 else 1)
    comptime SERIAL = is_defined["MOJOLEARN_ANN_SERIAL_SCAN"]()
    # lane ann-apple3, FAST on Apple, OPT-IN: the top-k of a chunk in one
    # launch when k fits (`select_group_kernel`); the partial lists then live
    # in threadgroup memory and these device buffers are one word
    var grouped = False
    comptime if SCAN_SELECT_GROUP:
        grouped = k <= SEL_KM
    var no_parts = SERIAL or grouped
    var dpd = ctx.enqueue_create_buffer[DType.float32](1 if no_parts else mc * SEL_T * k)
    var dpi = ctx.enqueue_create_buffer[DType.int32](1 if no_parts else mc * SEL_T * k)
    var dpn = ctx.enqueue_create_buffer[DType.int32](1 if no_parts else mc * SEL_T)
    var dpd2 = ctx.enqueue_create_buffer[DType.float32](1 if no_parts else mc * SEL_T * k)
    var dpi2 = ctx.enqueue_create_buffer[DType.int32](1 if no_parts else mc * SEL_T * k)
    var use_lut = 1 if pq_dim * n_codes <= LUT_MAX else 0
    var st = AnnStages("ivf_scan")
    # lane ann-apple2: codes, mask (and RaBitQ's norms and factors) gathered
    # into list order once per search; the score kernels index them by slot
    var n_slots = Int(offsets[n_lists])
    var width = pq_dim if KIND == 0 else (dim if KIND == 1 else words)
    var own_codes = not have_pre
    var own_mask = not (have_pre and mask_pre)
    var dpcodes = ctx.enqueue_create_buffer[DType.int32](max(n_slots * width, 1) if own_codes else 1)
    var dpmask = ctx.enqueue_create_buffer[DType.int32](max(n_slots, 1) if own_mask else 1)
    var dpa = ctx.enqueue_create_buffer[DType.float32](max(n_slots, 1) if (KIND == 2 and own_codes) else 1)
    var dpb = ctx.enqueue_create_buffer[DType.float32](max(n_slots, 1) if (KIND == 2 and own_codes) else 1)
    if n_slots > 0:
        if own_codes:
            ctx.enqueue_function[gather_i32_kernel](Int32(n_slots * width), Int32(width), dcodes, dli,
                                                    dpcodes.unsafe_ptr(), grid_dim=_grid(n_slots * width),
                                                    block_dim=TPB)
        if own_mask:
            ctx.enqueue_function[gather_i32_kernel](Int32(n_slots), Int32(1), dmask, dli, dpmask.unsafe_ptr(),
                                                    grid_dim=_grid(n_slots), block_dim=TPB)
        comptime if KIND == 2:
            if own_codes:
                ctx.enqueue_function[gather_f32_kernel](Int32(n_slots), fa, dli, dpa.unsafe_ptr(),
                                                        grid_dim=_grid(n_slots), block_dim=TPB)
                ctx.enqueue_function[gather_f32_kernel](Int32(n_slots), fb, dli, dpb.unsafe_ptr(),
                                                        grid_dim=_grid(n_slots), block_dim=TPB)
    # the list-order arrays the score kernels read: this search's or the
    # resident index's
    var gcodes = rebind[I32P](dpcodes.unsafe_ptr()) if own_codes else pre_codes
    var gmask = rebind[I32P](dpmask.unsafe_ptr()) if own_mask else pre_mask
    var ga = rebind[F32P](dpa.unsafe_ptr()) if own_codes else pre_a
    var gb = rebind[F32P](dpb.unsafe_ptr()) if own_codes else pre_b
    st.mark(ctx, "alloc")
    var q0 = 0
    while q0 < m:
        var c = mc if m - q0 > mc else m - q0
        ctx.enqueue_function[coarse_kernel](
            Int32(c * n_lists), Int32(q0), dq, Int32(dim), dc, Int32(n_lists),
            dcd.unsafe_ptr(), grid_dim=_grid(c * n_lists), block_dim=TPB,
        )
        st.mark(ctx, "coarse")
        comptime if SERIAL:
            ctx.enqueue_function[probe_kernel](
                Int32(c), dcd.unsafe_ptr(), Int32(n_lists), Int32(np), doff, dprobes.unsafe_ptr(),
                dpstart.unsafe_ptr(), grid_dim=_grid(c), block_dim=TPB,
            )
        else:
            ctx.enqueue_function[probe_group_kernel](
                dcd.unsafe_ptr(), Int32(n_lists), Int32(np), doff, dprobes.unsafe_ptr(),
                dpstart.unsafe_ptr(), grid_dim=c, block_dim=PTPB,
            )
        st.mark(ctx, "probe")
        comptime if KIND == 0:
            ctx.enqueue_function[pq_score_kernel](
                Int32(q0), Int32(np), dq, Int32(dim), dc, doff,
                dli, gcodes, fa, Int32(pq_dim), Int32(pq_len),
                Int32(n_codes), Int32(use_lut), dprobes.unsafe_ptr(), dpstart.unsafe_ptr(), Int32(stride),
                gmask, dcand.unsafe_ptr(), grid_dim=c * np, block_dim=STPB,
            )
        elif KIND == 1:
            ctx.enqueue_function[sq_score_kernel](
                Int32(q0), Int32(np), dq, Int32(dim), dc, doff,
                dli, gcodes, fa, fb, dprobes.unsafe_ptr(),
                dpstart.unsafe_ptr(), Int32(stride), gmask, dcand.unsafe_ptr(),
                grid_dim=c * np, block_dim=STPB,
            )
        else:
            ctx.enqueue_function[rq_rotate_kernel](
                Int32(c * np), Int32(q0), Int32(np), dq, Int32(dim), dc,
                dprobes.unsafe_ptr(), Int32(D), Int32(seed), scale, dws.unsafe_ptr(), dqn.unsafe_ptr(),
                grid_dim=_grid(c * np), block_dim=TPB,
            )
            ctx.enqueue_function[rq_score_kernel](
                Int32(np), doff, dli, gcodes, ga,
                gb, Int32(D), Int32(words), scale, dprobes.unsafe_ptr(), dpstart.unsafe_ptr(),
                Int32(stride), gmask, dws.unsafe_ptr(), dqn.unsafe_ptr(), dcand.unsafe_ptr(),
                grid_dim=c * np, block_dim=STPB,
            )
        st.mark(ctx, "score")
        comptime if SERIAL:
            ctx.enqueue_function[select_kernel](
                Int32(c), Int32(q0), Int32(np), doff, dli, dmask,
                dprobes.unsafe_ptr(), dpstart.unsafe_ptr(), Int32(stride), dcand.unsafe_ptr(), Int32(k),
                dd, di, dn, grid_dim=_grid(c), block_dim=TPB,
            )
        else:
            comptime if SCAN_SELECT_GROUP:
                if grouped:
                    ctx.enqueue_function[select_group_kernel](
                        Int32(q0), Int32(np), doff, dli, dmask, dprobes.unsafe_ptr(), dpstart.unsafe_ptr(),
                        Int32(stride), dcand.unsafe_ptr(), Int32(k), dd, di, dn, grid_dim=c, block_dim=SEL_T,
                    )
            if not grouped:
                ctx.enqueue_function[select_part_kernel](
                    Int32(np), doff, dli, dmask, dprobes.unsafe_ptr(), dpstart.unsafe_ptr(), Int32(stride),
                    dcand.unsafe_ptr(), Int32(k), dpd.unsafe_ptr(), dpi.unsafe_ptr(), dpn.unsafe_ptr(),
                    grid_dim=c, block_dim=SEL_T,
                )
                # the tree join (lane ann-apple2): SEL_T lists -> 1, ping-pong
                var ad = rebind[F32P](dpd.unsafe_ptr())
                var ai = rebind[I32P](dpi.unsafe_ptr())
                var bd = rebind[F32P](dpd2.unsafe_ptr())
                var bi = rebind[I32P](dpi2.unsafe_ptr())
                var lists = SEL_T
                while lists > 1:
                    var pairs = lists // 2
                    ctx.enqueue_function[select_pair_kernel](
                        Int32(c * pairs), Int32(pairs), Int32(k), ad, ai, bd, bi, grid_dim=_grid(c * pairs), block_dim=TPB,
                    )
                    var td = ad
                    var ti = ai
                    ad = bd
                    ai = bi
                    bd = td
                    bi = ti
                    lists = pairs
                ctx.enqueue_function[select_merge_kernel](
                    Int32(c), Int32(q0), Int32(np), doff, dli, dmask,
                    dprobes.unsafe_ptr(), dpstart.unsafe_ptr(), Int32(stride), dcand.unsafe_ptr(), Int32(k),
                    ad, ai, dpn.unsafe_ptr(), dd, di, dn, Int32(1), grid_dim=_grid(c), block_dim=TPB,
                )
        st.mark(ctx, "select")
        q0 += c
    ctx.synchronize()
    _ = dpb^
    _ = dpa^
    _ = dpmask^
    _ = dpcodes^
    _ = dpi2^
    _ = dpd2^
    _ = dpn^
    _ = dpi^
    _ = dpd^
    _ = dqn^
    _ = dws^
    _ = dcand^
    _ = dpstart^
    _ = dprobes^
    _ = dcd^
