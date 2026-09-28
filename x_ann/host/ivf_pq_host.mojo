# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""IVF-PQ on the host: `x_ann/ivf_pq_device.mojo`'s launches as loops over
the SAME cell functions (`x_ann/ivf_pq_core.mojo`), in the same launch order.
The coarse quantizer is the same Lloyd cells over whole rows."""

from std.sys.compile import is_defined

from cluster.host.kmeans_oracle import host_kmeans_fit
from cluster.impl.kmeans_params import INIT_KMEANS_PLUS_PLUS, METRIC_L2_EXPANDED
from ivf.host.ivf_host import host_ivf_build
from x_ann.refine_core import refine_cell
from checks.numerics import ftz, identical_div, identical_mul, identical_mul_add
from x_ann.host.ann_host_cells import ann_span, ann_task_count, ann_tasks, ftz_v, mul_add_v
from x_ann.ivf_rabitq_core import rq_encode_row, rq_pow2, rq_rotate, rq_scale
from x_ann.ivf_sq_core import sq_encode_cell, sq_range_cell
from x_ann.ivf_pq_core import (
    F32P, I32P, IvfPqIndex, pq_assign_cell, pq_coarse_dist, pq_inf, pq_insert, pq_labels_from_lists,
    pq_len_of, pq_lut_entry, pq_residual_cell, pq_validate,
)

comptime X_ANN_HOST_SABOTAGE = is_defined["MOJOLEARN_HOST_SABOTAGE"]()


def fp(mut l: List[Float32]) -> F32P:
    return l.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()


def ip(mut l: List[Int32]) -> I32P:
    return l.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()


def _coarse_host(
    x: List[Float32], n: Int, dim: Int, n_lists: Int, kmeans_n_iters: Int, seed: Int,
    mut centers: List[Float32], mut offsets: List[Int32], mut list_indices: List[Int32],
    mut labels: List[Int32],
) raises:
    """`ivf/host/ivf_host.mojo::host_ivf_build`, the host twin of the device
    driver's IVF-Flat coarse build."""
    var flat = host_ivf_build(x, n, dim, n_lists, kmeans_n_iters, METRIC_L2_EXPANDED, UInt64(seed))
    centers = flat.centers.copy()
    offsets = flat.offsets.copy()
    list_indices = List[Int32](capacity=n)
    for s in range(n):
        list_indices.append(Int32(Int(flat.list_indices[s])))
    labels = pq_labels_from_lists(offsets, list_indices, n_lists, n)


def _codebooks_host(
    r: List[Float32], n: Int, rot_dim: Int, pq_dim: Int, pq_len: Int, n_codes: Int, pq_iters: Int, seed: Int,
) raises -> List[Float32]:
    """`host_kmeans_fit` per subspace, the twin of the device `kmeans_fit`."""
    var codebooks = List[Float32](capacity=pq_dim * n_codes * pq_len)
    for j in range(pq_dim):
        var sub = List[Float32](capacity=n * pq_len)
        for i in range(n):
            for t in range(pq_len):
                sub.append(r[i * rot_dim + j * pq_len + t])
        var cb = List[Float32](length=n_codes * pq_len, fill=Float32(0.0))
        var lab = List[UInt32](length=n, fill=UInt32(0))
        _ = host_kmeans_fit(
            sub, n, pq_len, n_codes, cb, lab, List[Float32](), 0, pq_iters, Float64(1e-4), UInt64(seed), 1,
            INIT_KMEANS_PLUS_PLUS, METRIC_L2_EXPANDED, Float64(2.0),
        )
        for e in range(n_codes * pq_len):
            codebooks.append(cb[e])
    return codebooks^


#: Codes scored per vector step of the PQ encoding (x86-64-v3 is 8 lanes).
comptime PQ_ASSIGN_W = 8


def _assign_host(r: F32P, cb: F32P, n: Int, pq_dim: Int, rot_dim: Int, pq_len: Int, n_codes: Int, codes: I32P):
    """`pq_assign_cell` for every (row, subspace), on the caller's arrays.
    Each code's subspace distance is `pq_subdist`'s chain (t ascending,
    `ftz(ftz(a) - ftz(b))`, one fused step, flushed) computed in a vector
    lane beside seven other codes of the same subspace (the codebook read
    transposed, code innermost), then the cell's argmin over the codes
    ascending with the strict `<` (the lower code on a tie). A lane is one
    code's scalar chain, so every distance and every code is the cell's."""
    comptime W = PQ_ASSIGN_W
    if n_codes % W != 0:
        var tasks1 = ann_task_count(n, pq_dim * n_codes * pq_len)

        def scalar_rows(c: Int) {imm}:
            var span = ann_span(c, tasks1, n)
            for e in range(span[0] * pq_dim, span[1] * pq_dim):
                pq_assign_cell(e, r, cb, pq_dim, rot_dim, pq_len, n_codes, codes)

        ann_tasks(scalar_rows, tasks1)
        return
    # cbt[(j * pq_len + t) * n_codes + code] = ftz(cb[(j * n_codes + code) * pq_len + t])
    var cbt = List[Float32](length=pq_dim * pq_len * n_codes, fill=Float32(0.0))
    for j in range(pq_dim):
        for code in range(n_codes):
            for t in range(pq_len):
                cbt[(j * pq_len + t) * n_codes + code] = ftz(cb.unsafe_load((j * n_codes + code) * pq_len + t))
    var tp = fp(cbt)
    var tasks = ann_task_count(n, pq_dim * n_codes * pq_len)

    def rows(c: Int) {imm}:
        var span = ann_span(c, tasks, n)
        var dist = List[Float32](length=n_codes, fill=Float32(0.0))
        var dp = fp(dist)
        for i in range(span[0], span[1]):
            for j in range(pq_dim):
                var a_off = i * rot_dim + j * pq_len
                var blk = 0
                while blk < n_codes:
                    var acc = SIMD[DType.float32, W](0.0)
                    for t in range(pq_len):
                        var a = SIMD[DType.float32, W](ftz(r.unsafe_load(a_off + t)))
                        var b = tp.load[width=W]((j * pq_len + t) * n_codes + blk)
                        var diff = ftz_v[W](a - b)
                        acc = ftz_v[W](mul_add_v[W](diff, diff, acc))
                    dp.store(blk, acc)
                    blk += W
                var best = 0
                var bd = dp.unsafe_load(0)
                for code in range(1, n_codes):
                    var d = dp.unsafe_load(code)
                    if d < bd:
                        bd = d
                        best = code
                codes.unsafe_store(i * pq_dim + j, Int32(best))
        _ = dist^

    ann_tasks(rows, tasks)
    _ = cbt^


def ivf_pq_build_host(
    x_in: List[Float32], n: Int, dim: Int, n_lists: Int, kmeans_n_iters: Int, seed: Int,
    pq_dim: Int, pq_bits: Int, pq_iters: Int,
) raises -> IvfPqIndex:
    pq_validate(n, dim, n_lists, pq_dim, pq_bits, pq_iters)
    var pq_len = pq_len_of(dim, pq_dim)
    var rot_dim = pq_len * pq_dim
    var n_codes = 1 << pq_bits
    var x = x_in.copy()
    var centers = List[Float32]()
    var offsets = List[Int32]()
    var list_indices = List[Int32]()
    var labels = List[Int32]()
    _coarse_host(x, n, dim, n_lists, kmeans_n_iters, seed, centers, offsets, list_indices, labels)
    var r = List[Float32](length=n * rot_dim, fill=Float32(0.0))
    var xp = fp(x)
    var cp = fp(centers)
    var lp = ip(labels)
    var rp = fp(r)
    var row_tasks = ann_task_count(n, 4 * rot_dim)

    def residual_rows(c: Int) {imm}:
        var span = ann_span(c, row_tasks, n)
        for e in range(span[0] * rot_dim, span[1] * rot_dim):
            pq_residual_cell(e, xp, cp, lp, dim, rot_dim, rp)

    ann_tasks(residual_rows, row_tasks)
    var cb = _codebooks_host(r, n, rot_dim, pq_dim, pq_len, n_codes, pq_iters, seed)
    var codes = List[Int32](length=n * pq_dim, fill=Int32(0))
    _assign_host(rp, fp(cb), n, pq_dim, rot_dim, pq_len, n_codes, ip(codes))
    comptime if X_ANN_HOST_SABOTAGE:
        codes[0] = Int32((Int(codes[0]) + 1) % n_codes)
    _ = x^
    _ = r^
    _ = labels^
    return IvfPqIndex(n_lists, dim, n, pq_dim, pq_len, n_codes, centers^, offsets^, list_indices^, cb^, codes^)


def _probe_walk(coarse: F32P, n_lists: Int, n_probes: Int, probes: I32P) -> Int:
    """`pq_next_probe`'s walk over the query's coarse distances, computed
    once (`pq_coarse_dist`, the same cells) instead of once per probe: the
    same comparisons on the same values, so the same lists in the same
    order. Returns how many lists it found."""
    var prev_d = Float32(0.0)
    var prev_l = -1
    var count = 0
    for _ in range(n_probes):
        var best_d = Float32(0.0)
        var best_l = -1
        for l in range(n_lists):
            var d = coarse.unsafe_load(l)
            var after = prev_l < 0 or d > prev_d or (d == prev_d and l > prev_l)
            if after and (best_l < 0 or d < best_d or (d == best_d and l < best_l)):
                best_l = l
                best_d = d
        if best_l < 0:
            break
        probes.unsafe_store(count, Int32(best_l))
        count += 1
        prev_l = best_l
        prev_d = best_d
    return count


@always_inline
def _query_probes(
    queries: F32P, q_off: Int, centers: F32P, n_lists: Int, dim: Int, n_probes: Int, coarse: F32P,
    probes: I32P,
) -> Int:
    for l in range(n_lists):
        coarse.unsafe_store(l, pq_coarse_dist(queries, q_off, centers, l, dim))
    return _probe_walk(coarse, n_lists, n_probes, probes)


@always_inline
def _clear_topk(qi: Int, k: Int, out_d: F32P, out_i: I32P):
    for s in range(k):
        out_d.unsafe_store(qi * k + s, pq_inf())
        out_i.unsafe_store(qi * k + s, Int32(-1))


def ivf_pq_search_host(
    centers: F32P, offsets: I32P, list_indices: I32P, cb: F32P, codes: I32P, mask: I32P, n_lists: Int,
    dim: Int, pq_dim: Int, pq_bits: Int, queries: F32P, m: Int, k: Int, n_probes: Int,
    out_d: F32P, out_i: I32P, out_n: I32P,
):
    """`pq_search_cell` per query, reading and writing the caller's arrays.
    The coarse distances are computed once per query (`_probe_walk`), and a
    lookup-table entry `pq_lut_entry(j, code)` once per probed list, the first
    time a row reads it (a stamp per entry): the same function of the same
    inputs the cell evaluates per row, summed over j ascending into the same
    `ftz(total + entry)` chain, so every total and every top-k slot is the
    cell's."""
    var pq_len = pq_len_of(dim, pq_dim)
    var n_codes = 1 << pq_bits
    var n_lut = pq_dim * n_codes
    var tasks = ann_task_count(m, n_lists * dim + n_probes * n_lists)

    def task(c: Int) {imm}:
        var span = ann_span(c, tasks, m)
        var coarse = List[Float32](length=n_lists, fill=Float32(0.0))
        var probes = List[Int32](length=n_probes, fill=Int32(0))
        var lut = List[Float32](length=n_lut, fill=Float32(0.0))
        var stamp = List[Int32](length=n_lut, fill=Int32(-1))
        var cp = fp(coarse)
        var pp = ip(probes)
        var lp = fp(lut)
        var sp = ip(stamp)
        var gen = 0
        for qi in range(span[0], span[1]):
            var base = qi * k
            var q_off = qi * dim
            _clear_topk(qi, k, out_d, out_i)
            var n_cand = 0
            var np = _query_probes(queries, q_off, centers, n_lists, dim, n_probes, cp, pp)
            for p in range(np):
                var l = Int(pp.unsafe_load(p))
                gen += 1
                if gen > 0x7FFFFFF0:
                    for e in range(n_lut):
                        sp.unsafe_store(e, Int32(-1))
                    gen = 1
                var g = Int32(gen)
                for slot in range(Int(offsets.unsafe_load(l)), Int(offsets.unsafe_load(l + 1))):
                    var row = Int(list_indices.unsafe_load(slot))
                    if mask.unsafe_load(row) == 0:
                        continue
                    var total = Float32(0.0)
                    for j in range(pq_dim):
                        var code = Int(codes.unsafe_load(row * pq_dim + j))
                        var e = j * n_codes + code
                        if sp.unsafe_load(e) != g:
                            lp.unsafe_store(e, pq_lut_entry(queries, q_off, centers, l, dim, cb, j, code, pq_len, n_codes))
                            sp.unsafe_store(e, g)
                        total = ftz(total + lp.unsafe_load(e))
                    pq_insert(k, base, total, Int32(row), out_d, out_i)
                    n_cand += 1
            out_n.unsafe_store(qi, Int32(n_cand))
        _ = coarse^
        _ = probes^
        _ = lut^
        _ = stamp^

    ann_tasks(task, tasks)


def ivf_sq_build_host(
    x_in: List[Float32], n: Int, dim: Int, n_lists: Int, kmeans_n_iters: Int, seed: Int,
    mut centers: List[Float32], mut offsets: List[Int32], mut list_indices: List[Int32],
    mut vmin: List[Float32], mut delta: List[Float32], mut codes: List[Int32],
) raises:
    pq_validate(n, dim, n_lists, 1, 1, 1)
    var x = x_in.copy()
    var labels = List[Int32]()
    _coarse_host(x, n, dim, n_lists, kmeans_n_iters, seed, centers, offsets, list_indices, labels)
    var r = List[Float32](length=n * dim, fill=Float32(0.0))
    vmin = List[Float32](length=dim, fill=Float32(0.0))
    delta = List[Float32](length=dim, fill=Float32(0.0))
    codes = List[Int32](length=n * dim, fill=Int32(0))
    var xp = fp(x)
    var cp = fp(centers)
    var lp = ip(labels)
    var rp = fp(r)
    var vp = fp(vmin)
    var dp = fp(delta)
    var op = ip(codes)
    var row_tasks = ann_task_count(n, 4 * dim)

    def residual_rows(c: Int) {imm}:
        var span = ann_span(c, row_tasks, n)
        for e in range(span[0] * dim, span[1] * dim):
            pq_residual_cell(e, xp, cp, lp, dim, dim, rp)

    ann_tasks(residual_rows, row_tasks)
    var col_tasks = ann_task_count(dim, 2 * n)

    def range_cols(c: Int) {imm}:
        var span = ann_span(c, col_tasks, dim)
        for col in range(span[0], span[1]):
            sq_range_cell(col, rp, n, dim, vp, dp)

    ann_tasks(range_cols, col_tasks)

    def encode_rows(c: Int) {imm}:
        var span = ann_span(c, row_tasks, n)
        for e in range(span[0] * dim, span[1] * dim):
            sq_encode_cell(e, rp, dim, vp, dp, op)

    ann_tasks(encode_rows, row_tasks)
    _ = x^
    _ = r^
    _ = labels^


#: SQ codes are bytes: the decode table holds one entry per (column, code).
comptime SQ_CODES = 256


def ivf_sq_search_host(
    centers: F32P, offsets: I32P, list_indices: I32P, vmin: F32P, delta: F32P, codes: I32P, mask: I32P,
    n_lists: Int, dim: Int, queries: F32P, m: Int, k: Int, n_probes: Int,
    out_d: F32P, out_i: I32P, out_n: I32P,
):
    """`sq_search_cell` per query, reading and writing the caller's arrays.
    The coarse distances once per query (`_probe_walk`), the query residual
    `qr[c]` once per probed list, and the decoded value
    `ftz(identical_mul_add(code, delta[c], vmin[c]))` once per (column, code)
    per call: each the cell's own expression on the same inputs, then the
    cell's ascending fused square chain per row. A code outside [0, 255]
    (never written by the build) is decoded in place, as the cell does."""
    var dec = List[Float32](length=dim * SQ_CODES, fill=Float32(0.0))
    for c in range(dim):
        for code in range(SQ_CODES):
            dec[c * SQ_CODES + code] = ftz(identical_mul_add(
                Float32(code), delta.unsafe_load(c), vmin.unsafe_load(c)
            ))
    var decp = fp(dec)
    var tasks = ann_task_count(m, n_lists * dim + n_probes * n_lists)

    def task(t: Int) {imm}:
        var span = ann_span(t, tasks, m)
        var coarse = List[Float32](length=n_lists, fill=Float32(0.0))
        var probes = List[Int32](length=n_probes, fill=Int32(0))
        var qres = List[Float32](length=dim, fill=Float32(0.0))
        var cp = fp(coarse)
        var pp = ip(probes)
        var qr = fp(qres)
        for qi in range(span[0], span[1]):
            var base = qi * k
            var q_off = qi * dim
            _clear_topk(qi, k, out_d, out_i)
            var n_cand = 0
            var np = _query_probes(queries, q_off, centers, n_lists, dim, n_probes, cp, pp)
            for p in range(np):
                var l = Int(pp.unsafe_load(p))
                for c in range(dim):
                    qr.unsafe_store(c, ftz(ftz(queries.unsafe_load(q_off + c)) - ftz(centers.unsafe_load(l * dim + c))))
                for slot in range(Int(offsets.unsafe_load(l)), Int(offsets.unsafe_load(l + 1))):
                    var row = Int(list_indices.unsafe_load(slot))
                    if mask.unsafe_load(row) == 0:
                        continue
                    var acc = Float32(0.0)
                    for c in range(dim):
                        var code = Int(codes.unsafe_load(row * dim + c))
                        var dv: Float32
                        if code >= 0 and code < SQ_CODES:
                            dv = decp.unsafe_load(c * SQ_CODES + code)
                        else:
                            dv = ftz(identical_mul_add(Float32(code), delta.unsafe_load(c), vmin.unsafe_load(c)))
                        var diff = ftz(qr.unsafe_load(c) - dv)
                        acc = ftz(identical_mul_add(diff, diff, acc))
                    pq_insert(k, base, acc, Int32(row), out_d, out_i)
                    n_cand += 1
            out_n.unsafe_store(qi, Int32(n_cand))
        _ = coarse^
        _ = probes^
        _ = qres^

    ann_tasks(task, tasks)
    _ = dec^


def refine_host(
    x: F32P, n: Int, d: Int, queries: F32P, m: Int, cand: I32P, k0: Int, k: Int, out_d: F32P, out_i: I32P,
):
    """`refine_cell` per query, on the caller's arrays."""
    var tasks = ann_task_count(m, k0 * (d + k0))

    def task(t: Int) {imm}:
        var span = ann_span(t, tasks, m)
        for q in range(span[0], span[1]):
            refine_cell(q, x, n, d, queries, cand, k0, k, out_d, out_i)

    ann_tasks(task, tasks)


def ivf_rabitq_build_host(
    x_in: List[Float32], n: Int, dim: Int, n_lists: Int, kmeans_n_iters: Int, seed: Int,
    mut centers: List[Float32], mut offsets: List[Int32], mut list_indices: List[Int32],
    mut codes: List[Int32], mut norms: List[Float32], mut ips: List[Float32],
) raises:
    pq_validate(n, dim, n_lists, 1, 1, 1)
    var D = rq_pow2(dim)
    var words = (D + 31) // 32
    var scale = rq_scale(D)
    var x = x_in.copy()
    var labels = List[Int32]()
    _coarse_host(x, n, dim, n_lists, kmeans_n_iters, seed, centers, offsets, list_indices, labels)
    codes = List[Int32](length=n * words, fill=Int32(0))
    norms = List[Float32](length=n, fill=Float32(0.0))
    ips = List[Float32](length=n, fill=Float32(0.0))
    var xp = fp(x)
    var cp = fp(centers)
    var lp = ip(labels)
    var op = ip(codes)
    var np = fp(norms)
    var ipp = fp(ips)
    var tasks = ann_task_count(n, D * 8)

    def task(t: Int) {imm}:
        # `rq_encode_row`: the cell with one D-float workspace per task.
        var span = ann_span(t, tasks, n)
        var ws = List[Float32](length=D, fill=Float32(0.0))
        var wp = fp(ws)
        for i in range(span[0], span[1]):
            rq_encode_row(i, xp, cp, lp, dim, D, seed, scale, wp, 0, words, op, np, ipp)
        _ = ws^

    ann_tasks(task, tasks)
    _ = x^
    _ = labels^


def ivf_rabitq_search_host(
    centers: F32P, offsets: I32P, list_indices: I32P, codes: I32P, norms: F32P, ips: F32P, mask: I32P,
    n_lists: Int, dim: Int, seed: Int, queries: F32P, m: Int, k: Int, n_probes: Int,
    out_d: F32P, out_i: I32P, out_n: I32P,
):
    """`rq_search_cell` per query, on the caller's arrays; the coarse
    distances once per query (`_probe_walk`), the rotation and the scan the
    cell's statements."""
    var D = rq_pow2(dim)
    var words = (D + 31) // 32
    var scale = rq_scale(D)
    var tasks = ann_task_count(m, n_lists * dim + n_probes * n_lists)

    def task(t: Int) {imm}:
        var span = ann_span(t, tasks, m)
        var coarse = List[Float32](length=n_lists, fill=Float32(0.0))
        var probes = List[Int32](length=n_probes, fill=Int32(0))
        var ws = List[Float32](length=D, fill=Float32(0.0))
        var cp = fp(coarse)
        var pp = ip(probes)
        var wp = fp(ws)
        for qi in range(span[0], span[1]):
            var base = qi * k
            var q_off = qi * dim
            _clear_topk(qi, k, out_d, out_i)
            var n_cand = 0
            var np = _query_probes(queries, q_off, centers, n_lists, dim, n_probes, cp, pp)
            for p in range(np):
                var l = Int(pp.unsafe_load(p))
                rq_rotate(queries, q_off, centers, l * dim, dim, D, seed, scale, wp, 0)
                var qn2 = Float32(0.0)
                for j in range(D):
                    var v = wp.unsafe_load(j)
                    qn2 = ftz(identical_mul_add(v, v, qn2))
                for slot in range(Int(offsets.unsafe_load(l)), Int(offsets.unsafe_load(l + 1))):
                    var row = Int(list_indices.unsafe_load(slot))
                    if mask.unsafe_load(row) == 0:
                        continue
                    var est: Float32
                    var ipv = ips.unsafe_load(row)
                    var norm = norms.unsafe_load(row)
                    if ipv > Float32(0.0):
                        var dot = Float32(0.0)
                        for j in range(D):
                            var v = wp.unsafe_load(j)
                            var bit = (codes.unsafe_load(row * words + j // 32) >> Int32(j % 32)) & Int32(1)
                            dot = ftz(dot + (v if bit != 0 else -v))
                        var xq = ftz(identical_div(ftz(identical_mul(dot, scale)), ipv))
                        var nn = ftz(identical_mul(norm, norm))
                        est = ftz(ftz(nn + qn2) - ftz(identical_mul(Float32(2.0), ftz(identical_mul(norm, xq)))))
                    else:
                        est = qn2
                    pq_insert(k, base, est, Int32(row), out_d, out_i)
                    n_cand += 1
            out_n.unsafe_store(qi, Int32(n_cand))
        _ = coarse^
        _ = probes^
        _ = ws^

    ann_tasks(task, tasks)
