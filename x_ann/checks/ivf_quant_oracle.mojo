# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""IVF-SQ, IVF-RaBitQ, refine and the sample filter: their seams restated as
plain host code, apart from `x_ann/ivf_sq_core.mojo`,
`x_ann/ivf_rabitq_core.mojo` and `x_ann/refine_core.mojo`:

  5830  SQ range: rows scanned ascending with a STRICT compare (a -0.0 / 0.0
        tie keeps the first row's sign)
  5831  SQ rounding: roundf restated as trunc + an exact half test
  5832  SQ decode: vmin + code * delta as ONE fused step
  5840  RaBitQ rotation: hashed signs, butterflies h = 1, 2, 4, ... in order
  5841  RaBitQ zero residual: factor 0, estimate = the query residual's
        squared norm (never 0 / 0)
  5842  RaBitQ estimate: (|r|^2 + |q|^2) - 2 |r| <o, q>, in that association
  5850  refine: padding (< 0) and repeated ids skipped before scoring
  5855  the sample filter: a removed row is skipped before it is scored

Probes and top-k are IVF-PQ's (5803, 5804; `or_before` is reused)."""

from std.math import trunc
from std.memory import bitcast
from checks.numerics import ftz, identical_div, identical_mul, identical_mul_add, identical_sqrt
from x_ann.checks.ivf_pq_oracle import or_before, or_dist


def oq_inf() -> Float32:
    return bitcast[DType.float32](UInt32(0x7F800000))


def oq_sq_range(r: List[Float32], n: Int, dim: Int, mut vmin: List[Float32], mut delta: List[Float32],
                last_wins: Bool = False):
    vmin = List[Float32](length=dim, fill=Float32(0.0))
    delta = List[Float32](length=dim, fill=Float32(0.0))
    for c in range(dim):
        var lo = ftz(r[c])
        var hi = lo
        for i in range(1, n):
            var v = ftz(r[i * dim + c])
            if v < lo or (last_wins and v == lo):
                lo = v
            if v > hi:
                hi = v
        var rng = ftz(hi - lo)
        var margin = ftz(identical_mul(rng, Float32(0.05)))
        delta[c] = ftz(identical_div(ftz(rng + ftz(identical_mul(Float32(2.0), margin))), Float32(255.0))) if rng > Float32(0.0) else Float32(1.0)
        vmin[c] = ftz(lo - margin)


def oq_sq_code(v: Float32, vmin: Float32, delta: Float32, half_up_strict: Bool = False) -> Int:
    var x = ftz(identical_div(ftz(ftz(v) - vmin), delta))
    if not (x > Float32(0.0)):
        return 0
    var t = trunc(x)
    var frac = ftz(x - t)
    var up = frac > Float32(0.5) if half_up_strict else frac >= Float32(0.5)
    var code = t + Float32(1.0) if up else t
    return 255 if code > Float32(255.0) else Int(code)


def oq_sq_search(
    centers: List[Float32], offsets: List[Int32], list_indices: List[Int32], vmin: List[Float32],
    delta: List[Float32], codes: List[Int32], mask: List[Int32], n_lists: Int, dim: Int,
    queries: List[Float32], m: Int, k: Int, n_probes: Int,
    mut out_d: List[Float32], mut out_i: List[Int32], mut out_n: List[Int32],
):
    out_d = List[Float32](length=m * k, fill=oq_inf())
    out_i = List[Int32](length=m * k, fill=Int32(-1))
    out_n = List[Int32](length=m, fill=Int32(0))
    for q in range(m):
        var order = oq_probes(queries, q, centers, n_lists, dim, n_probes)
        var bd = List[Float32]()
        var bi = List[Int]()
        for p in range(len(order)):
            var l = order[p]
            for s in range(Int(offsets[l]), Int(offsets[l + 1])):
                var row = Int(list_indices[s])
                if mask[row] == 0:
                    continue
                var acc = Float32(0.0)
                for c in range(dim):
                    var qr = ftz(ftz(queries[q * dim + c]) - ftz(centers[l * dim + c]))
                    var dec = ftz(identical_mul_add(Float32(Int(codes[row * dim + c])), delta[c], vmin[c]))
                    var diff = ftz(qr - dec)
                    acc = ftz(identical_mul_add(diff, diff, acc))
                bd.append(acc)
                bi.append(row)
                out_n[q] += 1
        oq_topk(bd, bi, q, k, out_d, out_i)


def oq_probes(queries: List[Float32], q: Int, centers: List[Float32], n_lists: Int, dim: Int, n_probes: Int) -> List[Int]:
    var order = List[Int]()
    var taken = List[Bool](length=n_lists, fill=False)
    for _ in range(n_probes):
        var best = -1
        var bd = Float32(0.0)
        for l in range(n_lists):
            if taken[l]:
                continue
            var d = or_dist(queries, q * dim, centers, l * dim, dim)
            if best < 0 or or_before(d, l, bd, best):
                best = l
                bd = d
        taken[best] = True
        order.append(best)
    return order^


def oq_topk(bd: List[Float32], bi: List[Int], q: Int, k: Int, mut out_d: List[Float32], mut out_i: List[Int32]):
    var used = List[Bool](length=len(bd), fill=False)
    for s in range(k):
        var best = -1
        for t in range(len(bd)):
            if used[t]:
                continue
            if best < 0 or or_before(bd[t], bi[t], bd[best], bi[best]):
                best = t
        if best < 0:
            break
        used[best] = True
        out_d[q * k + s] = bd[best]
        out_i[q * k + s] = Int32(bi[best])


def oq_sign(seed: Int, j: Int) -> Bool:
    var h = UInt64(seed) * UInt64(0x9E3779B97F4A7C15) + UInt64(j) * UInt64(0xBF58476D1CE4E5B9)
    h = h ^ (h >> 31)
    h = h * UInt64(0x94D049BB133111EB)
    h = h ^ (h >> 29)
    return (h & UInt64(1)) == UInt64(1)


def oq_rotate(v: List[Float32], vo: Int, c: List[Float32], co: Int, dim: Int, D: Int, seed: Int,
              scale: Float32) -> List[Float32]:
    """5840: signs, then butterflies h = 1, 2, ... ascending, then the scale."""
    var w = List[Float32](length=D, fill=Float32(0.0))
    for j in range(dim):
        var t = ftz(ftz(v[vo + j]) - ftz(c[co + j]))
        w[j] = -t if oq_sign(seed, j) else t
    var h = 1
    while h < D:
        var i = 0
        while i < D:
            for j in range(i, i + h):
                var a = w[j]
                var b = w[j + h]
                w[j] = ftz(a + b)
                w[j + h] = ftz(a - b)
            i += 2 * h
        h *= 2
    for j in range(D):
        w[j] = ftz(identical_mul(w[j], scale))
    return w^


def oq_rq_encode(x: List[Float32], n: Int, dim: Int, centers: List[Float32], labels: List[Int32], seed: Int,
                 mut codes: List[Int32], mut norms: List[Float32], mut ips: List[Float32]):
    var D = 1
    while D < dim:
        D *= 2
    var words = (D + 31) // 32
    var scale = ftz(identical_div(Float32(1.0), ftz(identical_sqrt(Float32(D)))))
    codes = List[Int32](length=n * words, fill=Int32(0))
    norms = List[Float32](length=n, fill=Float32(0.0))
    ips = List[Float32](length=n, fill=Float32(0.0))
    for i in range(n):
        var w = oq_rotate(x, i * dim, centers, Int(labels[i]) * dim, dim, D, seed, scale)
        var sq = Float32(0.0)
        var sabs = Float32(0.0)
        for j in range(D):
            sq = ftz(identical_mul_add(w[j], w[j], sq))
            sabs = ftz(sabs + abs(w[j]))
            if w[j] > Float32(0.0):
                codes[i * words + j // 32] = codes[i * words + j // 32] | (Int32(1) << Int32(j % 32))
        var norm = ftz(identical_sqrt(sq))
        norms[i] = norm
        ips[i] = ftz(identical_div(ftz(identical_mul(sabs, scale)), norm)) if norm > Float32(0.0) else Float32(0.0)


def oq_rq_search(
    centers: List[Float32], offsets: List[Int32], list_indices: List[Int32], codes: List[Int32],
    norms: List[Float32], ips: List[Float32], mask: List[Int32], n_lists: Int, dim: Int, seed: Int,
    queries: List[Float32], m: Int, k: Int, n_probes: Int,
    mut out_d: List[Float32], mut out_i: List[Int32], mut out_n: List[Int32],
):
    var D = 1
    while D < dim:
        D *= 2
    var words = (D + 31) // 32
    var scale = ftz(identical_div(Float32(1.0), ftz(identical_sqrt(Float32(D)))))
    out_d = List[Float32](length=m * k, fill=oq_inf())
    out_i = List[Int32](length=m * k, fill=Int32(-1))
    out_n = List[Int32](length=m, fill=Int32(0))
    for q in range(m):
        var order = oq_probes(queries, q, centers, n_lists, dim, n_probes)
        var bd = List[Float32]()
        var bi = List[Int]()
        for p in range(len(order)):
            var l = order[p]
            var w = oq_rotate(queries, q * dim, centers, l * dim, dim, D, seed, scale)
            var qn2 = Float32(0.0)
            for j in range(D):
                qn2 = ftz(identical_mul_add(w[j], w[j], qn2))
            for s in range(Int(offsets[l]), Int(offsets[l + 1])):
                var row = Int(list_indices[s])
                if mask[row] == 0:
                    continue
                var est = qn2
                if ips[row] > Float32(0.0):
                    var dot = Float32(0.0)
                    for j in range(D):
                        var bit = (codes[row * words + j // 32] >> Int32(j % 32)) & Int32(1)
                        dot = ftz(dot + (w[j] if bit != 0 else -w[j]))
                    var xq = ftz(identical_div(ftz(identical_mul(dot, scale)), ips[row]))
                    var nn = ftz(identical_mul(norms[row], norms[row]))
                    est = ftz(ftz(nn + qn2) - ftz(identical_mul(Float32(2.0), ftz(identical_mul(norms[row], xq)))))
                bd.append(est)
                bi.append(row)
                out_n[q] += 1
        oq_topk(bd, bi, q, k, out_d, out_i)


def oq_refine(x: List[Float32], n: Int, d: Int, queries: List[Float32], m: Int, cand: List[Int32], k0: Int,
              k: Int, mut out_d: List[Float32], mut out_i: List[Int32], keep_repeats: Bool = False):
    out_d = List[Float32](length=m * k, fill=oq_inf())
    out_i = List[Int32](length=m * k, fill=Int32(-1))
    for q in range(m):
        var bd = List[Float32]()
        var bi = List[Int]()
        for t in range(k0):
            var v = Int(cand[q * k0 + t])
            if v < 0 or v >= n:
                continue
            var rep = False
            for u in range(t):
                if Int(cand[q * k0 + u]) == v:
                    rep = True
            if rep and not keep_repeats:
                continue
            bd.append(or_dist(queries, q * d, x, v * d, d))
            bi.append(v)
        oq_topk(bd, bi, q, k, out_d, out_i)
