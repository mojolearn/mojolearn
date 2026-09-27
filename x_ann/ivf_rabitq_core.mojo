# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""IVF-RaBitQ: 1-bit RaBitQ codes of the IVF residuals, per-cell arithmetic
shared by the GPU and the CPU (lane/algos-ann, pass 1).

Reference: cuVS `cpp/src/neighbors/ivf_rabitq/` (gpu_index: rotate the
residual, keep sign bits, store the residual norm and the <x_bar, o> factor;
estimate the distance from the bit inner product), after Gao & Long,
"RaBitQ" (SIGMOD 2024).

THE FIXED-ORDER DESIGN
  * the random orthogonal rotation is a RANDOMIZED HADAMARD TRANSFORM:
    +-1 signs from an integer hash of (seed, coordinate), zero padding to the
    next power of two D, the Walsh-Hadamard butterflies in their fixed
    sequential order, one pinned scale by 1/sqrt(D). cuVS draws a dense
    random orthogonal matrix (a GEMM whose reduction order is the library's);
    a host QR would not be bitwise across BLAS builds.
  * every norm / inner product is an ascending sequential fold; the square
    root is `identical_sqrt`; a zero residual (norm 0) keeps factor 0 and its
    estimate is the query residual's squared norm, never 0/0.
  * the query residual is kept in float32 (cuVS quantizes it to 4 bits for
    the bit-ops fast path; that is a FAST-tier schedule, pass 2).
  * top-k under (estimated distance, id).
"""

from checks.numerics import ftz, identical_div, identical_mul, identical_mul_add, identical_sqrt
from x_ann.ivf_pq_core import F32P, I32P, pq_inf, pq_insert, pq_next_probe


@always_inline
def rq_sign(seed: Int, j: Int) -> Bool:
    var h = UInt64(seed) * UInt64(0x9E3779B97F4A7C15) + UInt64(j) * UInt64(0xBF58476D1CE4E5B9)
    h = h ^ (h >> 31)
    h = h * UInt64(0x94D049BB133111EB)
    h = h ^ (h >> 29)
    return (h & UInt64(1)) == UInt64(1)


def rq_scale(D: Int) -> Float32:
    """1 / sqrt(D), pinned."""
    return ftz(identical_div(Float32(1.0), ftz(identical_sqrt(Float32(D)))))


def rq_pow2(dim: Int) -> Int:
    var d = 1
    while d < dim:
        d *= 2
    return d


@always_inline
def rq_rotate(
    src: F32P, s_off: Int, center: F32P, c_off: Int, dim: Int, D: Int, seed: Int, scale: Float32,
    ws: F32P, w_off: Int,
):
    """ws[w_off : w_off + D] = H (signs * pad(src - center)) * scale."""
    for j in range(D):
        var v = Float32(0.0)
        if j < dim:
            v = ftz(ftz(src.unsafe_load(s_off + j)) - ftz(center.unsafe_load(c_off + j)))
            if rq_sign(seed, j):
                v = -v
        ws.unsafe_store(w_off + j, v)
    var h = 1
    while h < D:
        var i = 0
        while i < D:
            for j in range(i, i + h):
                var a = ws.unsafe_load(w_off + j)
                var b = ws.unsafe_load(w_off + j + h)
                ws.unsafe_store(w_off + j, ftz(a + b))
                ws.unsafe_store(w_off + j + h, ftz(a - b))
            i += 2 * h
        h *= 2
    for j in range(D):
        ws.unsafe_store(w_off + j, ftz(identical_mul(ws.unsafe_load(w_off + j), scale)))


@always_inline
def rq_encode_cell(
    i: Int, x: F32P, centers: F32P, labels: I32P, dim: Int, D: Int, seed: Int, scale: Float32,
    ws: F32P, words: Int, codes: I32P, norms: F32P, ips: F32P,
):
    """Row i: rotated residual, its norm, the sign bits (0 counts as
    negative) and <x_bar, o> = sum |r_j| * scale / norm."""
    var l = Int(labels.unsafe_load(i))
    rq_rotate(x, i * dim, centers, l * dim, dim, D, seed, scale, ws, i * D)
    var sq = Float32(0.0)
    var sabs = Float32(0.0)
    for w in range(words):
        codes.unsafe_store(i * words + w, Int32(0))
    for j in range(D):
        var v = ws.unsafe_load(i * D + j)
        sq = ftz(identical_mul_add(v, v, sq))
        sabs = ftz(sabs + abs(v))
        if v > Float32(0.0):
            var w = i * words + j // 32
            codes.unsafe_store(w, codes.unsafe_load(w) | (Int32(1) << Int32(j % 32)))
    var norm = ftz(identical_sqrt(sq))
    norms.unsafe_store(i, norm)
    if norm > Float32(0.0):
        ips.unsafe_store(i, ftz(identical_div(ftz(identical_mul(sabs, scale)), norm)))
    else:
        ips.unsafe_store(i, Float32(0.0))


@always_inline
def rq_search_cell(
    qi: Int, queries: F32P, dim: Int, centers: F32P, n_lists: Int, offsets: I32P,
    list_indices: I32P, codes: I32P, norms: F32P, ips: F32P, D: Int, words: Int, seed: Int,
    scale: Float32, k: Int, n_probes: Int, mask: I32P, ws: F32P, out_d: F32P, out_i: I32P, out_n: I32P,
):
    var base = qi * k
    var q_off = qi * dim
    for s in range(k):
        out_d.unsafe_store(base + s, pq_inf())
        out_i.unsafe_store(base + s, Int32(-1))
    var prev_d = Float32(0.0)
    var prev_l = -1
    var n_cand = 0
    for _ in range(n_probes):
        var best_d = Float32(0.0)
        var best_l = pq_next_probe(queries, q_off, centers, n_lists, dim, prev_d, prev_l, best_d)
        if best_l < 0:
            break
        prev_l = best_l
        prev_d = best_d
        rq_rotate(queries, q_off, centers, best_l * dim, dim, D, seed, scale, ws, qi * D)
        var qn2 = Float32(0.0)
        for j in range(D):
            var v = ws.unsafe_load(qi * D + j)
            qn2 = ftz(identical_mul_add(v, v, qn2))
        for slot in range(Int(offsets.unsafe_load(best_l)), Int(offsets.unsafe_load(best_l + 1))):
            var row = Int(list_indices.unsafe_load(slot))
            if mask.unsafe_load(row) == 0:
                continue
            var est: Float32
            var ip = ips.unsafe_load(row)
            var norm = norms.unsafe_load(row)
            if ip > Float32(0.0):
                var dot = Float32(0.0)
                for j in range(D):
                    var v = ws.unsafe_load(qi * D + j)
                    var bit = (codes.unsafe_load(row * words + j // 32) >> Int32(j % 32)) & Int32(1)
                    dot = ftz(dot + (v if bit != 0 else -v))
                var xq = ftz(identical_div(ftz(identical_mul(dot, scale)), ip))
                var nn = ftz(identical_mul(norm, norm))
                est = ftz(ftz(nn + qn2) - ftz(identical_mul(Float32(2.0), ftz(identical_mul(norm, xq)))))
            else:
                est = qn2
            pq_insert(k, base, est, Int32(row), out_d, out_i)
            n_cand += 1
    out_n.unsafe_store(qi, Int32(n_cand))
