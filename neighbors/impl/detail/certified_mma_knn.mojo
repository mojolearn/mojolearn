# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""IDENTICAL k-NN on Apple: candidates from the simdgroup-matrix arm,
answers from the pinned chain, CERTIFIED per query (lane/neighbors-apple,
2026-09-28).

THE ANSWER IS UNCHANGED. The IDENTICAL tiled arm returns, per query, the k
smallest composite keys (z, index), ascending, where z is the pinned cell:
the dot product as ONE chain over the features ascending from +0.0, each
step `_rt_step`, then `max(ftz(fma(-2, dot, ftz(ftz(qn) + ftz(yn)))), 0)`
and, for the root metric, `ftz(identical_sqrt(.))`. This path returns the
same keys:

  1. CANDIDATES. `fast_mma_knn` (the FAST arm, any bits) lists each
     query's KC nearest rows by its own approximate squared distance v,
     ascending by (v, index); v_KC is the last one. Every row NOT listed
     has v >= v_KC (the arm keeps the KC smallest (v, index) pairs of every
     slice and merges them, and rounding is monotone).
  2. PINNED VALUES of the KC candidates: the chain and epilogue above,
     statement for statement (`_rt_step`, the register tile's epilogue),
     so each candidate's z is the tiled arm's z for that cell.
  3. THE k SMALLEST (z, index) of the candidates, ascending; T = the
     largest pinned squared value p among them.
  4. CERTIFICATE. Let D be the exact squared distance and n = qn + yn (the
     exact squared norms). Both the approximate v and the pinned p are
     within E * n of D (E = CERT_E_PER_FEATURE * (d + 4) * 2^-24, a
     factor of about ten above the float32 bound (6d + 6) * 2^-24 of the
     two expanded forms together; flushes are covered by CERT_ABS).
     Since yn <= 2D + 2qn and D <= v + E n, a row outside the list has
         p >= v (1 - 2E') - 3 E' qn,   E' = E / (1 - 2E),
     which increases with v, so p >= LB = v_KC (1 - 2E') - 3 E' qn
     (evaluated with a downward margin). If LB > T (1 + 2^-19), every
     unlisted row has p > T with room to spare, so its z (a monotone,
     correctly rounded function of p) is STRICTLY above the z of each of
     the k chosen candidates: it cannot enter the answer, and the answer is
     the k chosen, bit for bit.
  5. A query whose certificate fails (ties, duplicates, a NaN, an index
     shorter than KC) is FLAGGED and answered by the tiled arm itself.

`-D MOJOLEARN_KNN_CERTIFIED_MMA_OFF` keeps the tiled arm for every query.
"""

from std.gpu import block_idx, thread_idx
from std.memory import bitcast
from checks.numerics import ftz, identical_mul_add, identical_sqrt
from neighbors.checks.pinned_distance_tile import _rt_step

comptime CERT_TPB = 128
comptime CERT_E_PER_FEATURE = Float32(64.0)
comptime CERT_ABS = Float32(1.0e-30)
comptime CERT_REL = Float32(1.0) / Float32(524288.0)  # 2^-19
comptime CERT_BIG = Float32(3.0e38)


@always_inline
def _key_worse(da: Float32, ia: UInt32, db: Float32, ib: UInt32) -> Bool:
    """(da, ia) orders after (db, ib)."""
    return da > db or (da == db and ia > ib)


def certified_rescore_kernel[KC: Int, KS: Int](
    queries: MutPointer[Float32, MutAnyOrigin],
    index: MutPointer[Float32, MutAnyOrigin],
    q_norm: MutPointer[Float32, MutAnyOrigin],
    y_norm: MutPointer[Float32, MutAnyOrigin],
    cand_d: MutPointer[Float32, MutAnyOrigin],
    cand_i: MutPointer[UInt32, MutAnyOrigin],
    out_d: MutPointer[Float32, MutAnyOrigin],
    out_i: MutPointer[UInt32, MutAnyOrigin],
    flags: MutPointer[Int32, MutAnyOrigin],
    n_queries_in: Int32,
    n_index_in: Int32,
    d_in: Int32,
    k_in: Int32,
    is_sqrt_in: Int32,
):
    """One thread per query: pinned values of its KC candidates, their k
    smallest (z, index) ascending into `out_*`, and `flags[q]` = 1 when the
    certificate holds (0: the caller answers this query on the tiled arm)."""
    var nq = Int(n_queries_in)
    var q = Int(block_idx.x) * CERT_TPB + Int(thread_idx.x)
    if q >= nq:
        return
    var ni = Int(n_index_in)
    var d = Int(d_in)
    var k = Int(k_in)
    var qn = ftz(q_norm[unsafe_offset = q])
    var bz = SIMD[DType.float32, KS](CERT_BIG)
    var bp = SIMD[DType.float32, KS](0.0)
    var bi = SIMD[DType.uint32, KS](0xFFFFFFFF)
    var ok = True
    for j in range(KC):
        var ci = cand_i[unsafe_offset = q * KC + j]
        if Int(ci) >= ni:
            ok = False
            continue
        var acc = Float32(0.0)
        for f in range(d):
            acc = _rt_step(
                ftz(queries[unsafe_offset = q * d + f]),
                ftz(index[unsafe_offset = Int(ci) * d + f]),
                acc,
            )
        var p = ftz(
            identical_mul_add(
                Float32(-2.0), acc, ftz(qn + ftz(y_norm[unsafe_offset = Int(ci)]))
            )
        )
        if p <= Float32(0.0):
            p = Float32(0.0)
        if p != p:
            ok = False
        var z = p
        if Int(is_sqrt_in) != 0:
            z = ftz(identical_sqrt(p))
        var cz = z
        var cp = p
        var cc = ci
        comptime for t in range(KS):
            if t < k and _key_worse(bz[t], bi[t], cz, cc):
                var tz = bz[t]
                var tp = bp[t]
                var tc = bi[t]
                bz[t] = cz
                bp[t] = cp
                bi[t] = cc
                cz = tz
                cp = tp
                cc = tc
    var t_sq = Float32(0.0)
    comptime for t in range(KS):
        if t < k:
            t_sq = max(t_sq, bp[t])
            out_d[unsafe_offset = q * k + t] = bz[t]
            out_i[unsafe_offset = q * k + t] = bi[t]
    # The certificate (see the module docstring), float32 with margins.
    var e = CERT_E_PER_FEATURE * Float32(d + 4) / Float32(16777216.0)
    var ep = e / (Float32(1.0) - Float32(2.0) * e)
    var vk = cand_d[unsafe_offset = q * KC + KC - 1]
    var w = vk * (Float32(1.0) - CERT_REL)
    var lb = w * (Float32(1.0) - Float32(2.0) * ep) - Float32(3.0) * ep * qn
    lb = lb - CERT_REL * (abs(w) + Float32(3.0) * ep * qn) - CERT_ABS * Float32(d + 4)
    var need = t_sq * (Float32(1.0) + CERT_REL) + CERT_ABS
    var cert = ok and vk > Float32(0.0) and vk < CERT_BIG and lb > need
    flags[unsafe_offset = q] = Int32(1) if cert else Int32(0)


def gather_rows_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    src: MutPointer[Float32, MutAnyOrigin],
    rows: MutPointer[Int32, MutAnyOrigin],
    n_rows_in: Int32,
    width_in: Int32,
):
    """dst[r, :] = src[rows[r], :]."""
    var w = Int(width_in)
    var e = Int(block_idx.x) * CERT_TPB + Int(thread_idx.x)
    if e >= Int(n_rows_in) * w:
        return
    var r = e // w
    var c = e - r * w
    dst[unsafe_offset = e] = src[unsafe_offset = Int(rows[unsafe_offset = r]) * w + c]


def scatter_rows_f32_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    src: MutPointer[Float32, MutAnyOrigin],
    rows: MutPointer[Int32, MutAnyOrigin],
    n_rows_in: Int32,
    width_in: Int32,
):
    """dst[rows[r], :] = src[r, :]."""
    var w = Int(width_in)
    var e = Int(block_idx.x) * CERT_TPB + Int(thread_idx.x)
    if e >= Int(n_rows_in) * w:
        return
    var r = e // w
    var c = e - r * w
    dst[unsafe_offset = Int(rows[unsafe_offset = r]) * w + c] = src[unsafe_offset = e]


def scatter_rows_u32_kernel(
    dst: MutPointer[UInt32, MutAnyOrigin],
    src: MutPointer[UInt32, MutAnyOrigin],
    rows: MutPointer[Int32, MutAnyOrigin],
    n_rows_in: Int32,
    width_in: Int32,
):
    """dst[rows[r], :] = src[r, :]."""
    var w = Int(width_in)
    var e = Int(block_idx.x) * CERT_TPB + Int(thread_idx.x)
    if e >= Int(n_rows_in) * w:
        return
    var r = e // w
    var c = e - r * w
    dst[unsafe_offset = Int(rows[unsafe_offset = r]) * w + c] = src[unsafe_offset = e]
