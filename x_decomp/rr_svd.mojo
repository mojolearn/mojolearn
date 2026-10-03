# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The one-sided Jacobi SVD of the decomp kit in the ROUND-ROBIN order
(cgr-decomp, 2026-10-03). The cyclic kernels (`one_sided_svd2_chunk_kernel`,
`svd_of_r`) ran every rotation of a sweep one after another on ONE block;
here a sweep is m - 1 rounds of the circle method on the n columns (m = n
or n + 1, the odd one a bye), the m / 2 pairs of a round DISJOINT, one block
each, all at once.

A pair (p, q) of round r: its three sums (||r_p||^2, ||r_q||^2, r_p . r_q)
over the n rows of R^T, lane t of RS_TPB accumulating rows t, t + RS_TPB,
... ascending (fma), then the pairwise tree over the lanes; the rotation
when |r_p . r_q| > tol ||r_p|| ||r_q|| and it is significant
(`svd_rotation_significant`), (c, s) = `jacobi_rotation_cs`, applied to the
two rows of R^T and of V^T (`rr_sub` / `rr_add`). A sweep with no rotation
is converged. The singular values are the rows' norms (the same fold), V =
(V^T)^T. These cells are the device kernels' (x_decomp/rr_svd_device.mojo)
and the host column's (`host_rr_svd`), so the words are the same. No GPU
import."""
from checks.numerics import ftz, identical_mul, identical_mul_add, identical_sqrt
from decomposition.checks.jacobi_eigh_device import jacobi_rotation_cs
from x_decomp.cells import F32Ptr, svd_rotation_significant
from x_decomp.rr import pj_first, pj_second, rr_add, rr_sub

comptime RS_TPB = 256
"""Lanes of a pair's fold (the block width of `rs_round_kernel`)."""


@always_inline
def rs_pair(r: Int, b: Int, m: Int) -> Tuple[Int, Int]:
    """Pair b of round r, (p, q) with p < q (q >= n is the bye)."""
    var a0 = pj_first(r, b, m)
    var a1 = pj_second(r, b, m)
    return (min(a0, a1), max(a0, a1))


@always_inline
def rs_lane3(rt: F32Ptr, n: Int, p: Int, q: Int, t: Int) -> SIMD[DType.float32, 4]:
    """Lane t's (pp, qq, pq, 0) over rows t, t + RS_TPB, ... ascending."""
    var pp = Float32(0.0)
    var qq = Float32(0.0)
    var pq = Float32(0.0)
    var i = t
    while i < n:
        var xp = ftz(rt.unsafe_load(p * n + i))
        var xq = ftz(rt.unsafe_load(q * n + i))
        pp = ftz(identical_mul_add(xp, xp, pp))
        qq = ftz(identical_mul_add(xq, xq, qq))
        pq = ftz(identical_mul_add(xp, xq, pq))
        i += RS_TPB
    return SIMD[DType.float32, 4](pp, qq, pq, 0.0)


@always_inline
def rs_lane_sq(rt: F32Ptr, n: Int, j: Int, t: Int) -> Float32:
    var acc = Float32(0.0)
    var i = t
    while i < n:
        var x = ftz(rt.unsafe_load(j * n + i))
        acc = ftz(identical_mul_add(x, x, acc))
        i += RS_TPB
    return acc


@always_inline
def rs_decide(pp: Float32, qq: Float32, pq: Float32, tol: Float32) -> SIMD[DType.float32, 4]:
    """(rotate 1 / 0, c, s, 0) of a pair from its folded sums."""
    var thresh = ftz(identical_mul(tol, ftz(identical_mul(ftz(identical_sqrt(pp)), ftz(identical_sqrt(qq))))))
    if abs(pq) > thresh and svd_rotation_significant(pp, qq):
        var cs = jacobi_rotation_cs(pp, qq, pq)
        return SIMD[DType.float32, 4](1.0, cs[0], cs[1], 0.0)
    return SIMD[DType.float32, 4](0.0, 1.0, 0.0, 0.0)


@always_inline
def rs_apply_lane(rt: F32Ptr, vt: F32Ptr, n: Int, p: Int, q: Int, c: Float32, s: Float32, t: Int):
    """Lane t's rows of the rotation of rows p and q of R^T and V^T."""
    var i = t
    while i < n:
        var xp = rt.unsafe_load(p * n + i)
        var xq = rt.unsafe_load(q * n + i)
        rt.unsafe_store(p * n + i, rr_sub(c, xp, s, xq))
        rt.unsafe_store(q * n + i, rr_add(s, xp, c, xq))
        var vp = vt.unsafe_load(p * n + i)
        var vq = vt.unsafe_load(q * n + i)
        vt.unsafe_store(p * n + i, rr_sub(c, vp, s, vq))
        vt.unsafe_store(q * n + i, rr_add(s, vp, c, vq))
        i += RS_TPB


def _host_tree4(mut so: List[SIMD[DType.float32, 4]]) -> SIMD[DType.float32, 4]:
    var w = RS_TPB // 2
    while w > 0:
        for t in range(w):
            so[t] = SIMD[DType.float32, 4](
                ftz(so[t][0] + so[t + w][0]), ftz(so[t][1] + so[t + w][1]), ftz(so[t][2] + so[t + w][2]), 0.0
            )
        w = w // 2
    return so[0]


def host_rr_svd(rt: F32Ptr, vt: F32Ptr, s: F32Ptr, n: Int, sweeps: Int, tol: Float32) -> Tuple[Bool, Int]:
    """The device's rounds on the host: rt = R^T (consumed), vt = V^T (set
    to I here, rotated), s = the rows' norms. Returns (converged, sweeps)."""
    var m = n + (n % 2)
    var h = m // 2
    for i in range(n):
        for j in range(n):
            vt.unsafe_store(i * n + j, Float32(1.0) if i == j else Float32(0.0))
    var lanes = List[SIMD[DType.float32, 4]](length=RS_TPB, fill=SIMD[DType.float32, 4](0.0))
    var converged = n < 2
    var executed = 0
    while not converged and executed < sweeps:
        executed += 1
        var any = False
        for rd in range(m - 1):
            for b in range(h):
                var pq = rs_pair(rd, b, m)
                if pq[1] >= n:
                    continue
                for t in range(RS_TPB):
                    lanes[t] = rs_lane3(rt, n, pq[0], pq[1], t)
                var tot = _host_tree4(lanes)
                var d = rs_decide(tot[0], tot[1], tot[2], tol)
                if d[0] > Float32(0.0):
                    any = True
                    for t in range(RS_TPB):
                        rs_apply_lane(rt, vt, n, pq[0], pq[1], d[1], d[2], t)
        if not any:
            converged = True
    for j in range(n):
        for t in range(RS_TPB):
            lanes[t] = SIMD[DType.float32, 4](rs_lane_sq(rt, n, j, t), 0.0, 0.0, 0.0)
        s.unsafe_store(j, ftz(identical_sqrt(_host_tree4(lanes)[0])))
    return (converged, executed)
