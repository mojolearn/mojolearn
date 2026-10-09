# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The round-robin Jacobi's schedule and pinned steps, shared by the device
kernels (x_decomp/jacobi_par.mojo) and the host solver below (no GPU imports)."""
from std.math import sqrt
from decomposition.checks.jacobi_eigh_device import jacobi_rotation_cs
from x_decomp.cells import F32Ptr
from checks.numerics import ftz, identical_mul_add, identical_mul, identical_div


@always_inline
def pj_first(r: Int, b: Int, m: Int) -> Int:
    """One player of pair b in round r of the circle method on m players
    (m even, 0 <= r < m - 1, 0 <= b < m / 2): pair 0 is (r, m - 1), pair b
    is ((r + b) mod (m - 1), (r - b) mod (m - 1))."""
    if b == 0:
        return r
    return (r + b) % (m - 1)


@always_inline
def pj_second(r: Int, b: Int, m: Int) -> Int:
    if b == 0:
        return m - 1
    return (r + m - 1 - b) % (m - 1)


# ------------------------------------------------ the round-robin eigh as THE order (lane/neural-pass104)
# Andrew, 2026-10-01: the two-sided Jacobi in the round-robin ordering is the
# eigh of every column and of the host, at every size (cgr-decomp,
# 2026-10-03: the cyclic fallback and its define are deleted). Its arithmetic is pinned here as the
# cyclic solver's: every product `identical_mul`, every rotation step the
# host cyclic `_rot_sub_1` / `_rot_add_1` form (one fma, the other product
# flushed and negated), every result flushed, the (c, s) of
# `jacobi_rotation_cs`. A round's blocks, cells and V rows each have one
# writer and no other reader, so the host, running them one after another
# in place, computes the device's words.
@always_inline
def rr_sub(c: Float32, x: Float32, s: Float32, y: Float32) -> Float32:
    """c x - s y."""
    return ftz(identical_mul_add(c, x, -ftz(identical_mul(s, y))))


@always_inline
def rr_add(s: Float32, x: Float32, c: Float32, y: Float32) -> Float32:
    """s x + c y."""
    return ftz(identical_mul_add(s, x, ftz(identical_mul(c, y))))


@always_inline
def rr_cs(a: F32Ptr, n: Int, m: Int, r: Int, b: Int) -> SIMD[DType.float32, 2]:
    """Pair b of round r: its rotation, (1, 0) for the bye."""
    var a0 = pj_first(r, b, m)
    var a1 = pj_second(r, b, m)
    var p = min(a0, a1)
    var q = max(a0, a1)
    if q >= n:
        return SIMD[DType.float32, 2](Float32(1.0), Float32(0.0))
    return jacobi_rotation_cs(a.unsafe_load(p * n + p), a.unsafe_load(q * n + q), a.unsafe_load(p * n + q))


@always_inline
def rr_block(a: F32Ptr, cs: F32Ptr, n: Int, m: Int, r: Int, i: Int, j: Int):
    """Block (i, j), i <= j, of round r: J_i^T B J_j (and its mirror); the
    pair's own block in closed form."""
    var i0 = pj_first(r, i, m)
    var i1 = pj_second(r, i, m)
    var pi = min(i0, i1)
    var qi = max(i0, i1)
    var ci = cs.unsafe_load(2 * i)
    var si = cs.unsafe_load(2 * i + 1)
    if i == j:
        if qi < n:
            var app = a.unsafe_load(pi * n + pi)
            var aqq = a.unsafe_load(qi * n + qi)
            var apq = a.unsafe_load(pi * n + qi)
            var tt = ftz(identical_div(si, ci))
            var dlt = ftz(identical_mul(tt, apq))
            a.unsafe_store(pi * n + pi, ftz(app - dlt))
            a.unsafe_store(qi * n + qi, ftz(aqq + dlt))
            a.unsafe_store(pi * n + qi, Float32(0.0))
            a.unsafe_store(qi * n + pi, Float32(0.0))
        return
    var j0 = pj_first(r, j, m)
    var j1 = pj_second(r, j, m)
    var pj = min(j0, j1)
    var qj = max(j0, j1)
    var cj = cs.unsafe_load(2 * j)
    var sj = cs.unsafe_load(2 * j + 1)
    var vi = qi < n
    var vj = qj < n
    var b00 = a.unsafe_load(pi * n + pj)
    var b01 = Float32(0.0)
    var b10 = Float32(0.0)
    var b11 = Float32(0.0)
    if vj:
        b01 = a.unsafe_load(pi * n + qj)
    if vi:
        b10 = a.unsafe_load(qi * n + pj)
    if vi and vj:
        b11 = a.unsafe_load(qi * n + qj)
    var t00 = rr_sub(cj, b00, sj, b01)
    var t01 = rr_add(sj, b00, cj, b01)
    var t10 = rr_sub(cj, b10, sj, b11)
    var t11 = rr_add(sj, b10, cj, b11)
    var n00 = rr_sub(ci, t00, si, t10)
    var n01 = rr_sub(ci, t01, si, t11)
    var n10 = rr_add(si, t00, ci, t10)
    var n11 = rr_add(si, t01, ci, t11)
    a.unsafe_store(pi * n + pj, n00)
    a.unsafe_store(pj * n + pi, n00)
    if vj:
        a.unsafe_store(pi * n + qj, n01)
        a.unsafe_store(qj * n + pi, n01)
    if vi:
        a.unsafe_store(qi * n + pj, n10)
        a.unsafe_store(pj * n + qi, n10)
    if vi and vj:
        a.unsafe_store(qi * n + qj, n11)
        a.unsafe_store(qj * n + qi, n11)


@always_inline
def rr_vrow(v: F32Ptr, cs: F32Ptr, n: Int, m: Int, r: Int, k: Int, j: Int):
    """V = V J for row k, pair j of round r."""
    var j0 = pj_first(r, j, m)
    var j1 = pj_second(r, j, m)
    var pj = min(j0, j1)
    var qj = max(j0, j1)
    if qj < n:
        var cj = cs.unsafe_load(2 * j)
        var sj = cs.unsafe_load(2 * j + 1)
        var vkp = v.unsafe_load(k * n + pj)
        var vkq = v.unsafe_load(k * n + qj)
        v.unsafe_store(k * n + pj, rr_sub(cj, vkp, sj, vkq))
        v.unsafe_store(k * n + qj, rr_add(sj, vkp, cj, vkq))


@always_inline
def rr_row_off(a: F32Ptr, n: Int, k: Int) -> SIMD[DType.float32, 2]:
    """(row k's off-diagonal squares, j ascending; a_kk squared)."""
    var acc = Float32(0.0)
    for j in range(n):
        if j != k:
            var x = a.unsafe_load(k * n + j)
            acc = ftz(identical_mul_add(x, x, acc))
    var d = a.unsafe_load(k * n + k)
    return SIMD[DType.float32, 2](acc, ftz(identical_mul(d, d)))



comptime RR_OFF_TPB = 256
"""Block width of the convergence test's fold (`eigh_par_off_part_kernel`,
`eigh_par_off_fold_kernel`)."""


def _rr_tree(mut so: InlineArray[Float32, RR_OFF_TPB], mut sd: InlineArray[Float32, RR_OFF_TPB]):
    """The pairwise tree over the RR_OFF_TPB slots (slot 0 the sum)."""
    var w = RR_OFF_TPB // 2
    while w > 0:
        for t in range(w):
            so[t] = ftz(so[t] + so[t + w])
            sd[t] = ftz(sd[t] + sd[t + w])
        w = w // 2


def rr_off_fold(a: F32Ptr, n: Int) -> SIMD[DType.float32, 2]:
    """(sum of the rows' off-diagonal squares, sum of a_kk^2) in the device's
    order: block b of RR_OFF_TPB rows a pairwise tree
    (`eigh_par_off_part_kernel`), then slot t adds block sums t, t +
    RR_OFF_TPB, ... ascending and the pairwise tree again
    (`eigh_par_off_fold_kernel`)."""
    var nb = (n + RR_OFF_TPB - 1) // RR_OFF_TPB
    var po = List[Float32](length=max(nb, 1), fill=Float32(0.0))
    var pd = List[Float32](length=max(nb, 1), fill=Float32(0.0))
    for b in range(nb):
        var so = InlineArray[Float32, RR_OFF_TPB](fill=Float32(0.0))
        var sd = InlineArray[Float32, RR_OFF_TPB](fill=Float32(0.0))
        for t in range(RR_OFF_TPB):
            var k = b * RR_OFF_TPB + t
            if k < n:
                var o = rr_row_off(a, n, k)
                so[t] = o[0]
                sd[t] = o[1]
        _rr_tree(so, sd)
        po[b] = so[0]
        pd[b] = sd[0]
    var so = InlineArray[Float32, RR_OFF_TPB](fill=Float32(0.0))
    var sd = InlineArray[Float32, RR_OFF_TPB](fill=Float32(0.0))
    for t in range(RR_OFF_TPB):
        var b = t
        while b < nb:
            so[t] = ftz(so[t] + po[b])
            sd[t] = ftz(sd[t] + pd[b])
            b += RR_OFF_TPB
    _rr_tree(so, sd)
    return SIMD[DType.float32, 2](so[0], sd[0])


comptime RR_EIGH_SWEEPS = 60
"""The round-robin eigh's sweep budget, every column and the host (cgr-decomp,
2026-10-03: the cyclic one-block solver it used to fall back to is deleted;
a solve that does not converge in this budget raises)."""


@always_inline
def rr_converged(off: Float32, dg: Float32, tol: Float32) -> Bool:
    """The convergence test on the folded sums, float32 only (the batched
    kernel decides it on the device, Metal has no float64): off <= tol^2
    (off + dg)."""
    var fro = ftz(off + dg)
    var t2 = ftz(identical_mul(tol, tol))
    return off <= ftz(identical_mul(t2, fro))


@always_inline
def rr_fro_kept(fro_in: Float32, fro_now: Float32) -> Bool:
    """J^T A J keeps ||A||_F: |fro_now - fro_in| <= 1e-3 fro_in, float32."""
    return abs(ftz(fro_now - fro_in)) <= ftz(identical_mul(Float32(1.0e-3), fro_in))


@always_inline
def rr_gate_state(off: Float32, dg: Float32, mark: Float32, state: F32Ptr, tol: Float32):
    """The round-robin solve's device convergence decision on the six state
    words (PCA_RR_STATE; `pca_rr_gate_kernel`, decomposition/impl/linalg/
    detail/pca.mojo, and the one-block sweep x_decomp/rr_one_block.mojo both
    run exactly this): state[0] = 1 once `rr_converged` holds (sticky),
    state[1] = the off-diagonal sum, state[2] = the first test's ||A||_F^2
    (the caller fills -1), state[3] = this test's, state[4] = -1 when a block
    of the test did not run (`mark` < 0), state[5] = sweeps started. One
    thread. Extracted unchanged from `pca_rr_gate_kernel` (lane fg-pca P1)."""
    if state.unsafe_load(0) == Float32(0.0):
        if not (mark >= Float32(0.0)):
            state.unsafe_store(4, Float32(-1.0))
        var fro = ftz(off + dg)
        state.unsafe_store(1, off)
        state.unsafe_store(3, fro)
        if state.unsafe_load(2) < Float32(0.0):
            state.unsafe_store(2, fro)
        if rr_converged(off, dg, tol):
            state.unsafe_store(0, Float32(1.0))
        else:
            state.unsafe_store(5, state.unsafe_load(5) + Float32(1.0))


def host_eigh_rr(mut a: List[Float32], mut v: List[Float32], n: Int, sweeps: Int, tol: Float32) -> Tuple[Bool, Int]:
    """The device driver's solve on the host (x_decomp/device.mojo `_eigh_par`,
    x_decomp/rr_batch.mojo `rr_batch_kernel`): `a` consumed in place (its
    diagonal the eigenvalues), v = the vectors in columns (row major). The
    same rounds in the same order, the same convergence test (`rr_off_fold`,
    `rr_converged`, before every sweep) and the same Frobenius check
    (`rr_fro_kept`). Returns (converged, sweeps run)."""
    var m = n + (n % 2)
    var h = m // 2
    for i in range(n):
        for j in range(n):
            v[i * n + j] = Float32(1.0) if i == j else Float32(0.0)
    var ap = F32Ptr(unsafe_from_address=Int(a.unsafe_ptr()))
    var vp = F32Ptr(unsafe_from_address=Int(v.unsafe_ptr()))
    var csl = List[Float32](length=max(2 * h, 2), fill=Float32(0.0))
    var cs = F32Ptr(unsafe_from_address=Int(csl.unsafe_ptr()))
    var converged = False
    var executed = 0
    var fro_in = Float32(-1.0)
    var fro_now = Float32(0.0)
    for sweep in range(sweeps + 1):
        var sums = rr_off_fold(ap, n)
        fro_now = ftz(sums[0] + sums[1])
        if fro_in < Float32(0.0):
            fro_in = fro_now
        if rr_converged(sums[0], sums[1], tol):
            converged = True
            break
        if sweep == sweeps:
            break
        executed += 1
        for rd in range(m - 1):
            for b in range(h):
                var got = rr_cs(ap, n, m, rd, b)
                cs.unsafe_store(2 * b, got[0])
                cs.unsafe_store(2 * b + 1, got[1])
            for i in range(h):
                for j in range(i, h):
                    rr_block(ap, cs, n, m, rd, i, j)
            for k in range(n):
                for j in range(h):
                    rr_vrow(vp, cs, n, m, rd, k, j)
    if converged and not rr_fro_kept(fro_in, fro_now):
        converged = False
    _ = csl^
    return (converged, executed)
