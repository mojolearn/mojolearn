# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Opt-in Apple FAST Jacobi: tangent rotations and one immutable-source round.

Represent c*x - s*y as x - s*(y + tau*x), tau=s/(1+c). This
avoids repeatedly scaling by a rounded cosine close to one. Every round
reads A from src and writes its disjoint pair blocks to dst, so coefficients
can be recomputed without a separate kernel or an inter-block race.
V is updated in place: each (row, pair) has a unique writer.
"""
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from std.gpu import block_dim, block_idx, thread_idx
from std.math import sqrt, fma
from std.memory import stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, ftz, identical_mul
from x_decomp.cells import F32Ptr
from x_decomp.rr import pj_first, pj_second

# Failed M3 tag gap26-eigh-tangent-quality-fix, head 7e0422e95:
# max eigenvalue error board128 1.27636e-6 -> 1.57335e-6,
# gram128 8.81720e-7 -> 1.19155e-6; board257 1.49875e-6 -> 2.03518e-6.
# Residual and orthogonality improved, but transformed-diagonal eigenvalues
# drifted. Current candidate extracts normalized Rayleigh quotients from the
# original GPU matrix, with compensated sums; no quality gate is relaxed.
# Repaired source 131a0d78a, current M3 manager result (2026-10-04):
# quality PASS, but synthetic 43796.817 -> 51254.836 ms (+17.03%): HOLD.
# Eigenvalue error 6.08669e-5 -> 3.52564e-7; residual 5.37499e-5 ->
# 4.82917e-6. Retain Rayleigh repair; do not default the slower experiment.
# New opt-in caches each rotation once instead of repeating its sqrt/divide
# for every matrix block and V row. Pending quality and speed, not a KEEP.
comptime EIGH_TANGENT_CACHE = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and is_defined["MOJOLEARN_EIGH_TANGENT_CACHE"]()
)

comptime EIGH_FAST_TANGENT = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST
    and has_apple_gpu_accelerator()
    and (is_defined["MOJOLEARN_EIGH_FAST_TANGENT"]() or EIGH_TANGENT_CACHE)
)

@always_inline
def stable_cs(a: F32Ptr, n: Int, m: Int, r: Int, b: Int) -> SIMD[DType.float32, 4]:
    var p = min(pj_first(r, b, m), pj_second(r, b, m))
    var q = max(pj_first(r, b, m), pj_second(r, b, m))
    if q >= n:
        return SIMD[DType.float32, 4](0.0, 0.0, 0.0, 1.0)
    var apq = a.unsafe_load(p*n+q)
    if apq == Float32(0):
        return SIMD[DType.float32, 4](0.0, 0.0, 0.0, 1.0)
    var delta = a.unsafe_load(q*n+q) - a.unsafe_load(p*n+p)
    var scale = max(abs(delta), abs(apq))
    var d = delta / scale
    var z = apq / scale
    var h = sqrt(fma(d, d, Float32(4)*z*z))
    var t = Float32(1)
    if delta != Float32(0):
        t = (Float32(2)*z) / (d + (h if d > Float32(0) else -h))
    var c = Float32(1) / sqrt(fma(t, t, Float32(1)))
    var s = t*c
    return SIMD[DType.float32, 4](s, s/(Float32(1)+c), t, c)

@always_inline
def round_cs[cached: Bool](a: F32Ptr, coeff: F32Ptr, n: Int, m: Int, r: Int, b: Int) -> SIMD[DType.float32, 4]:
    comptime if cached:
        return coeff.unsafe_load[width=4](4 * b)
    else:
        return stable_cs(a, n, m, r, b)


@always_inline
def stable_sub(tau: Float32, x: Float32, s: Float32, y: Float32) -> Float32:
    return fma(-s, fma(tau, x, y), x)

@always_inline
def stable_add(s: Float32, x: Float32, tau: Float32, y: Float32) -> Float32:
    return fma(s, fma(-tau, y, x), y)

@always_inline
def stable_block[cached: Bool = False](src: F32Ptr, a: F32Ptr, n: Int, m: Int, r: Int, i: Int, j: Int, coeff: F32Ptr):
    """Block (i, j), i <= j, of round r: J_i^T B J_j (and its mirror); the
    pair's own block in closed form."""
    var i0 = pj_first(r, i, m)
    var i1 = pj_second(r, i, m)
    var pi = min(i0, i1)
    var qi = max(i0, i1)
    var ri = round_cs[cached](src, coeff, n, m, r, i)
    var ci = ri[1]
    var si = ri[0]
    if i == j:
        if qi < n:
            var app = src.unsafe_load(pi * n + pi)
            var aqq = src.unsafe_load(qi * n + qi)
            var apq = src.unsafe_load(pi * n + qi)
            var tt = ri[2]
            var dlt = ftz(identical_mul(tt, apq))
            a.unsafe_store(pi * n + pi, ftz(app - dlt))
            a.unsafe_store(qi * n + qi, ftz(aqq + dlt))
            a.unsafe_store(pi * n + qi, Float32(0.0))
            a.unsafe_store(qi * n + pi, Float32(0.0))
        else:
            a.unsafe_store(pi * n + pi, src.unsafe_load(pi * n + pi))
        return
    var j0 = pj_first(r, j, m)
    var j1 = pj_second(r, j, m)
    var pj = min(j0, j1)
    var qj = max(j0, j1)
    var rj = round_cs[cached](src, coeff, n, m, r, j)
    var cj = rj[1]
    var sj = rj[0]
    var vi = qi < n
    var vj = qj < n
    var b00 = src.unsafe_load(pi * n + pj)
    var b01 = Float32(0.0)
    var b10 = Float32(0.0)
    var b11 = Float32(0.0)
    if vj:
        b01 = src.unsafe_load(pi * n + qj)
    if vi:
        b10 = src.unsafe_load(qi * n + pj)
    if vi and vj:
        b11 = src.unsafe_load(qi * n + qj)
    var t00 = stable_sub(cj, b00, sj, b01)
    var t01 = stable_add(sj, b00, cj, b01)
    var t10 = stable_sub(cj, b10, sj, b11)
    var t11 = stable_add(sj, b10, cj, b11)
    var n00 = stable_sub(ci, t00, si, t10)
    var n01 = stable_sub(ci, t01, si, t11)
    var n10 = stable_add(si, t00, ci, t10)
    var n11 = stable_add(si, t01, ci, t11)
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
def stable_vrow[cached: Bool = False](src: F32Ptr, v: F32Ptr, n: Int, m: Int, r: Int, k: Int, j: Int, coeff: F32Ptr):
    """V = V J for row k, pair j of round r."""
    var j0 = pj_first(r, j, m)
    var j1 = pj_second(r, j, m)
    var pj = min(j0, j1)
    var qj = max(j0, j1)
    if qj < n:
        var rj = round_cs[cached](src, coeff, n, m, r, j)
        var cj = rj[1]
        var sj = rj[0]
        var vkp = v.unsafe_load(k * n + pj)
        var vkq = v.unsafe_load(k * n + qj)
        v.unsafe_store(k * n + pj, stable_sub(cj, vkp, sj, vkq))
        v.unsafe_store(k * n + qj, stable_add(sj, vkp, cj, vkq))



def eigh_tangent_round_kernel(src: F32Ptr, dst: F32Ptr, v: F32Ptr, n_in: Int32, m_in: Int32, round_in: Int32):
    var n = Int(n_in)
    var m = Int(m_in)
    var h = m // 2
    var r = Int(round_in)
    var t = Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if t < h*h:
        var i = t // h
        var j = t % h
        if i <= j:
            stable_block(src, dst, n, m, r, i, j, src)
    elif t < h*h+n*h:
        var u = t-h*h
        stable_vrow(src, v, n, m, r, u//h, u%h, src)


def eigh_tangent_cs_kernel(a: F32Ptr, coeff: F32Ptr, n_in: Int32, m_in: Int32, round_in: Int32):
    """One parallel coefficient calculation per pair, before any A write."""
    var b = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if b < Int(m_in) // 2:
        coeff.unsafe_store[width=4](
            4 * b, stable_cs(a, Int(n_in), Int(m_in), Int(round_in), b)
        )


def eigh_tangent_cached_round_kernel(a: F32Ptr, v: F32Ptr, coeff: F32Ptr, n_in: Int32, m_in: Int32, round_in: Int32):
    """Same stable arithmetic as the repaired tangent round, cached c/s/tau.

    Every triangular pair-of-pairs owns its disjoint 2x2 block and mirror;
    every V row/pair owns two entries. The only cross-block inputs were the
    diagonals/off-diagonal pivot entries used by stable_cs, now read from the
    immutable coefficient cache. Consequently A may be updated in place.
    """
    var n = Int(n_in)
    var m = Int(m_in)
    var h = m // 2
    var r = Int(round_in)
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t < h * h:
        var i = t // h
        var j = t % h
        if i <= j:
            stable_block[True](a, a, n, m, r, i, j, coeff)
    elif t < h * h + n * h:
        var u = t - h * h
        stable_vrow[True](a, v, n, m, r, u // h, u % h, coeff)


def eigh_tangent_copy_kernel(src: F32Ptr, dst: F32Ptr, n_in: Int32):
    var t = Int(block_idx.x)*Int(block_dim.x)+Int(thread_idx.x)
    if t < Int(n_in)*Int(n_in):
        dst.unsafe_store(t, src.unsafe_load(t))


@always_inline
def rayleigh_add(a: SIMD[DType.float32, 2], b: SIMD[DType.float32, 2]) -> SIMD[DType.float32, 2]:
    # Error-free two-sum of the high words, then renormalize the low words.
    var s = a[0] + b[0]
    var z = s - a[0]
    var e = ((a[0] - (s - z)) + (b[0] - z)) + (a[1] + b[1])
    var h = s + e
    return SIMD[DType.float32, 2](h, e - (h - s))


@always_inline
def rayleigh_product(a: Float32, b: Float32) -> SIMD[DType.float32, 2]:
    var h = a * b
    return SIMD[DType.float32, 2](h, fma(a, b, -h))


def eigh_rayleigh_kernel(v: F32Ptr, av: F32Ptr, diag: F32Ptr, n_in: Int32):
    """One block per vector: normalized v^T (A_original v), compensated
    dot products and pairwise reduction; A_original v is GPU matrix GEMM.
    Quotient normalization removes vector-norm drift from eigenvalues.
    """
    comptime NT = 256
    var n = Int(n_in)
    var j = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var nh = stack_allocation[NT, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var nl = stack_allocation[NT, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var dh = stack_allocation[NT, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var dl = stack_allocation[NT, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var num = SIMD[DType.float32, 2](0.0, 0.0)
    var den = SIMD[DType.float32, 2](0.0, 0.0)
    var i = tid
    while i < n:
        var x = v.unsafe_load(i*n+j)
        num = rayleigh_add(num, rayleigh_product(x, av.unsafe_load(i*n+j)))
        den = rayleigh_add(den, rayleigh_product(x, x))
        i += NT
    nh[tid] = num[0]
    nl[tid] = num[1]
    dh[tid] = den[0]
    dl[tid] = den[1]
    barrier()
    var step = NT // 2
    while step > 0:
        if tid < step:
            num = rayleigh_add(SIMD[DType.float32, 2](nh[tid], nl[tid]),
                               SIMD[DType.float32, 2](nh[tid+step], nl[tid+step]))
            den = rayleigh_add(SIMD[DType.float32, 2](dh[tid], dl[tid]),
                               SIMD[DType.float32, 2](dh[tid+step], dl[tid+step]))
            nh[tid] = num[0]
            nl[tid] = num[1]
            dh[tid] = den[0]
            dl[tid] = den[1]
        barrier()
        step //= 2
    if tid == 0:
        var q = nh[0] / dh[0]
        var r = fma(-q, dh[0], nh[0]) + nl[0] - q*dl[0]
        diag.unsafe_store(j, q + r/dh[0])
