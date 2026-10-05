# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The eigh's power-of-two range scale (lane idn-cov-overflow, 2026-10-05).

The Jacobi's convergence test folds squares: sum of a_ij^2 off the diagonal
and sum of a_kk^2 (x_decomp/rr.mojo `rr_off_fold`). In float32 those sums
overflow once max |a_ij| passes about 2^63 / n, while every entry, every
rotation and every eigenvalue is still finite. FastICA's whitening Gram and
IncrementalPCA's batch Gram on Istella (M3 idn5, n = 220: "off-diagonal
mass 6.6e+31 of inf") are such matrices: the diagonal sum is inf,
`rr_converged(off, inf)` holds at sweep 0 and `rr_fro_kept(inf, inf)`
(inf - inf = nan) refuses the solve (DEVIATION 590), with no sweep run.
Symmetrically, below about 2^-63 the squares flush to zero and the test
passes on an unrotated matrix.

The rule: max |a_ij| in [2^(e - 1), 2^e). When e lies in [ES_LO, ES_HI] the
matrix is untouched (no write, the same words as before this lane). Outside
it A is multiplied by 2^-e (max |a| then in [0.5, 1), so the folded sums are
at most n^2 and n < 2^32 cannot overflow them), the solve runs, and the
eigenvalues are multiplied by 2^e. A power-of-two scale is exact in binary32
unless a value leaves the normal range, and the rotation (c, s), the test's
ratio off <= tol^2 fro and the Frobenius check are all homogeneous in A, so
the vectors are the unscaled solve's wherever that solve's arithmetic was
finite and normal. A value-range rule from the float32 exponent range, not a
size or dataset rule. The max is exact in any order, so every column (NVIDIA,
AMD, Apple, host) reads the same e. A NaN or infinite entry (exponent field
255), a zero or subnormal max (field 0) keeps the matrix as it is.

Each scale is two multiplications (2^-e1, then 2^-e2, e1 = e // 2): every
factor is a normal power of two for e in [-125, 128], and each product is
flushed (`ftz`), the host's and every device's denormal policy."""
from std.gpu import block_idx, thread_idx
from std.memory import bitcast, stack_allocation
from max.gpu.memory import AddressSpace
from max.gpu.sync import barrier

from checks.numerics import ftz, identical_mul
from x_decomp.cells import F32Ptr

comptime ES_LO = -32
"""The least exponent e left unscaled (max |a| >= 2^-33)."""
comptime ES_HI = 32
"""The greatest exponent e left unscaled (max |a| < 2^32): the folded squares
are then at most n^2 2^64 < 2^128 for every n < 2^32."""
comptime ES_TPB = 256
"""Block width of `es_fold_kernel`."""


@always_inline
def _pow2(k: Int) -> Float32:
    """2^k for k in [-126, 127] (a normal binary32)."""
    return bitcast[DType.float32](UInt32(k + 127) << 23)


@always_inline
def es_absmax(acc: Float32, x: Float32) -> Float32:
    """max(acc, |x|) by an ordered compare: a NaN never wins (it compares
    false), so every column folds the same word in any order."""
    var a = abs(x)
    return a if a > acc else acc


def es_factors(mx: Float32) -> SIMD[DType.float32, 4]:
    """(f1, f2, g1, g2) for a matrix whose max |a_ij| is mx: A f1 f2 is the
    scaled matrix and w g1 g2 the eigenvalues back; (1, 1, 1, 1) inside the
    band, for a zero or subnormal mx and for a nonfinite one."""
    var one = SIMD[DType.float32, 4](1.0, 1.0, 1.0, 1.0)
    var be = Int((bitcast[DType.uint32](mx) >> 23) & UInt32(0xFF))
    if be == 0 or be == 255:
        return one
    var e = be - 126
    if e >= ES_LO and e <= ES_HI:
        return one
    var e1 = e // 2
    var e2 = e - e1
    return SIMD[DType.float32, 4](_pow2(-e1), _pow2(-e2), _pow2(e1), _pow2(e2))


@always_inline
def es_mul2(x: Float32, f1: Float32, f2: Float32) -> Float32:
    """x f1 f2, each product flushed."""
    return ftz(identical_mul(ftz(identical_mul(x, f1)), f2))


def es_row_max(a: F32Ptr, n: Int, r: Int) -> Float32:
    """max |a| over the n words of row r (a + r n)."""
    var mx = Float32(0.0)
    for j in range(n):
        mx = es_absmax(mx, a.unsafe_load(r * n + j))
    return mx


def es_rowmax_kernel(a: F32Ptr, rm: F32Ptr, rows_in: Int32, n_in: Int32):
    """rm[r] = max |a| over row r of `rows` rows of n (batch problems of n x n
    stacked: rows = batch n). One thread a row."""
    var r = Int(block_idx.x) * ES_TPB + Int(thread_idx.x)
    if r < Int(rows_in):
        rm.unsafe_store(r, es_row_max(a, Int(n_in), r))


def es_fold_kernel(rm: F32Ptr, fac: F32Ptr, n_in: Int32):
    """Block b: the max over problem b's n row maxima (thread t takes rows t,
    t + ES_TPB, ..., then the tree), fac[4 b .. 4 b + 3] = `es_factors`.
    One block a problem (the max is exact in any order)."""
    var n = Int(n_in)
    var b = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var sm = stack_allocation[ES_TPB, Scalar[DType.float32], address_space = AddressSpace.SHARED]()
    var mx = Float32(0.0)
    var k = tid
    while k < n:
        mx = es_absmax(mx, rm.unsafe_load(b * n + k))
        k += ES_TPB
    sm[tid] = mx
    barrier()
    var w = ES_TPB // 2
    while w > 0:
        if tid < w:
            sm[tid] = es_absmax(sm[tid], sm[tid + w])
        barrier()
        w = w // 2
    if tid == 0:
        var f = es_factors(sm[0])
        for i in range(4):
            fac.unsafe_store(4 * b + i, f[i])


def es_apply_kernel(a: F32Ptr, fac: F32Ptr, per_in: Int32, total_in: Int32):
    """a[t] = a[t] f1 f2 with problem t // per's (f1, f2); no write when both
    are 1 (inside the band)."""
    var t = Int(block_idx.x) * ES_TPB + Int(thread_idx.x)
    if t < Int(total_in):
        var b = t // Int(per_in)
        var f1 = fac.unsafe_load(4 * b)
        var f2 = fac.unsafe_load(4 * b + 1)
        if f1 != Float32(1.0) or f2 != Float32(1.0):
            a.unsafe_store(t, es_mul2(a.unsafe_load(t), f1, f2))


def es_unscale_kernel(w: F32Ptr, fac: F32Ptr, per_in: Int32, total_in: Int32):
    """w[t] = w[t] g1 g2 with problem t // per's (g1, g2); no write when both
    are 1."""
    var t = Int(block_idx.x) * ES_TPB + Int(thread_idx.x)
    if t < Int(total_in):
        var b = t // Int(per_in)
        var g1 = fac.unsafe_load(4 * b + 2)
        var g2 = fac.unsafe_load(4 * b + 3)
        if g1 != Float32(1.0) or g2 != Float32(1.0):
            w.unsafe_store(t, es_mul2(w.unsafe_load(t), g1, g2))


def host_es_scale(mut m: List[Float32], n: Int) -> SIMD[DType.float32, 4]:
    """The host's `es_rowmax_kernel`, `es_fold_kernel` and `es_apply_kernel`
    on one n x n problem (m row major, scaled in place): the factors."""
    var mx = Float32(0.0)
    for i in range(n * n):
        mx = es_absmax(mx, m[i])
    var f = es_factors(mx)
    if f[0] != Float32(1.0) or f[1] != Float32(1.0):
        for i in range(n * n):
            m[i] = es_mul2(m[i], f[0], f[1])
    return f


def host_es_unscale(mut w: List[Float32], f: SIMD[DType.float32, 4]):
    """The host's `es_unscale_kernel`: every word of w times g1 g2."""
    if f[2] != Float32(1.0) or f[3] != Float32(1.0):
        for i in range(len(w)):
            w[i] = es_mul2(w[i], f[2], f[3])
