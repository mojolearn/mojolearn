# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE DECOMP LANE'S ONE SOURCE OF ARITHMETIC (lane/algos-decomp, 2026-09-27).

Every number the decomp lane computes is computed by a cell in this file.
The device kernels (`x_decomp/device.mojo`) run a cell per thread; the host
loops (`x_decomp/host.mojo`) run the same cell per index. There is no second
spelling of any arithmetic, so the CPU column and a GPU column can only
differ where the SAME cell compiles differently for two targets, and every
cell is written so it cannot:

  * fixed reduction order: every sum is one sequential loop, ascending index,
    one thread per output; no atomics, no tree, no split-k;
  * every operand and every result through `ftz` (the denormal policy);
  * every product `identical_mul` / `identical_mul_add` (the contraction
    pin), every quotient `identical_div`, every sqrt/exp/log/tanh/cos the
    `identical_*` (portable, one arithmetic) spelling;
  * no computed NaN: every division and log is guarded explicitly, a zero
    divisor yields 0 (IDENTITY_PATHS Clause B), never a vendor's payload;
  * no float64 anywhere.
"""
from std.memory import bitcast

from checks.numerics import (
    ftz,
    identical_cos,
    identical_div,
    identical_exp,
    identical_log,
    identical_mul,
    identical_mul_add,
    identical_sqrt,
    identical_tanh,
)
from core.philox import philox4x32_10

#: The one-sided Jacobi SVD's sweep budget in this lane (x_decomp svd):
#: decomposition/'s JACOBI_SWEEPS (15, RAFT's) stops short on the `wide`
#: fixture's 1e-4..1e4 column scales; the solver is the same, the refusal of
#: an unconverged answer is kept, only the budget is larger.
comptime X_DECOMP_SVD_SWEEPS = 60
#: ... and its rotation threshold: |apq| > tol * sqrt(app * aqq). JACOBI_TOL
#: (1e-7) sits under float32 epsilon (1.19e-7), so on an ill-conditioned
#: matrix one rotation per sweep can fire forever (a rounding limit cycle,
#: measured on the `wide` fixture through FactorAnalysis). 8 epsilon is the
#: usual float32 one-sided Jacobi threshold (LAPACK sgesvj uses m*eps).
comptime X_DECOMP_SVD_TOL = Float32(9.5367431640625e-07)

comptime F32Ptr = MutPointer[Float32, MutAnyOrigin]
comptime I32Ptr = MutPointer[Int32, MutAnyOrigin]

# elementwise op codes (python/mojolearn/_expansion_decomp.py `_OP`)
comptime OP_ADD = 0
comptime OP_SUB = 1
comptime OP_MUL = 2
comptime OP_DIV = 3
comptime OP_AXPY = 4
comptime OP_MAXS = 5
comptime OP_MU = 6
comptime OP_SQRT = 7
comptime OP_SQ = 8
comptime OP_EXP = 9
comptime OP_LOGS = 10
comptime OP_TANH = 11
comptime OP_ONEMSQ = 12
comptime OP_ABS = 13
comptime OP_SCALE = 14
comptime OP_FMA = 15
comptime OP_RECIP = 16
comptime OP_SOFT = 17
comptime OP_SUBMUL = 18
comptime OP_MINS = 19
comptime OP_COPYB = 20
comptime OP_SQDIFF = 21
comptime OP_ADDS = 22
comptime OP_GTS = 23
comptime OP_DIGAMMA = 24
comptime OP_EXPG = 25
comptime OP_EXPGP = 26
comptime OP_CUBE = 27
comptime OP_CUBEP = 28
comptime OP_MAX = 30
comptime OP_MIN = 31
comptime OP_SIGN = 33
comptime OP_LE = 34
comptime OP_SELECT = 35
comptime OP_MUZ = 36


@always_inline
def mul(a: Float32, b: Float32) -> Float32:
    return ftz(identical_mul(ftz(a), ftz(b)))


@always_inline
def add(a: Float32, b: Float32) -> Float32:
    return ftz(ftz(a) + ftz(b))


@always_inline
def sub(a: Float32, b: Float32) -> Float32:
    return ftz(ftz(a) - ftz(b))


@always_inline
def div0(a: Float32, b: Float32) -> Float32:
    """a / b, and 0 for a zero divisor (never inf, never NaN)."""
    var bb = ftz(b)
    if bb == Float32(0):
        return Float32(0)
    return ftz(identical_div(ftz(a), bb))


@always_inline
def sqrt0(a: Float32) -> Float32:
    var x = ftz(a)
    if not (x > Float32(0)):
        return Float32(0)
    return ftz(identical_sqrt(x))


@always_inline
def log_floor(a: Float32, floor: Float32) -> Float32:
    var x = ftz(a)
    var f = ftz(floor)
    if not (x > f):
        x = f
    if not (x > Float32(0)):
        return Float32(0)
    return ftz(identical_log(x))


@always_inline
def exp_c(a: Float32) -> Float32:
    var x = ftz(a)
    # clamp to the finite range of expf so no inf ever enters a product
    if x > Float32(88.0):
        x = Float32(88.0)
    if x < Float32(-103.0):
        return Float32(0)
    return ftz(identical_exp(x))


def digamma(a: Float32) -> Float32:
    """psi(x) for x > 0: the recurrence up to 6, then the asymptotic series
    (sklearn's LDA calls scipy.special.psi; this is its standard expansion)."""
    var x = ftz(a)
    if not (x > Float32(0)):
        return Float32(0)
    var r = Float32(0)
    while x < Float32(6):
        r = sub(r, div0(Float32(1), x))
        x = add(x, Float32(1))
    var inv = div0(Float32(1), x)
    var inv2 = mul(inv, inv)
    # 1/(12x^2) - 1/(120x^4) + 1/(252x^6), Horner in inv2
    var t = mul(inv2, Float32(0.003968253968253968))
    t = sub(Float32(0.008333333333333333), t)
    t = mul(inv2, t)
    t = sub(Float32(0.08333333333333333), t)
    t = mul(inv2, t)
    r = add(r, log_floor(x, Float32(0)))
    r = sub(r, mul(Float32(0.5), inv))
    return sub(r, t)


def ew_cell(op: Int, x_in: Float32, y_in: Float32, z_in: Float32, s_in: Float32) -> Float32:
    var x = ftz(x_in)
    var y = ftz(y_in)
    var z = ftz(z_in)
    var s = ftz(s_in)
    var r = Float32(0)
    if op == OP_ADD:
        r = add(x, y)
    elif op == OP_SUB:
        r = sub(x, y)
    elif op == OP_MUL:
        r = mul(x, y)
    elif op == OP_DIV:
        r = div0(x, y)
    elif op == OP_AXPY:
        r = ftz(identical_mul_add(s, y, x))
    elif op == OP_MAXS:
        r = x if x > s else s
    elif op == OP_MU:
        r = div0(mul(x, y), add(z, s))
    elif op == OP_SQRT:
        r = sqrt0(x)
    elif op == OP_SQ:
        r = mul(x, x)
    elif op == OP_EXP:
        r = exp_c(x)
    elif op == OP_LOGS:
        r = log_floor(x, s)
    elif op == OP_TANH:
        r = ftz(identical_tanh(x))
    elif op == OP_ONEMSQ:
        r = sub(Float32(1), mul(x, x))
    elif op == OP_ABS:
        r = abs(x)
    elif op == OP_SCALE:
        r = mul(x, s)
    elif op == OP_FMA:
        r = ftz(identical_mul_add(x, y, z))
    elif op == OP_RECIP:
        r = div0(Float32(1), x)
    elif op == OP_SOFT:
        var m = sub(abs(x), s)
        if m > Float32(0):
            r = m if x > Float32(0) else -m
        else:
            r = Float32(0)
    elif op == OP_SUBMUL:
        r = mul(sub(x, y), z)
    elif op == OP_MINS:
        r = x if x < s else s
    elif op == OP_COPYB:
        r = y
    elif op == OP_SQDIFF:
        var t = sub(x, y)
        r = mul(t, t)
    elif op == OP_ADDS:
        r = add(x, s)
    elif op == OP_GTS:
        r = Float32(1) if x > s else Float32(0)
    elif op == OP_DIGAMMA:
        r = digamma(x)
    elif op == OP_EXPG:
        r = mul(x, exp_c(mul(Float32(-0.5), mul(x, x))))
    elif op == OP_EXPGP:
        var x2 = mul(x, x)
        r = mul(sub(Float32(1), x2), exp_c(mul(Float32(-0.5), x2)))
    elif op == OP_CUBE:
        r = mul(mul(x, x), x)
    elif op == OP_CUBEP:
        r = mul(Float32(3), mul(x, x))
    elif op == OP_MAX:
        r = x if x > y else y
    elif op == OP_MIN:
        r = x if x < y else y
    elif op == OP_SIGN:
        r = Float32(1) if x > Float32(0) else (Float32(-1) if x < Float32(0) else Float32(0))
    elif op == OP_LE:
        r = Float32(1) if x <= y else Float32(0)
    elif op == OP_SELECT:
        r = y if x > s else z
    elif op == OP_MUZ:
        # sklearn NMF multiplicative update: x * (y / z), a zero z replaced by s
        r = mul(x, div0(y, z if z != Float32(0) else s))
    return ftz(r)


@always_inline
def bidx(mode: Int, i: Int, d: Int) -> Int:
    """Broadcast index: 0 full, 1 row vector (i % d), 2 column vector (i // d), 3 scalar."""
    if mode == 0:
        return i
    if mode == 1:
        return i % d
    if mode == 2:
        return i // d
    return 0


@always_inline
def gemm_cell(
    a: F32Ptr, b: F32Ptr, i: Int, j: Int, m: Int, k: Int, n: Int, ta: Bool, tb: Bool
) -> Float32:
    """C[i, j] = sum_p op(A)[i, p] op(B)[p, j], p ascending, one fused
    multiply-add per term. A is m x k (k x m when ta), B is k x n (n x k when tb)."""
    var acc = Float32(0)
    for p in range(k):
        var x = a.unsafe_load(p * m + i) if ta else a.unsafe_load(i * k + p)
        var y = b.unsafe_load(j * k + p) if tb else b.unsafe_load(p * n + j)
        acc = ftz(identical_mul_add(ftz(x), ftz(y), acc))
    return acc


@always_inline
def colsum_cell(a: F32Ptr, j: Int, n: Int, d: Int) -> Float32:
    var acc = Float32(0)
    for i in range(n):
        acc = add(acc, a.unsafe_load(i * d + j))
    return acc


@always_inline
def rowsum_cell(a: F32Ptr, i: Int, d: Int) -> Float32:
    var acc = Float32(0)
    for j in range(d):
        acc = add(acc, a.unsafe_load(i * d + j))
    return acc


@always_inline
def sqdist_cell(a: F32Ptr, b: F32Ptr, i: Int, j: Int, d: Int) -> Float32:
    """||a_i - b_j||^2, features ascending."""
    var acc = Float32(0)
    for p in range(d):
        var t = sub(a.unsafe_load(i * d + p), b.unsafe_load(j * d + p))
        acc = ftz(identical_mul_add(t, t, acc))
    return acc


@always_inline
def rand_cell(i: Int, seed: UInt32, stream: UInt32, kind: Int) -> Float32:
    """Counter-based draw i of stream `stream`: Philox4x32-10 at counter
    (i, stream, 0, 0), key (seed, 0x5EED). kind 0: uniform [0, 1) on the
    2^-24 grid; kind 1: standard normal (Box-Muller, cos arm); kind 2:
    Rademacher (+1/-1); kind 3: sparse {-1, 0, +1} ternary with P(+-1) = s
    given as the top 24 bits compared on the grid (handled by the caller via
    kind 0 and an OP_SELECT)."""
    var r = philox4x32_10(
        SIMD[DType.uint32, 4](UInt32(i & 0xFFFFFFFF), stream, UInt32(i >> 32), 0),
        SIMD[DType.uint32, 2](seed, UInt32(0x5EED)),
    )
    var grid = Float32(5.9604644775390625e-08)  # 2^-24
    if kind == 0:
        return mul(Float32(r[0] >> 8), grid)
    if kind == 2:
        return Float32(1) if (r[0] & 1) == 1 else Float32(-1)
    var u1 = mul(Float32((r[0] >> 8) + 1), grid)  # (0, 1]
    var u2 = mul(Float32(r[1] >> 8), grid)
    var rad = sqrt0(mul(Float32(-2), log_floor(u1, Float32(0))))
    var ang = mul(Float32(6.2831854820251465), u2)
    return mul(rad, ftz(identical_cos(ang)))


def cd_row(W: F32Ptr, HHt: F32Ptr, XHt: F32Ptr, perm: I32Ptr, i: Int, k: Int) -> Float32:
    """sklearn `_cdnmf_fast.pyx::_update_cdnmf_fast` for ONE row i of W (the
    rows are independent): components in `perm` order, the gradient summed
    r ascending, the projected-gradient violation of this row returned."""
    var viol = Float32(0)
    for s in range(k):
        var t = Int(perm.unsafe_load(s))
        var grad = -ftz(XHt.unsafe_load(i * k + t))
        for r in range(k):
            grad = ftz(identical_mul_add(ftz(HHt.unsafe_load(t * k + r)), ftz(W.unsafe_load(i * k + r)), grad))
        var w = ftz(W.unsafe_load(i * k + t))
        var pg = grad
        if w == Float32(0):
            pg = grad if grad < Float32(0) else Float32(0)
        viol = add(viol, abs(pg))
        var hess = ftz(HHt.unsafe_load(t * k + t))
        if hess != Float32(0):
            var nw = sub(w, div0(grad, hess))
            W.unsafe_store(i * k + t, nw if nw > Float32(0) else Float32(0))
    return viol


# ------------------------------------------------------------------ serial
# Small dense routines run by ONE thread on the device (a single-thread
# kernel) and by the host loop: the same function body both ways.


def lu_serial(a: F32Ptr, piv: I32Ptr, n: Int, info: F32Ptr):
    """In-place LU with partial pivoting (LAPACK getrf semantics, unblocked):
    the pivot is the largest |a[i, k]| for i >= k, ties broken by the LOWEST
    row index (strict >). L unit-lower below the diagonal, U on and above.
    info[0] = 0 on success, k + 1 at the first exactly-zero pivot."""
    info.unsafe_store(0, Float32(0))
    for k in range(n):
        var p = k
        var best = abs(ftz(a.unsafe_load(k * n + k)))
        for i in range(k + 1, n):
            var v = abs(ftz(a.unsafe_load(i * n + k)))
            if v > best:
                best = v
                p = i
        piv.unsafe_store(k, Int32(p))
        if p != k:
            for j in range(n):
                var t = a.unsafe_load(k * n + j)
                a.unsafe_store(k * n + j, a.unsafe_load(p * n + j))
                a.unsafe_store(p * n + j, t)
        var d = ftz(a.unsafe_load(k * n + k))
        if d == Float32(0):
            if info.unsafe_load(0) == Float32(0):
                info.unsafe_store(0, Float32(k + 1))
            continue
        for i in range(k + 1, n):
            var l = div0(a.unsafe_load(i * n + k), d)
            a.unsafe_store(i * n + k, l)
            for j in range(k + 1, n):
                var v = ftz(identical_mul_add(-l, ftz(a.unsafe_load(k * n + j)), ftz(a.unsafe_load(i * n + j))))
                a.unsafe_store(i * n + j, v)


def lu_solve_serial(lu: F32Ptr, piv: I32Ptr, b: F32Ptr, n: Int, nrhs: Int):
    """getrs: apply the row swaps to B (n x nrhs, row major) in order, then
    forward substitution with unit L and back substitution with U, each
    inner sum ascending."""
    for k in range(n):
        var p = Int(piv.unsafe_load(k))
        if p != k:
            for c in range(nrhs):
                var t = b.unsafe_load(k * nrhs + c)
                b.unsafe_store(k * nrhs + c, b.unsafe_load(p * nrhs + c))
                b.unsafe_store(p * nrhs + c, t)
    for c in range(nrhs):
        for i in range(n):
            var acc = ftz(b.unsafe_load(i * nrhs + c))
            for j in range(i):
                acc = ftz(identical_mul_add(-ftz(lu.unsafe_load(i * n + j)), ftz(b.unsafe_load(j * nrhs + c)), acc))
            b.unsafe_store(i * nrhs + c, acc)
        for ii in range(n):
            var i = n - 1 - ii
            var acc = ftz(b.unsafe_load(i * nrhs + c))
            for j in range(i + 1, n):
                acc = ftz(identical_mul_add(-ftz(lu.unsafe_load(i * n + j)), ftz(b.unsafe_load(j * nrhs + c)), acc))
            b.unsafe_store(i * nrhs + c, div0(acc, lu.unsafe_load(i * n + i)))


def orth_serial(a: F32Ptr, m: Int, l: Int):
    """In-place orthonormalization of the l columns of a row-major m x l
    matrix: modified Gram-Schmidt, each column projected against the
    earlier ones TWICE (MGS2), every dot product rows ascending; a column
    whose remaining norm is 0 becomes 0."""
    for j in range(l):
        for _ in range(2):
            for i in range(j):
                var r = Float32(0)
                for t in range(m):
                    r = ftz(identical_mul_add(ftz(a.unsafe_load(t * l + i)), ftz(a.unsafe_load(t * l + j)), r))
                for t in range(m):
                    a.unsafe_store(
                        t * l + j,
                        ftz(identical_mul_add(-r, ftz(a.unsafe_load(t * l + i)), ftz(a.unsafe_load(t * l + j)))),
                    )
        var nrm = Float32(0)
        for t in range(m):
            var v = ftz(a.unsafe_load(t * l + j))
            nrm = ftz(identical_mul_add(v, v, nrm))
        var s = sqrt0(nrm)
        for t in range(m):
            a.unsafe_store(t * l + j, div0(a.unsafe_load(t * l + j), s))


def chol_serial(a: F32Ptr, n: Int, info: F32Ptr):
    """In-place lower Cholesky (potrf 'L', unblocked, left-looking, sums
    ascending). The upper triangle is zeroed. info[0] = k + 1 at the first
    non-positive pivot (the factor is then not used)."""
    info.unsafe_store(0, Float32(0))
    for j in range(n):
        var acc = ftz(a.unsafe_load(j * n + j))
        for p in range(j):
            var l = ftz(a.unsafe_load(j * n + p))
            acc = ftz(identical_mul_add(-l, l, acc))
        if not (acc > Float32(0)):
            if info.unsafe_load(0) == Float32(0):
                info.unsafe_store(0, Float32(j + 1))
            acc = Float32(1)
        var d = sqrt0(acc)
        a.unsafe_store(j * n + j, d)
        for i in range(j + 1, n):
            var s = ftz(a.unsafe_load(i * n + j))
            for p in range(j):
                s = ftz(identical_mul_add(-ftz(a.unsafe_load(i * n + p)), ftz(a.unsafe_load(j * n + p)), s))
            a.unsafe_store(i * n + j, div0(s, d))
        for i in range(j + 1, n):
            a.unsafe_store(j * n + i, Float32(0))
