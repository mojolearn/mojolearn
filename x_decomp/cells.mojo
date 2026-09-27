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
comptime OP_LGAMMA = 37


@always_inline
def mul(a: Float32, b: Float32) -> Float32:
    return ftz(identical_mul(ftz(a), ftz(b)))


# DEVIATION 5304 (PIN; row 131, row 10's policy): every operand and result
# through ftz; arm 5304_add_flush.
@always_inline
def add(a: Float32, b: Float32) -> Float32:
    return ftz(ftz(a) + ftz(b))


@always_inline
def sub(a: Float32, b: Float32) -> Float32:
    return ftz(ftz(a) - ftz(b))


# DEVIATION 5303 (REPLACE; row 131, Clause B): a zero divisor yields 0, a log
# argument is floored, exp is clamped, sqrt of a non-positive is 0: no computed
# NaN or inf reaches an output; arm 5303_zero_guard.
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


# DEVIATION 5305 (REPLACE; row 132): digamma and lgamma as the recurrence to 6
# then the asymptotic series, in the pinned primitives (scipy calls Cephes);
# arm 5305_series_recurrence.
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


def lgamma(a: Float32) -> Float32:
    """log Gamma(x) for x > 0 (0 otherwise): the recurrence up to 6 (the
    logs of the shifted factors summed ascending), then Stirling's series
    (x - 1/2) log x - x + log(2 pi)/2 + 1/(12x) - 1/(360x^3) + 1/(1260x^5)."""
    var x = ftz(a)
    if not (x > Float32(0)):
        return Float32(0)
    var shift = Float32(0)
    while x < Float32(6):
        shift = add(shift, log_floor(x, Float32(0)))
        x = add(x, Float32(1))
    var inv = div0(Float32(1), x)
    var inv2 = mul(inv, inv)
    var t = mul(inv2, Float32(0.0007936507936507937))
    t = sub(Float32(0.002777777777777778), t)
    t = mul(inv2, t)
    t = sub(Float32(0.08333333333333333), t)
    t = mul(inv, t)
    var r = mul(sub(x, Float32(0.5)), log_floor(x, Float32(0)))
    r = sub(r, x)
    r = add(r, Float32(0.9189385332046727))
    r = add(r, t)
    return sub(r, shift)


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
    elif op == OP_LGAMMA:
        r = lgamma(x)
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


# DEVIATION 5300 (PIN; IDENTITY_PATHS row 130): p ascending, one fused multiply-add
# per term; check x_decomp/checks/fold_ew_check.mojo, arm 5300_gemm_order.
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


# DEVIATION 5301 (PIN; row 130): rows (columns) ascending, one flushed add each;
# arm 5301_sum_order.
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


# DEVIATION 5302 (PIN; row 130): features ascending, t = a - b flushed, one fused
# multiply-add per term; arm 5302_sqdist_order.
@always_inline
def sqdist_cell(a: F32Ptr, b: F32Ptr, i: Int, j: Int, d: Int) -> Float32:
    """||a_i - b_j||^2, features ascending."""
    var acc = Float32(0)
    for p in range(d):
        var t = sub(a.unsafe_load(i * d + p), b.unsafe_load(j * d + p))
        acc = ftz(identical_mul_add(t, t, acc))
    return acc


# DEVIATION 5306 (REPLACE; row 133): draws are Philox4x32-10 at a counter, the
# uniform (r0 >> 8) 2^-24, the normal Box-Muller's cos arm, the Gamma
# Marsaglia-Tsang with per-attempt counters; numpy's generators are not
# reproduced; arm 5306_draw_mapping.
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


# DEVIATION 5310 (PIN; row 135): NMF CD per row, components in `perm` order,
# gradients r ascending; arm 5310_cd_order.
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


# DEVIATION 5311 (PIN; row 135): Lasso CD per row, coordinates ascending, the
# stop a sweep's max update <= tol max|w| (not sklearn's duality gap);
# arm 5311_lasso_order.
def lasso_row(
    G: F32Ptr, Q: F32Ptr, W: F32Ptr, H: F32Ptr, i: Int, k: Int, alpha: Float32, max_iter: Int,
    tol: Float32, positive: Bool,
) -> Float32:
    """sklearn `_cd_fast.pyx::enet_coordinate_descent_gram` (beta 0) for ONE
    row i: minimize 1/2 ||x - D^T w||^2 + alpha ||w||_1 given the Gram
    G = D D^T (k x k) and q = D x (row i of Q, n x k); w (row i of W) is the
    warm start and the answer; H (row i) is scratch holding G w. Coordinates
    ascending; a sweep whose largest update is at most tol times the largest
    |w| (or leaves every w at 0) ends it. Returns the sweeps run."""
    var base = i * k
    for j in range(k):
        var acc = Float32(0)
        for l in range(k):
            acc = ftz(identical_mul_add(ftz(G.unsafe_load(j * k + l)), ftz(W.unsafe_load(base + l)), acc))
        H.unsafe_store(base + j, acc)
    var it = 0
    for _ in range(max_iter):
        it += 1
        var w_max = Float32(0)
        var d_w_max = Float32(0)
        for j in range(k):
            var gjj = ftz(G.unsafe_load(j * k + j))
            if gjj == Float32(0):
                continue
            var w_j = ftz(W.unsafe_load(base + j))
            if w_j != Float32(0):
                for l in range(k):
                    H.unsafe_store(base + l, ftz(identical_mul_add(-w_j, ftz(G.unsafe_load(l * k + j)), ftz(H.unsafe_load(base + l)))))
            var tmp = sub(Q.unsafe_load(base + j), H.unsafe_load(base + j))
            var nw = Float32(0)
            if positive:
                if tmp > Float32(0):
                    var m = sub(tmp, alpha)
                    nw = div0(m, gjj) if m > Float32(0) else Float32(0)
            else:
                var m = sub(abs(tmp), alpha)
                if m > Float32(0):
                    nw = div0(m if tmp > Float32(0) else -m, gjj)
            W.unsafe_store(base + j, nw)
            if nw != Float32(0):
                for l in range(k):
                    H.unsafe_store(base + l, ftz(identical_mul_add(nw, ftz(G.unsafe_load(l * k + j)), ftz(H.unsafe_load(base + l)))))
            var dw = abs(sub(nw, w_j))
            if dw > d_w_max:
                d_w_max = dw
            if abs(nw) > w_max:
                w_max = abs(nw)
        if w_max == Float32(0) or d_w_max <= mul(tol, w_max):
            break
    return Float32(it)


# DEVIATION 5312 (PIN; row 135): OMP's atom is the largest |correlation|, ties
# to the LOWER atom; the refit a Cholesky rebuilt row by row; arm 5312_omp_tie.
def omp_row(G: F32Ptr, Q: F32Ptr, W: F32Ptr, S: F32Ptr, i: Int, k: Int, nnz: Int) -> Float32:
    """Orthogonal matching pursuit on the Gram (sklearn `_omp.py::_gram_omp`,
    tol None) for ONE row: greedily add the atom with the largest |Xy - G_S
    gamma| (ties to the LOWER index; stop if it is already active or its
    square is under float32 eps), refit gamma on the active set by the
    Cholesky of G[S, S] (sums ascending), until nnz atoms. W's row gets gamma
    scattered to the active atoms. S is per-row scratch of k*k + 3k floats.
    Returns the number of active atoms."""
    var sb = i * (k * k + 3 * k)
    var L = sb                    # k x k Cholesky of G[S, S]
    var act = sb + k * k          # active indices (as floats, exact)
    var gam = act + k             # gamma
    var tmp = gam + k             # triangular-solve scratch
    var base = i * k
    for j in range(k):
        W.unsafe_store(base + j, Float32(0))
    var na = 0
    var eps = Float32(1.1920928955078125e-07)
    while na < nnz:
        # residual correlations alpha = Xy - G[:, S] gamma
        var lam = 0
        var best = Float32(-1)
        for j in range(k):
            var acc = ftz(Q.unsafe_load(base + j))
            for t in range(na):
                var a = Int(S.unsafe_load(act + t))
                acc = ftz(identical_mul_add(-ftz(G.unsafe_load(j * k + a)), ftz(S.unsafe_load(gam + t)), acc))
            if abs(acc) > best:
                best = abs(acc)
                lam = j
        var already = False
        for t in range(na):
            if Int(S.unsafe_load(act + t)) == lam:
                already = True
        if already or mul(best, best) < eps:
            break
        # extend the Cholesky factor by one row
        for t in range(na):
            var a = Int(S.unsafe_load(act + t))
            var acc = ftz(G.unsafe_load(lam * k + a))
            for u in range(t):
                acc = ftz(identical_mul_add(-ftz(S.unsafe_load(L + na * k + u)), ftz(S.unsafe_load(L + t * k + u)), acc))
            S.unsafe_store(L + na * k + t, div0(acc, S.unsafe_load(L + t * k + t)))
        var v = Float32(0)
        for t in range(na):
            var x = ftz(S.unsafe_load(L + na * k + t))
            v = ftz(identical_mul_add(x, x, v))
        var lkk = sub(G.unsafe_load(lam * k + lam), v)
        if not (lkk > eps):
            break
        S.unsafe_store(L + na * k + na, sqrt0(lkk))
        S.unsafe_store(act + na, Float32(lam))
        na += 1
        # gamma = (L L^T)^-1 Xy[S]: forward then back substitution
        for t in range(na):
            var acc = ftz(Q.unsafe_load(base + Int(S.unsafe_load(act + t))))
            for u in range(t):
                acc = ftz(identical_mul_add(-ftz(S.unsafe_load(L + t * k + u)), ftz(S.unsafe_load(tmp + u)), acc))
            S.unsafe_store(tmp + t, div0(acc, S.unsafe_load(L + t * k + t)))
        for tt in range(na):
            var t = na - 1 - tt
            var acc = ftz(S.unsafe_load(tmp + t))
            for u in range(t + 1, na):
                acc = ftz(identical_mul_add(-ftz(S.unsafe_load(L + u * k + t)), ftz(S.unsafe_load(gam + u)), acc))
            S.unsafe_store(gam + t, div0(acc, S.unsafe_load(L + t * k + t)))
    for t in range(na):
        W.unsafe_store(base + Int(S.unsafe_load(act + t)), S.unsafe_load(gam + t))
    return Float32(na)


def gamma_cell(i: Int, seed: UInt32, stream: UInt32, shape: Float32) -> Float32:
    """Gamma(shape, 1) for shape >= 1 by Marsaglia and Tsang (2000): attempt
    j draws a normal and a uniform from Philox counter (i, stream, j); the
    first accepted attempt is the answer (at most 64; the last attempt's
    proposal is returned if none is, which at shape 100 has probability
    below 1e-80)."""
    var d = sub(shape, Float32(0.3333333333333333))
    var c = div0(Float32(1), sqrt0(mul(Float32(9), d)))
    var grid = Float32(5.9604644775390625e-08)
    var last = d
    for j in range(64):
        var r = philox4x32_10(
            SIMD[DType.uint32, 4](UInt32(i & 0xFFFFFFFF), stream, UInt32(j), UInt32(0x6A33)),
            SIMD[DType.uint32, 2](seed, UInt32(0x5EED)),
        )
        var u1 = mul(Float32((r[0] >> 8) + 1), grid)
        var u2 = mul(Float32(r[1] >> 8), grid)
        var x = mul(sqrt0(mul(Float32(-2), log_floor(u1, Float32(0)))), ftz(identical_cos(mul(Float32(6.2831854820251465), u2))))
        var v = add(Float32(1), mul(c, x))
        if not (v > Float32(0)):
            continue
        v = mul(mul(v, v), v)
        var u = mul(Float32((r[2] >> 8) + 1), grid)
        last = mul(d, v)
        var x2 = mul(x, x)
        if u < sub(Float32(1), mul(Float32(0.0331), mul(x2, x2))):
            return last
        var rhs = add(mul(Float32(0.5), x2), mul(d, add(sub(Float32(1), v), log_floor(v, Float32(0)))))
        if log_floor(u, Float32(0)) < rhs:
            return last
    return last


# DEVIATION 5313 (PIN; row 136): one thread per document, both word folds
# ascending, the stop mean |change| < tol; arm 5313_lda_order.
def lda_doc_row(
    X: F32Ptr, EW: F32Ptr, D: F32Ptr, E: F32Ptr, S: F32Ptr, i: Int, k: Int, v: Int, prior: Float32,
    max_iter: Int, tol: Float32,
) -> Float32:
    """sklearn `_lda.py::_update_doc_distribution` for ONE document i (row i
    of X, n x v): doc_topic D (n x k) and its exp-Dirichlet expectation E
    (n x k) are the start and the answer, EW (k x v) is exp(E[log beta]).
    Per iteration: norm_phi_w = sum_t E_t EW_tw + EPS (t ascending); new
    D_t = E_t * sum_w (X_w / norm_phi_w) EW_tw (w ascending, zero counts
    skipped); `_dirichlet_expectation_1d` (D_t += prior; E_t = exp(psi(D_t)
    - psi(sum D))); stop when mean |last - D| < tol. S is per-row scratch of
    v + k floats. Returns the iterations run."""
    var base = i * k
    var sb = i * (v + k)
    var eps = Float32(2.220446049250313e-16)
    var it = 0
    for _ in range(max_iter):
        it += 1
        for t in range(k):
            S.unsafe_store(sb + v + t, D.unsafe_load(base + t))
        for w in range(v):
            var xw = ftz(X.unsafe_load(i * v + w))
            if xw == Float32(0):
                continue
            var acc = Float32(0)
            for t in range(k):
                acc = ftz(identical_mul_add(ftz(E.unsafe_load(base + t)), ftz(EW.unsafe_load(t * v + w)), acc))
            S.unsafe_store(sb + w, div0(xw, add(acc, eps)))
        var total = Float32(0)
        for t in range(k):
            var acc = Float32(0)
            for w in range(v):
                if ftz(X.unsafe_load(i * v + w)) == Float32(0):
                    continue
                acc = ftz(identical_mul_add(ftz(S.unsafe_load(sb + w)), ftz(EW.unsafe_load(t * v + w)), acc))
            var dt = add(mul(E.unsafe_load(base + t), acc), prior)
            D.unsafe_store(base + t, dt)
            total = add(total, dt)
        var psi_total = digamma(total)
        var change = Float32(0)
        for t in range(k):
            var dt = D.unsafe_load(base + t)
            E.unsafe_store(base + t, exp_c(sub(digamma(dt), psi_total)))
            change = add(change, abs(sub(S.unsafe_load(sb + v + t), dt)))
        if div0(change, Float32(k)) < tol:
            break
    return Float32(it)


# DEVIATION 5314 (PIN; row 137): an undirected edge weighs the smaller nonzero
# of W[u, v] and W[v, u]; the visiting order cannot reach a distance (each is
# the exact minimum of sums formed alike); arm 5314_edge_weight.
def dijkstra_row(W: F32Ptr, dist: F32Ptr, done: F32Ptr, i: Int, n: Int) -> Float32:
    """Single-source shortest paths from node i on a dense UNDIRECTED graph
    (scipy `shortest_path(directed=False)`): W (n x n) holds edge weights,
    0 meaning no edge, and edge u-v weighs the smaller nonzero of W[u, v] and
    W[v, u]. Dijkstra with the next node the unfinished one of least
    distance, ties to the LOWER index; an unreachable node keeps -1 (never
    inf). dist and done are row i of n x n outputs/scratch. Returns the
    number of nodes reached."""
    var base = i * n
    for v in range(n):
        dist.unsafe_store(base + v, Float32(-1))
        done.unsafe_store(base + v, Float32(0))
    dist.unsafe_store(base + i, Float32(0))
    var reached = 0
    for _ in range(n):
        var u = -1
        var best = Float32(0)
        for v in range(n):
            if done.unsafe_load(base + v) != Float32(0):
                continue
            var dv = dist.unsafe_load(base + v)
            if dv < Float32(0):
                continue
            if u < 0 or dv < best:
                u = v
                best = dv
        if u < 0:
            break
        done.unsafe_store(base + u, Float32(1))
        reached += 1
        for v in range(n):
            if done.unsafe_load(base + v) != Float32(0):
                continue
            var a = ftz(W.unsafe_load(u * n + v))
            var b = ftz(W.unsafe_load(v * n + u))
            var w = a
            if w == Float32(0) or (b != Float32(0) and b < w):
                w = b
            if w == Float32(0):
                continue
            var nd = add(best, w)
            var dv = dist.unsafe_load(base + v)
            if dv < Float32(0) or nd < dv:
                dist.unsafe_store(base + v, nd)
    return Float32(reached)


# DEVIATION 5315 (PIN; row 137): G += reg * trace(G) I (reg when the trace is
# 0), the Cholesky solve ascending, the weights normalized; arm 5315_bary_reg.
def barycenter_row(
    X: F32Ptr, Y: F32Ptr, nbr: F32Ptr, Wt: F32Ptr, S: F32Ptr, i: Int, d: Int, k: Int, reg: Float32
) -> Float32:
    """sklearn `_locally_linear.py::barycenter_weights` for ONE query row i
    of X (n x d) against its k neighbors in Y (indices in row i of nbr,
    stored as exact floats): Z = Y[nbr] - x, G = Z Z^T (sums over features
    ascending), G += R I with R = reg * trace(G) (reg when the trace is 0),
    w = G^-1 1 by the Cholesky of G, w /= sum(w). S is per-row scratch of
    k*k + k*d floats. Returns 0, or 1 if the Cholesky met a non-positive
    pivot (its weights are then the uniform 1/k)."""
    var zb = i * (k * k + k * d)
    var gb = zb + k * d
    for a in range(k):
        var ya = Int(nbr.unsafe_load(i * k + a))
        for f in range(d):
            S.unsafe_store(zb + a * d + f, sub(Y.unsafe_load(ya * d + f), X.unsafe_load(i * d + f)))
    var trace = Float32(0)
    for a in range(k):
        for b in range(k):
            var acc = Float32(0)
            for f in range(d):
                acc = ftz(identical_mul_add(ftz(S.unsafe_load(zb + a * d + f)), ftz(S.unsafe_load(zb + b * d + f)), acc))
            S.unsafe_store(gb + a * k + b, acc)
        trace = add(trace, S.unsafe_load(gb + a * k + a))
    var R = mul(reg, trace) if trace > Float32(0) else reg
    for a in range(k):
        S.unsafe_store(gb + a * k + a, add(S.unsafe_load(gb + a * k + a), R))
    var bad = False
    # lower Cholesky in place (left-looking, sums ascending)
    for j in range(k):
        var acc = ftz(S.unsafe_load(gb + j * k + j))
        for p in range(j):
            var l = ftz(S.unsafe_load(gb + j * k + p))
            acc = ftz(identical_mul_add(-l, l, acc))
        if not (acc > Float32(0)):
            bad = True
            acc = Float32(1)
        var dj = sqrt0(acc)
        S.unsafe_store(gb + j * k + j, dj)
        for r in range(j + 1, k):
            var s = ftz(S.unsafe_load(gb + r * k + j))
            for p in range(j):
                s = ftz(identical_mul_add(-ftz(S.unsafe_load(gb + r * k + p)), ftz(S.unsafe_load(gb + j * k + p)), s))
            S.unsafe_store(gb + r * k + j, div0(s, dj))
    if bad:
        for a in range(k):
            Wt.unsafe_store(i * k + a, div0(Float32(1), Float32(k)))
        return Float32(1)
    # solve L y = 1, then L^T w = y (w in Wt's row)
    for a in range(k):
        var acc = Float32(1)
        for p in range(a):
            acc = ftz(identical_mul_add(-ftz(S.unsafe_load(gb + a * k + p)), ftz(Wt.unsafe_load(i * k + p)), acc))
        Wt.unsafe_store(i * k + a, div0(acc, S.unsafe_load(gb + a * k + a)))
    for aa in range(k):
        var a = k - 1 - aa
        var acc = ftz(Wt.unsafe_load(i * k + a))
        for p in range(a + 1, k):
            acc = ftz(identical_mul_add(-ftz(S.unsafe_load(gb + p * k + a)), ftz(Wt.unsafe_load(i * k + p)), acc))
        Wt.unsafe_store(i * k + a, div0(acc, S.unsafe_load(gb + a * k + a)))
    var tot = Float32(0)
    for a in range(k):
        tot = add(tot, Wt.unsafe_load(i * k + a))
    for a in range(k):
        Wt.unsafe_store(i * k + a, div0(Wt.unsafe_load(i * k + a), tot))
    return Float32(0)


# DEVIATION 5316 (PIN; row 138): per row, items ascending into A and b, the
# Cholesky solve (posv) ascending; arm 5316_als_order.
def als_row(
    C: F32Ptr, Y: F32Ptr, YtY: F32Ptr, X: F32Ptr, S: F32Ptr, u: Int, m: Int, f: Int, reg: Float32
) -> Float32:
    """implicit's `cpu/_als.pyx::least_squares` for ONE user u: A = YtY +
    reg I + sum_i (c_ui - 1) y_i y_i^T and b = sum_{c_ui > 0} c_ui y_i over
    the items i with c_ui != 0 (ascending; a negative confidence enters A
    with its magnitude and b not at all), then x_u = A^-1 b by Cholesky
    (posv). C is the dense n x m confidence matrix (0 = no interaction), Y
    the m x f item factors. S is per-row scratch of f*f + f floats. Returns
    1 if the Cholesky met a non-positive pivot (x_u is then 0), else 0."""
    var ab = u * (f * f + f)
    var bb = ab + f * f
    for j in range(f * f):
        S.unsafe_store(ab + j, YtY.unsafe_load(j))
    for j in range(f):
        S.unsafe_store(ab + j * f + j, add(S.unsafe_load(ab + j * f + j), reg))
        S.unsafe_store(bb + j, Float32(0))
    for i in range(m):
        var conf = ftz(C.unsafe_load(u * m + i))
        if conf == Float32(0):
            continue
        if conf > Float32(0):
            for j in range(f):
                S.unsafe_store(bb + j, ftz(identical_mul_add(conf, ftz(Y.unsafe_load(i * f + j)), ftz(S.unsafe_load(bb + j)))))
        else:
            conf = -conf
        var cm1 = sub(conf, Float32(1))
        for j in range(f):
            var t = mul(cm1, Y.unsafe_load(i * f + j))
            for l in range(f):
                S.unsafe_store(ab + j * f + l, ftz(identical_mul_add(t, ftz(Y.unsafe_load(i * f + l)), ftz(S.unsafe_load(ab + j * f + l)))))
    # Cholesky (lower, left-looking, sums ascending), then the two solves
    for j in range(f):
        var acc = ftz(S.unsafe_load(ab + j * f + j))
        for p in range(j):
            var l = ftz(S.unsafe_load(ab + j * f + p))
            acc = ftz(identical_mul_add(-l, l, acc))
        if not (acc > Float32(0)):
            for q in range(f):
                X.unsafe_store(u * f + q, Float32(0))
            return Float32(1)
        var dj = sqrt0(acc)
        S.unsafe_store(ab + j * f + j, dj)
        for r in range(j + 1, f):
            var s = ftz(S.unsafe_load(ab + r * f + j))
            for p in range(j):
                s = ftz(identical_mul_add(-ftz(S.unsafe_load(ab + r * f + p)), ftz(S.unsafe_load(ab + j * f + p)), s))
            S.unsafe_store(ab + r * f + j, div0(s, dj))
    for a in range(f):
        var acc = ftz(S.unsafe_load(bb + a))
        for p in range(a):
            acc = ftz(identical_mul_add(-ftz(S.unsafe_load(ab + a * f + p)), ftz(S.unsafe_load(bb + p)), acc))
        S.unsafe_store(bb + a, div0(acc, S.unsafe_load(ab + a * f + a)))
    for aa in range(f):
        var a = f - 1 - aa
        var acc = ftz(S.unsafe_load(bb + a))
        for p in range(a + 1, f):
            acc = ftz(identical_mul_add(-ftz(S.unsafe_load(ab + p * f + a)), ftz(S.unsafe_load(bb + p)), acc))
        S.unsafe_store(bb + a, div0(acc, S.unsafe_load(ab + a * f + a)))
    for q in range(f):
        X.unsafe_store(u * f + q, S.unsafe_load(bb + q))
    return Float32(0)


# ------------------------------------------------------------------ serial
# Small dense routines run by ONE thread on the device (a single-thread
# kernel) and by the host loop: the same function body both ways.


# DEVIATION 5307 (PIN; row 134): the pivot is the largest |a| with ties to the
# LOWEST row (strict >); arm 5307_pivot_tie.
# DEVIATION 5308 (PIN; row 134): every substitution and Cholesky fold ascending;
# arm 5308_getrs_order.
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


# DEVIATION 5309 (PIN; row 134): MGS with exactly two projection passes per
# column (not LAPACK geqrf's Householder Q); arm 5309_mgs_passes.
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
