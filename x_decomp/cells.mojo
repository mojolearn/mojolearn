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


#: The reduction block of the folds (DEVIATIONS 5300, 5301): a reduction over
#: more than FOLD_BLOCK terms is cut into ceil(len / FOLD_BLOCK) consecutive
#: blocks, each folded ascending by its own thread, and the block partials are
#: then folded ascending. The block count is a function of the SHAPE only
#: (IDENTITY_PATHS row 7's rule), never of the device.
comptime FOLD_BLOCK = 4096


# DEVIATION 5300 (PIN; IDENTITY_PATHS row 130): p ascending inside a block, one
# fused multiply-add per term, the blocks' partials added ascending (FOLD_BLOCK);
# check x_decomp/checks/fold_ew_check.mojo, arm 5300_gemm_order.
@always_inline
def gemm_cell(
    a: F32Ptr, b: F32Ptr, i: Int, j: Int, m: Int, k: Int, n: Int, ta: Bool, tb: Bool
) -> Float32:
    """C[i, j] = sum_p op(A)[i, p] op(B)[p, j] over one block (k <= FOLD_BLOCK).
    A is m x k (k x m when ta), B is k x n (n x k when tb)."""
    return gemm_part_cell(a, b, i, j, m, k, n, ta, tb, 0, k)


@always_inline
def gemm_part_cell(
    a: F32Ptr, b: F32Ptr, i: Int, j: Int, m: Int, k: Int, n: Int, ta: Bool, tb: Bool, p0: Int, p1: Int
) -> Float32:
    """The partial sum of C[i, j] over p in [p0, p1), ascending."""
    var acc = Float32(0)
    for p in range(p0, p1):
        var x = a.unsafe_load(p * m + i) if ta else a.unsafe_load(i * k + p)
        var y = b.unsafe_load(j * k + p) if tb else b.unsafe_load(p * n + j)
        acc = ftz(identical_mul_add(ftz(x), ftz(y), acc))
    return acc


# DEVIATION 5301 (PIN; row 130): rows (columns) ascending, one flushed add each;
# arm 5301_sum_order.
@always_inline
def colsum_cell(a: F32Ptr, j: Int, n: Int, d: Int) -> Float32:
    return colsum_part_cell(a, j, n, d, 0, n)


@always_inline
def colsum_part_cell(a: F32Ptr, j: Int, n: Int, d: Int, r0: Int, r1: Int) -> Float32:
    var acc = Float32(0)
    for i in range(r0, r1):
        acc = add(acc, a.unsafe_load(i * d + j))
    return acc


# DEVIATION 5317 (PIN; row 139): the sign of a vector (sklearn svd_flip,
# _deterministic_vector_sign_flip) is that of its largest-|.| entry, ties to the
# LOWER index; -1 only when that entry is < 0; arm 5317_absmax_tie.
# A long vector is scanned in FOLD_BLOCK slices (`absmax_part_cell`, one GPU
# thread each) and the slices' (max |.|, entry) pairs are folded in slice order
# with the same strict `>` (`absmax_fold_cell`): only comparisons and copies,
# so the entry picked is the serial scan's (the first of the largest |.|), bit
# for bit (lane/algos-decomp 2026-09-28; the one-thread-per-vector scan of a
# 1M-row column was 0.37 s of an lstsq).
@always_inline
def absmax_part_cell(
    a: F32Ptr, t: Int, n: Int, d: Int, by_col: Bool, q0: Int, q1: Int
) -> SIMD[DType.float32, 2]:
    """(largest |.|, that entry) over positions [q0, q1) of vector t, the
    first on a tie; (-1, 0) when no entry is comparable (empty, or NaN)."""
    var best = Float32(-1)
    var val = Float32(0)
    for q in range(q0, q1):
        var v = ftz(a.unsafe_load(q * d + t)) if by_col else ftz(a.unsafe_load(t * d + q))
        if abs(v) > best:
            best = abs(v)
            val = v
    return SIMD[DType.float32, 2](best, val)


@always_inline
def absmax_fold_cell(p: F32Ptr, t: Int, nb: Int, stride: Int) -> Float32:
    """The sign from the slice pairs of vector t (pair b at 2 * (t + b *
    stride)), slices ascending, a later slice winning only on a strictly
    larger |.|."""
    var best = Float32(-1)
    var val = Float32(0)
    for b in range(nb):
        var o = 2 * (t + b * stride)
        var pb = p.unsafe_load(o)
        if pb > best:
            best = pb
            val = p.unsafe_load(o + 1)
    return Float32(-1) if val < Float32(0) else Float32(1)


@always_inline
def absmax_sign_cell(a: F32Ptr, t: Int, n: Int, d: Int, by_col: Bool) -> Float32:
    """by_col: column t of the n x d matrix; else row t."""
    var r = absmax_part_cell(a, t, n, d, by_col, 0, n if by_col else d)
    return Float32(-1) if r[1] < Float32(0) else Float32(1)


@always_inline
def fold_cell(p: F32Ptr, t: Int, nb: Int, stride: Int) -> Float32:
    """The block partials of output t (at t + b * stride), b ascending."""
    var acc = Float32(0)
    for b in range(nb):
        acc = add(acc, p.unsafe_load(t + b * stride))
    return acc


@always_inline
def rowsum_cell(a: F32Ptr, i: Int, d: Int) -> Float32:
    return rowsum_part_cell(a, i, d, 0, d)


@always_inline
def rowsum_part_cell(a: F32Ptr, i: Int, d: Int, c0: Int, c1: Int) -> Float32:
    var acc = Float32(0)
    for j in range(c0, c1):
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


comptime PD_MANHATTAN = 1
comptime PD_CHEBYSHEV = 2
comptime PD_MINKOWSKI = 3
comptime PD_COSINE = 4


# DEVIATION 5319 (PIN; row 130): the non-Euclidean distances (sklearn's
# pairwise metrics for Isomap and MDS): features ascending, t = a - b flushed;
# manhattan sums |t| by IEEE adds, chebyshev keeps the first strict maximum,
# minkowski p sums exp(p log|t|) (|t| = 0 adds nothing) and returns
# exp(log(sum) / p), cosine folds a.b, a.a and b.b by fused multiply-adds and
# returns 1 - a.b / (sqrt(a.a) sqrt(b.b)) clipped to [0, 2] (a zero norm
# counts as 1, sklearn's normalize of a zero row); arm 5319_pdist_order.
def pdist_cell(a: F32Ptr, b: F32Ptr, i: Int, j: Int, d: Int, kind: Int, pw: Float32) -> Float32:
    if kind == PD_COSINE:
        var ab = Float32(0)
        var aa = Float32(0)
        var bb = Float32(0)
        for q in range(d):
            var x = ftz(a.unsafe_load(i * d + q))
            var y = ftz(b.unsafe_load(j * d + q))
            ab = ftz(identical_mul_add(x, y, ab))
            aa = ftz(identical_mul_add(x, x, aa))
            bb = ftz(identical_mul_add(y, y, bb))
        var na = sqrt0(aa)
        var nb = sqrt0(bb)
        if na == Float32(0):
            na = Float32(1)
        if nb == Float32(0):
            nb = Float32(1)
        var r = sub(Float32(1), div0(ab, mul(na, nb)))
        if r < Float32(0):
            r = Float32(0)
        if r > Float32(2):
            r = Float32(2)
        return r
    var acc = Float32(0)
    for q in range(d):
        var t = abs(sub(a.unsafe_load(i * d + q), b.unsafe_load(j * d + q)))
        if kind == PD_MANHATTAN:
            acc = add(acc, t)
        elif kind == PD_CHEBYSHEV:
            if t > acc:
                acc = t
        elif t > Float32(0):
            acc = add(acc, exp_c(mul(pw, log_floor(t, Float32(1.1754943508222875e-38)))))
    if kind == PD_MINKOWSKI:
        if not (acc > Float32(0)):
            return Float32(0)
        return exp_c(div0(log_floor(acc, Float32(1.1754943508222875e-38)), pw))
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
# the exact minimum of sums formed alike); arm 5314_edge_weight. Speed (lane
# decomp-apple): the rows walk compressed arcs with a (distance, index) heap,
# the dense scan's pick, so O(E log n) a row instead of O(n^2) (Isomap at
# 10k rows ran for hours on AMD).
def dijkstra_arc_count(W: F32Ptr, n: Int) -> Int:
    """The number of arcs `dijkstra_arcs` writes: ordered pairs (u, v) where
    W[u, v] or W[v, u] is nonzero."""
    var e = 0
    for u in range(n):
        for v in range(n):
            if W.unsafe_load(u * n + v) != Float32(0) or W.unsafe_load(v * n + u) != Float32(0):
                e += 1
    return e


def dijkstra_arcs(W: F32Ptr, n: Int, rp: I32Ptr, adj: I32Ptr, wa: F32Ptr, wb: F32Ptr):
    """The arcs of the dense graph W (n x n, 0 = no edge) in compressed rows,
    for `dijkstra_row`: node u's arcs are rp[u] .. rp[u + 1] - 1, each to
    adj[e] (ascending) with the RAW weights wa[e] = W[u, v] and wb[e] =
    W[v, u]; the cell applies the undirected rule. Serial, the host side of
    both executors (data movement, no arithmetic)."""
    var e = 0
    for u in range(n):
        rp.unsafe_store(u, Int32(e))
        for v in range(n):
            var x = W.unsafe_load(u * n + v)
            var y = W.unsafe_load(v * n + u)
            if x != Float32(0) or y != Float32(0):
                adj.unsafe_store(e, Int32(v))
                wa.unsafe_store(e, x)
                wb.unsafe_store(e, y)
                e += 1
    rp.unsafe_store(n, Int32(e))


@always_inline
def _dj_less(dist: F32Ptr, base: Int, a: Int, b: Int) -> Bool:
    """Heap order: the smaller distance, ties to the LOWER index (the dense
    scan's pick)."""
    var da = dist.unsafe_load(base + a)
    var db = dist.unsafe_load(base + b)
    return da < db or (da == db and a < b)


def _dj_up(heap: I32Ptr, pos: I32Ptr, dist: F32Ptr, base: Int, hb: Int, at: Int):
    var k = at
    var x = Int(heap.unsafe_load(hb + k))
    while k > 0:
        var p = (k - 1) // 2
        var y = Int(heap.unsafe_load(hb + p))
        if not _dj_less(dist, base, x, y):
            break
        heap.unsafe_store(hb + k, Int32(y))
        pos.unsafe_store(base + y, Int32(k))
        k = p
    heap.unsafe_store(hb + k, Int32(x))
    pos.unsafe_store(base + x, Int32(k))


def _dj_down(heap: I32Ptr, pos: I32Ptr, dist: F32Ptr, base: Int, hb: Int, size: Int):
    var k = 0
    var x = Int(heap.unsafe_load(hb))
    while True:
        var c = 2 * k + 1
        if c >= size:
            break
        var cy = Int(heap.unsafe_load(hb + c))
        if c + 1 < size:
            var c2 = Int(heap.unsafe_load(hb + c + 1))
            if _dj_less(dist, base, c2, cy):
                c = c + 1
                cy = c2
        if not _dj_less(dist, base, cy, x):
            break
        heap.unsafe_store(hb + k, Int32(cy))
        pos.unsafe_store(base + cy, Int32(k))
        k = c
    heap.unsafe_store(hb + k, Int32(x))
    pos.unsafe_store(base + x, Int32(k))


def dijkstra_row(
    rp: I32Ptr, adj: I32Ptr, wa: F32Ptr, wb: F32Ptr, dist: F32Ptr, heap: I32Ptr, pos: I32Ptr,
    i: Int, n: Int, hb: Int,
) -> Float32:
    """Single-source shortest paths from node i on an UNDIRECTED graph
    (scipy `shortest_path(directed=False)`), the arcs of `dijkstra_arcs`:
    edge u-v weighs the smaller nonzero of W[u, v] and W[v, u] (0 = no
    edge). Dijkstra with the next node the unfinished one of least
    distance, ties to the LOWER index (a binary heap on (distance, index):
    the same pick as a scan over every node); an unreachable node keeps -1
    (never inf). dist is row i of the n x n output; pos is row i of an
    n x n int scratch (-1 not queued, -2 finished, else the heap slot);
    heap is n ints from hb. Returns the number of nodes reached."""
    var base = i * n
    for v in range(n):
        dist.unsafe_store(base + v, Float32(-1))
        pos.unsafe_store(base + v, Int32(-1))
    dist.unsafe_store(base + i, Float32(0))
    heap.unsafe_store(hb, Int32(i))
    pos.unsafe_store(base + i, Int32(0))
    var size = 1
    var reached = 0
    while size > 0:
        var u = Int(heap.unsafe_load(hb))
        size -= 1
        pos.unsafe_store(base + u, Int32(-2))
        if size > 0:
            heap.unsafe_store(hb, heap.unsafe_load(hb + size))
            _dj_down(heap, pos, dist, base, hb, size)
        reached += 1
        var best = dist.unsafe_load(base + u)
        for e in range(Int(rp.unsafe_load(u)), Int(rp.unsafe_load(u + 1))):
            var v = Int(adj.unsafe_load(e))
            var pv = Int(pos.unsafe_load(base + v))
            if pv == -2:
                continue
            var a = ftz(wa.unsafe_load(e))
            var b = ftz(wb.unsafe_load(e))
            var w = a
            if w == Float32(0) or (b != Float32(0) and b < w):
                w = b
            if w == Float32(0):
                continue
            var nd = add(best, w)
            var dv = dist.unsafe_load(base + v)
            if dv < Float32(0) or nd < dv:
                dist.unsafe_store(base + v, nd)
                if pv == -1:
                    heap.unsafe_store(hb + size, Int32(v))
                    pos.unsafe_store(base + v, Int32(size))
                    size += 1
                    _dj_up(heap, pos, dist, base, hb, size - 1)
                else:
                    _dj_up(heap, pos, dist, base, hb, pv)
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
    return als_row_solve(X, S, u, f)


def als_row_solve(X: F32Ptr, S: F32Ptr, u: Int, f: Int) -> Float32:
    """`als_row`'s tail on its accumulated scratch (A at u * (f*f + f), b
    after it): the Cholesky (lower, left-looking, sums ascending), then the
    two solves into row u of X. Returns 1 at a non-positive pivot (x_u = 0),
    else 0. Split out so the device's team kernel (x_decomp/device.mojo
    `als_team_kernel`) runs the same tail after a parallel accumulation."""
    var ab = u * (f * f + f)
    var bb = ab + f * f
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


# DEVIATION 5321 (PIN; row 138): implicit's `_least_squares_cg` for one row,
# from the previous factor: every dot product and every YtY row product
# ascending in the factor index, the items ascending; arm 5321_als_cg_order.
comptime ALS_CG_EPS = Float32(1e-20)


@always_inline
def _yt_x(Y: F32Ptr, i: Int, x: F32Ptr, xo: Int, f: Int) -> Float32:
    var acc = Float32(0)
    for j in range(f):
        acc = ftz(identical_mul_add(ftz(Y.unsafe_load(i * f + j)), ftz(x.unsafe_load(xo + j)), acc))
    return acc


@always_inline
def _dot_s(a: F32Ptr, ao: Int, b: F32Ptr, bo: Int, f: Int) -> Float32:
    var acc = Float32(0)
    for j in range(f):
        acc = ftz(identical_mul_add(ftz(a.unsafe_load(ao + j)), ftz(b.unsafe_load(bo + j)), acc))
    return acc


def _yty_times(YtY: F32Ptr, reg: Float32, x: F32Ptr, xo: Int, dst: F32Ptr, d0: Int, f: Int, sign: Float32):
    """dst = sign * (YtY + reg I) x, each row's sum ascending."""
    for j in range(f):
        var acc = Float32(0)
        for l in range(f):
            var a = ftz(YtY.unsafe_load(j * f + l))
            if l == j:
                a = add(a, reg)
            acc = ftz(identical_mul_add(a, ftz(x.unsafe_load(xo + l)), acc))
        dst.unsafe_store(d0 + j, mul(sign, acc))


def als_cg_row(
    C: F32Ptr, Y: F32Ptr, YtY: F32Ptr, X: F32Ptr, S: F32Ptr, u: Int, m: Int, f: Int, reg: Float32, cg_steps: Int
) -> Float32:
    """implicit's `cpu/_als.pyx::_least_squares_cg` for ONE user u, in place
    on X[u] (the previous factor is the start): r = b - A x with A = YtY +
    reg I + sum_i (|c_ui| - 1) y_i y_i^T and b = sum_{c_ui > 0} c_ui y_i
    (items with c_ui != 0 ascending), then `cg_steps` conjugate-gradient
    steps, stopping when r.r < 1e-20. S is per-row scratch of 3 f floats
    (r, p, Ap). Returns the number of steps taken."""
    var ro = u * 3 * f
    var po = ro + f
    var ao = po + f
    var xo = u * f
    _yty_times(YtY, reg, X, xo, S, ro, f, Float32(-1))
    for i in range(m):
        var conf = ftz(C.unsafe_load(u * m + i))
        if conf == Float32(0):
            continue
        var temp = Float32(0)
        if conf > Float32(0):
            temp = conf
        else:
            conf = -conf
        temp = sub(temp, mul(sub(conf, Float32(1)), _yt_x(Y, i, X, xo, f)))
        for j in range(f):
            S.unsafe_store(ro + j, ftz(identical_mul_add(temp, ftz(Y.unsafe_load(i * f + j)), ftz(S.unsafe_load(ro + j)))))
    for j in range(f):
        S.unsafe_store(po + j, S.unsafe_load(ro + j))
    var rsold = _dot_s(S, ro, S, ro, f)
    if rsold < ALS_CG_EPS:
        return Float32(0)
    var steps = 0
    for _ in range(cg_steps):
        steps += 1
        _yty_times(YtY, reg, S, po, S, ao, f, Float32(1))
        for i in range(m):
            var conf = ftz(C.unsafe_load(u * m + i))
            if conf == Float32(0):
                continue
            if conf < Float32(0):
                conf = -conf
            var temp = mul(sub(conf, Float32(1)), _yt_x(Y, i, S, po, f))
            for j in range(f):
                S.unsafe_store(ao + j, ftz(identical_mul_add(temp, ftz(Y.unsafe_load(i * f + j)), ftz(S.unsafe_load(ao + j)))))
        var alpha = div0(rsold, _dot_s(S, po, S, ao, f))
        for j in range(f):
            X.unsafe_store(xo + j, ftz(identical_mul_add(alpha, ftz(S.unsafe_load(po + j)), ftz(X.unsafe_load(xo + j)))))
        for j in range(f):
            S.unsafe_store(ro + j, ftz(identical_mul_add(-alpha, ftz(S.unsafe_load(ao + j)), ftz(S.unsafe_load(ro + j)))))
        var rsnew = _dot_s(S, ro, S, ro, f)
        if rsnew < ALS_CG_EPS:
            break
        var beta = div0(rsnew, rsold)
        for j in range(f):
            S.unsafe_store(po + j, ftz(identical_mul_add(beta, ftz(S.unsafe_load(po + j)), ftz(S.unsafe_load(ro + j)))))
        rsold = rsnew
    return Float32(steps)


# ------------------------------------------------------------------ serial
# Small dense routines run by ONE thread on the device (a single-thread
# kernel) and by the host loop: the same function body both ways.


# DEVIATION 5307 (PIN; row 134): the pivot is the largest |a| with ties to the
# LOWEST row (strict >); arm 5307_pivot_tie.
# DEVIATION 5308 (PIN; row 134): every substitution and Cholesky fold ascending,
# getrs 'N' and 'T' alike ('T' undoes the swaps last to first); arm
# 5308_getrs_order.
def lu_pivot(a: F32Ptr, piv: I32Ptr, k: Int, n: Int):
    """Step k's pivot row: the largest |a[i, k]| for i >= k, ties to the
    LOWEST row (strict >), into piv[k]."""
    var p = k
    var best = abs(ftz(a.unsafe_load(k * n + k)))
    for i in range(k + 1, n):
        var v = abs(ftz(a.unsafe_load(i * n + k)))
        if v > best:
            best = v
            p = i
    piv.unsafe_store(k, Int32(p))


@always_inline
def lu_swap_elem(a: F32Ptr, piv: I32Ptr, k: Int, j: Int, n: Int):
    """Column j of rows k and piv[k] exchanged (nothing when they are one)."""
    var p = Int(piv.unsafe_load(k))
    if p != k:
        var t = a.unsafe_load(k * n + j)
        a.unsafe_store(k * n + j, a.unsafe_load(p * n + j))
        a.unsafe_store(p * n + j, t)


def lu_diag(a: F32Ptr, info: F32Ptr, scal: F32Ptr, k: Int, n: Int):
    """scal = [the pivot d, 1 when step k eliminates else 0]; an exactly-zero
    pivot records info = k + 1 (the first one only) and skips the step."""
    var d = ftz(a.unsafe_load(k * n + k))
    scal.unsafe_store(0, d)
    if d == Float32(0):
        if info.unsafe_load(0) == Float32(0):
            info.unsafe_store(0, Float32(k + 1))
        scal.unsafe_store(1, Float32(0))
    else:
        scal.unsafe_store(1, Float32(1))


@always_inline
def lu_l_elem(a: F32Ptr, scal: F32Ptr, k: Int, i: Int, n: Int):
    """l[i] = a[i, k] / d, stored in place (i > k)."""
    if scal.unsafe_load(1) != Float32(0):
        a.unsafe_store(i * n + k, div0(a.unsafe_load(i * n + k), scal.unsafe_load(0)))


@always_inline
def lu_update_elem(a: F32Ptr, scal: F32Ptr, k: Int, i: Int, j: Int, n: Int):
    """a[i, j] = -l[i] a[k, j] + a[i, j], one fused multiply-add (i, j > k)."""
    if scal.unsafe_load(1) != Float32(0):
        var l = a.unsafe_load(i * n + k)
        a.unsafe_store(i * n + j, ftz(identical_mul_add(-l, ftz(a.unsafe_load(k * n + j)), ftz(a.unsafe_load(i * n + j)))))


def lu_serial(a: F32Ptr, piv: I32Ptr, n: Int, info: F32Ptr):
    """In-place LU with partial pivoting (LAPACK getrf semantics, unblocked):
    the pivot is the largest |a[i, k]| for i >= k, ties broken by the LOWEST
    row index (strict >). L unit-lower below the diagonal, U on and above.
    info[0] = 0 on success, k + 1 at the first exactly-zero pivot. The host
    column runs these cells in this loop; the device the same cells with the
    swap, the multipliers and the trailing update in parallel (each cell of
    the trailing matrix takes step k's one fused multiply-add, steps in
    order; lane/algos-decomp 2026-09-28: one device thread took 7 s at 512)."""
    info.unsafe_store(0, Float32(0))
    var scal = InlineArray[Float32, 2](fill=Float32(0))
    var sp = F32Ptr(unsafe_from_address=Int(scal.unsafe_ptr()))
    for k in range(n):
        lu_pivot(a, piv, k, n)
        for j in range(n):
            lu_swap_elem(a, piv, k, j, n)
        lu_diag(a, info, sp, k, n)
        for i in range(k + 1, n):
            lu_l_elem(a, sp, k, i, n)
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                lu_update_elem(a, sp, k, i, j, n)


def lu_solve_col(lu: F32Ptr, piv: I32Ptr, b: F32Ptr, n: Int, nrhs: Int, trans: Int, c: Int):
    """`lu_solve_serial` for ONE right-hand side column `c` (lane/neural-net-
    experiment, 2026-09-30, the classical pass): the columns of B are
    independent in every statement of the serial solve (each swap, each
    substitution sum reads and writes column c alone), so a thread per
    column runs the same cells in the same order and writes the same bits.
    The device launches one thread per column; the host keeps
    `lu_solve_serial`, whose column loop is this function called c ascending."""
    if trans != 0:
        for i in range(n):
            var acc = ftz(b.unsafe_load(i * nrhs + c))
            for j in range(i):
                acc = ftz(identical_mul_add(-ftz(lu.unsafe_load(j * n + i)), ftz(b.unsafe_load(j * nrhs + c)), acc))
            b.unsafe_store(i * nrhs + c, div0(acc, lu.unsafe_load(i * n + i)))
        for ii in range(n):
            var i = n - 1 - ii
            var acc = ftz(b.unsafe_load(i * nrhs + c))
            for j in range(i + 1, n):
                acc = ftz(identical_mul_add(-ftz(lu.unsafe_load(j * n + i)), ftz(b.unsafe_load(j * nrhs + c)), acc))
            b.unsafe_store(i * nrhs + c, acc)
        for kk in range(n):
            var k = n - 1 - kk
            var p = Int(piv.unsafe_load(k))
            if p != k:
                var t = b.unsafe_load(k * nrhs + c)
                b.unsafe_store(k * nrhs + c, b.unsafe_load(p * nrhs + c))
                b.unsafe_store(p * nrhs + c, t)
        return
    for k in range(n):
        var p = Int(piv.unsafe_load(k))
        if p != k:
            var t = b.unsafe_load(k * nrhs + c)
            b.unsafe_store(k * nrhs + c, b.unsafe_load(p * nrhs + c))
            b.unsafe_store(p * nrhs + c, t)
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


def lu_solve_serial(lu: F32Ptr, piv: I32Ptr, b: F32Ptr, n: Int, nrhs: Int, trans: Int = 0):
    """getrs: apply the row swaps to B (n x nrhs, row major) in order, then
    forward substitution with unit L and back substitution with U, each
    inner sum ascending. trans != 0 solves A^T X = B (getrs 'T'; a real
    matrix's 'C' is the same): forward substitution with U^T, back
    substitution with unit L^T, each inner sum ascending in j, then the row
    swaps in REVERSE order."""
    if trans != 0:
        for c in range(nrhs):
            for i in range(n):
                var acc = ftz(b.unsafe_load(i * nrhs + c))
                for j in range(i):
                    acc = ftz(identical_mul_add(-ftz(lu.unsafe_load(j * n + i)), ftz(b.unsafe_load(j * nrhs + c)), acc))
                b.unsafe_store(i * nrhs + c, div0(acc, lu.unsafe_load(i * n + i)))
            for ii in range(n):
                var i = n - 1 - ii
                var acc = ftz(b.unsafe_load(i * nrhs + c))
                for j in range(i + 1, n):
                    acc = ftz(identical_mul_add(-ftz(lu.unsafe_load(j * n + i)), ftz(b.unsafe_load(j * nrhs + c)), acc))
                b.unsafe_store(i * nrhs + c, acc)
        for kk in range(n):
            var k = n - 1 - kk
            var p = Int(piv.unsafe_load(k))
            if p != k:
                for c in range(nrhs):
                    var t = b.unsafe_load(k * nrhs + c)
                    b.unsafe_store(k * nrhs + c, b.unsafe_load(p * nrhs + c))
                    b.unsafe_store(p * nrhs + c, t)
        return
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


# DEVIATION 5309 (PIN; row 134): the orthonormal basis of a tall A is two
# passes of {R = the Householder QR's R of decomposition/ (qr_factor, TSQR
# slices a function of the shape); orth_rank_guard(R); Q = A R^-1, one thread
# per row}; not LAPACK's orgqr; arm 5309_orth_trsm_order.

#: DEVIATION 5318 (PIN; row 134): a column of A whose residual after the
#: earlier columns is at most 2^-16 of its own norm (R[j, j]^2 <= 2^-32 *
#: sum_t R[t, j]^2) is numerically dependent: its Q column is 0, never the
#: rounding noise A R^-1 divides out of a tiny R[j, j] (that column is neither
#: unit nor orthogonal, and a later one-sided Jacobi SVD of Q^T M rotates it
#: forever; measured: randomized_svd on the `denormal` fixture). 2^-16 sits
#: above float32 QR noise at 4000 rows (sqrt(m) eps = 7.5e-6) and far below
#: an independent column (>= 7e-2 on every fixture). Arm 5318_orth_rank_guard.
comptime ORTH_RANK_TOL2 = Float32(2.3283064365386963e-10)


def orth_rank_guard(R: F32Ptr, l: Int):
    """Zero R[j, j] for every numerically dependent column j (DEVIATION
    5318), so trsm_row's div0 makes its Q column 0. The column is scaled by
    its largest |R[t, j]| first (so no square underflows), the sum of squares
    t ascending."""
    for j in range(l):
        var mx = Float32(0)
        for t in range(j + 1):
            var v = abs(ftz(R.unsafe_load(t * l + j)))
            if v > mx:
                mx = v
        if mx == Float32(0):
            continue
        var nrm = Float32(0)
        for t in range(j + 1):
            var v = ftz(identical_div(ftz(R.unsafe_load(t * l + j)), mx))
            nrm = ftz(identical_mul_add(v, v, nrm))
        var d = ftz(identical_div(ftz(R.unsafe_load(j * l + j)), mx))
        if ftz(identical_mul(d, d)) <= ftz(identical_mul(ORTH_RANK_TOL2, nrm)):
            R.unsafe_store(j * l + j, Float32(0))


def trsm_row(A: F32Ptr, R: F32Ptr, Q: F32Ptr, i: Int, l: Int):
    """Row i of Q = A R^-1 (R upper l x l): q_j = (a_j - sum_{t<j} q_t R[t, j])
    / R[j, j], j ascending, t ascending; a zero diagonal gives 0."""
    for j in range(l):
        var acc = ftz(A.unsafe_load(i * l + j))
        for t in range(j):
            acc = ftz(identical_mul_add(-ftz(Q.unsafe_load(i * l + t)), ftz(R.unsafe_load(t * l + j)), acc))
        Q.unsafe_store(i * l + j, div0(acc, R.unsafe_load(j * l + j)))


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


def chol_diag(a: F32Ptr, info: F32Ptr, j: Int, n: Int):
    """Column step j of `chol_serial`, the diagonal: `acc = a[j, j] - sum_p
    l[j, p]^2` (p ascending, the same fma chain), the info store at the
    first non-positive pivot, and `a[j, j] = sqrt(acc)`. Reads columns
    0..j-1 of row j, final since their own steps. One thread."""
    var acc = ftz(a.unsafe_load(j * n + j))
    for p in range(j):
        var l = ftz(a.unsafe_load(j * n + p))
        acc = ftz(identical_mul_add(-l, l, acc))
    if not (acc > Float32(0)):
        if info.unsafe_load(0) == Float32(0):
            info.unsafe_store(0, Float32(j + 1))
        acc = Float32(1)
    a.unsafe_store(j * n + j, sqrt0(acc))


def chol_col_elem(a: F32Ptr, j: Int, i: Int, n: Int):
    """Column step j of `chol_serial`, row i > j: `a[i, j] = (a[i, j] -
    sum_p a[i, p] a[j, p]) / a[j, j]` (p ascending, the same fma chain,
    the same `div0`), and the mirror cell `a[j, i]` zeroed. Reads columns
    0..j-1 of rows i and j and the diagonal `a[j, j]` (`chol_diag`'s
    store), all final; writes cells no other thread of the step reads.
    One thread per row."""
    var d = a.unsafe_load(j * n + j)
    var s = ftz(a.unsafe_load(i * n + j))
    for p in range(j):
        s = ftz(identical_mul_add(-ftz(a.unsafe_load(i * n + p)), ftz(a.unsafe_load(j * n + p)), s))
    a.unsafe_store(i * n + j, div0(s, d))
    a.unsafe_store(j * n + i, Float32(0))


# DEVIATION 5320 (PIN; row 134): the Householder QR that KEEPS its reflectors
# (LAPACK geqrf, unblocked, dlarfg's sign: beta = -sign(alpha) * ||(alpha, x)||)
# and the explicit Q (orgqr, one column per thread): every norm is the scaled
# sum of squares ascending in the row index, alpha first; every reflector
# product w = v^T c is folded ascending in the row index with v's implicit
# leading 1 first; reflectors are applied to A in order k ascending and to e_j
# in order k descending. Arm 5320_householder_order.
def reflector_norm(a: F32Ptr, k: Int, col: Int, m: Int, n: Int) -> Float32:
    """||(a[k, col], a[k+1, col], ..., a[m-1, col])||, scaled by its largest
    |entry| first so no square overflows or underflows, the sum of squares
    ascending in the row index."""
    var mx = Float32(0)
    for i in range(k, m):
        var v = abs(ftz(a.unsafe_load(i * n + col)))
        if v > mx:
            mx = v
    if mx == Float32(0):
        return Float32(0)
    var acc = Float32(0)
    for i in range(k, m):
        var v = ftz(identical_div(ftz(a.unsafe_load(i * n + col)), mx))
        acc = ftz(identical_mul_add(v, v, acc))
    return ftz(identical_mul(sqrt0(acc), mx))


def geqrf_head(a: F32Ptr, tau: F32Ptr, scal: F32Ptr, k: Int, m: Int, n: Int):
    """Step k's reflector (dlarfg): tau[k], beta on the diagonal, and in
    `scal` [the divisor alpha - beta, 1 when the step acts else 0]. A column
    whose sub-diagonal part is exactly zero gets tau = 0 (H = I), as dlarfg."""
    var alpha = ftz(a.unsafe_load(k * n + k))
    var xmax = Float32(0)
    for i in range(k + 1, m):
        var v = abs(ftz(a.unsafe_load(i * n + k)))
        if v > xmax:
            xmax = v
    if xmax == Float32(0):
        tau.unsafe_store(k, Float32(0))
        scal.unsafe_store(0, Float32(1))
        scal.unsafe_store(1, Float32(0))
        return
    var nrm = reflector_norm(a, k, k, m, n)
    var beta = -nrm if alpha >= Float32(0) else nrm
    tau.unsafe_store(k, div0(sub(beta, alpha), beta))
    scal.unsafe_store(0, sub(alpha, beta))
    scal.unsafe_store(1, Float32(1))
    a.unsafe_store(k * n + k, beta)


@always_inline
def geqrf_scale_elem(a: F32Ptr, scal: F32Ptr, k: Int, i: Int, n: Int):
    """v[i] = a[i, k] / (alpha - beta), i > k, when step k acts."""
    if scal.unsafe_load(1) != Float32(0):
        a.unsafe_store(i * n + k, div0(a.unsafe_load(i * n + k), scal.unsafe_load(0)))


@always_inline
def geqrf_dot(a: F32Ptr, scal: F32Ptr, k: Int, j: Int, m: Int, n: Int) -> Float32:
    """w = v^T a[k:, j], v's implicit leading 1 first, rows ascending."""
    if scal.unsafe_load(1) == Float32(0):
        return Float32(0)
    var w = ftz(a.unsafe_load(k * n + j))
    for i in range(k + 1, m):
        w = ftz(identical_mul_add(ftz(a.unsafe_load(i * n + k)), ftz(a.unsafe_load(i * n + j)), w))
    return w


@always_inline
def geqrf_update_elem(a: F32Ptr, tau: F32Ptr, scal: F32Ptr, k: Int, i: Int, j: Int, n: Int, w: Float32):
    """a[i, j] -= tau v[i] w (i >= k, j > k) when step k acts: row k the
    plain subtraction (v[k] = 1), the rows below one fused multiply-add."""
    if scal.unsafe_load(1) == Float32(0):
        return
    var tw = ftz(identical_mul(tau.unsafe_load(k), w))
    if i == k:
        a.unsafe_store(k * n + j, sub(a.unsafe_load(k * n + j), tw))
    else:
        a.unsafe_store(i * n + j, ftz(identical_mul_add(-tw, ftz(a.unsafe_load(i * n + k)), ftz(a.unsafe_load(i * n + j)))))


def geqrf_serial(a: F32Ptr, tau: F32Ptr, m: Int, n: Int):
    """In-place Householder QR of the row-major m x n A (geqrf semantics,
    unblocked): for k < min(m, n), dlarfg makes H_k = I - tau_k v v^T with
    v[k] = 1 implicit and v[k+1:] stored below the diagonal, beta on it; R is
    the upper triangle. The host column runs these cells in this loop; the
    device runs the same cells with the rows (scale, update) and the columns
    (dot) in parallel, every fold still one thread ascending (lane/algos-decomp
    2026-09-28: one device thread took 217 s at 1M x 28)."""
    var kk = m if m < n else n
    var scal = InlineArray[Float32, 2](fill=Float32(0))
    var sp = F32Ptr(unsafe_from_address=Int(scal.unsafe_ptr()))
    for k in range(kk):
        geqrf_head(a, tau, sp, k, m, n)
        for i in range(k + 1, m):
            geqrf_scale_elem(a, sp, k, i, n)
        for j in range(k + 1, n):
            var w = geqrf_dot(a, sp, k, j, m, n)
            for i in range(k, m):
                geqrf_update_elem(a, tau, sp, k, i, j, n, w)


@always_inline
def orgqr_init_elem(q: F32Ptr, i: Int, j: Int, qc: Int):
    q.unsafe_store(i * qc + j, Float32(1) if i == j else Float32(0))


@always_inline
def orgqr_dot(h: F32Ptr, tau: F32Ptr, q: F32Ptr, k: Int, j: Int, m: Int, n: Int, qc: Int) -> Float32:
    """w = v_k^T q[k:, j], the implicit 1 first, rows ascending (0 when H_k = I)."""
    if ftz(tau.unsafe_load(k)) == Float32(0):
        return Float32(0)
    var w = ftz(q.unsafe_load(k * qc + j))
    for i in range(k + 1, m):
        w = ftz(identical_mul_add(ftz(h.unsafe_load(i * n + k)), ftz(q.unsafe_load(i * qc + j)), w))
    return w


@always_inline
def orgqr_update_elem(h: F32Ptr, tau: F32Ptr, q: F32Ptr, k: Int, i: Int, j: Int, n: Int, qc: Int, w: Float32):
    """q[i, j] -= tau v[i] w (i >= k) when H_k acts."""
    var t = ftz(tau.unsafe_load(k))
    if t == Float32(0):
        return
    var tw = ftz(identical_mul(t, w))
    if i == k:
        q.unsafe_store(k * qc + j, sub(q.unsafe_load(k * qc + j), tw))
    else:
        q.unsafe_store(i * qc + j, ftz(identical_mul_add(-tw, ftz(h.unsafe_load(i * n + k)), ftz(q.unsafe_load(i * qc + j)))))


def orgqr_col(h: F32Ptr, tau: F32Ptr, q: F32Ptr, j: Int, m: Int, n: Int, kk: Int, qc: Int):
    """Column j of Q = H_0 H_1 ... H_{kk-1} (m x qc, row major): e_j with the
    reflectors of the m x n factored `h` applied last to first. The column is
    built in place in q (the host column; the device runs the same cells
    with the rows in parallel)."""
    for i in range(m):
        orgqr_init_elem(q, i, j, qc)
    for r in range(kk):
        var k = kk - 1 - r
        var w = orgqr_dot(h, tau, q, k, j, m, n, qc)
        for i in range(k, m):
            orgqr_update_elem(h, tau, q, k, i, j, n, qc, w)
