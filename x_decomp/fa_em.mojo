# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""FactorAnalysis's EM loop in Mojo (lane py-runtime-b, 2026-10-05): the
`for it in range(1, max_iter + 1)` of `_expansion_decomp.FactorAnalysis.fit`
(main's route; the FAST Apple `_fit_fast` keeps its own entries), statement
for statement on `Kit[E]`: the same cells, broadcast modes and float32
scalars, the SVD of the scaled R (n >= d: `Kit.svd`, values descending with
ties to the lower index, `_Kit.svd`'s order) or the eigh of the scaled Gram
(n < d, columns reversed), and Python's float64 log-likelihood (`_dsum`'s
sequential IEEE adds of the float32 words, in ascending order) and its
stopping test in the same order. So the IDENTICAL words are the Python
driver's on every column; x_decomp/fa_em_dev.mojo runs the same statements
on resident device matrices."""
from std.math import inf, sqrt
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from x_decomp.api import _f, _n
from x_decomp.exec_trait import Exec
from x_decomp.kit import (
    Kit, Mat, OP_ADD, OP_ADDS, OP_DIV, OP_LOGS, OP_MAXS, OP_MINS, OP_MUL, OP_SCALE, OP_SQ, OP_SQRT,
    mat_const, mat_from, mat_rows, mat_t, mat_vec_t,
)
from x_decomp.moves import order_f

comptime FA_SMALL: Float64 = 1e-12
comptime FA_LOG_FLOOR: Float64 = 1.1754943508222875e-38


def dsum(A: Mat, a: Int, b: Int) -> Float64:
    """`_dsum(A.s[a:b])`: sequential binary64 adds, ascending, from 0.0."""
    var t = 0.0
    for i in range(a, b):  # small-loop(b: features): the d float32 words of one log-likelihood term
        t += Float64(A.d[i])
    return t


def cols_of(A: Mat, a: Int, b: Int) -> Mat:
    """`_M.cols(a, b)`: exact copies of columns a..b-1."""
    var w = b - a
    var out = Mat(A.r, w)
    for i in range(A.r):  # small-loop(A: a d-wide factor row): exact copies
        for j in range(w):  # small-loop(w: components): exact copies
            out.d[i * w + j] = A.d[i * A.c + a + j]
    return out^


def rev_cols(A: Mat) -> Mat:
    """`take_cols(range(c - 1, -1, -1))`: exact copies, columns reversed."""
    var out = Mat(A.r, A.c)
    for i in range(A.r):  # small-loop(A: d x d eigenvectors): exact copies
        for j in range(A.c):  # small-loop(A: d x d eigenvectors): exact copies
            out.d[i * A.c + j] = A.d[i * A.c + A.c - 1 - j]
    return out^


def svd_desc(s: Mat, v: Mat, mut sd: Mat, mut vt: Mat) raises:
    """`_Kit.svd`'s tail: o = the stable ascending order of -s (NaN refused,
    `order_f`), S = s[o] (1 x n) and Vt = the rows of v^T in that order."""
    var n = s.c
    var neg = Mat(1, n)
    for i in range(n):  # small-loop(n: singular values, one per feature): exact negations
        neg.d[i] = -s.d[i]
    var o = Mat(n, 1)
    order_f(neg.p(), n, o.p())
    sd = Mat(1, n)
    vt = Mat(n, n)
    for a in range(n):  # small-loop(n: singular values, one per feature): the descending gather
        var j = Int(o.d[a])
        sd.d[a] = s.d[j]
        for i in range(n):  # small-loop(n: features): exact copies
            vt.d[a * n + i] = v.d[i * n + j]


def fa_mask(nc: Int, d: Int, keep: Bool) -> Mat:
    """`_M.of([1.0] * nc + [0.0] * (d - nc))` (keep) or its complement."""
    var out = Mat(1, d)
    for j in range(d):  # small-loop(d: features): the 0/1 component mask
        var on = j < nc
        out.d[j] = Float32(1) if on == keep else Float32(0)
    return out^


struct FaArgs(Copyable, Movable):
    var n: Int
    var d: Int
    var nc: Int
    var max_iter: Int
    var tol: Float64
    var llconst: Float64

    def __init__(out self, n: Int, d: Int, nc: Int, max_iter: Int, tol: Float64, llconst: Float64):
        self.n = n
        self.d = d
        self.nc = nc
        self.max_iter = max_iter
        self.tol = tol
        self.llconst = llconst


def fa_em[E: Exec](A: Mat, mut psi: Mat, mut W: Mat, mut ll_out: List[Float64], a: FaArgs) raises -> Int:
    """The EM loop over A (Rx, d x d, when n >= d; else Xc, n x d): psi (1 x
    d) and W (nc x d) replaced, the log-likelihoods appended. Returns it."""
    var k = Kit[E]()
    var n = a.n
    var d = a.d
    var nc = a.nc
    var nsqrt = sqrt(Float64(n))
    var old_ll = -inf[DType.float64]()
    var it = 0
    for i in range(1, a.max_iter + 1):
        it = i
        var sqrt_psi = k.ew1(OP_ADDS, k.ew1(OP_SQRT, psi, 0.0), FA_SMALL)
        var s2: Mat
        var Vfull: Mat
        if n >= d:
            var B = k.ew1(OP_SCALE, k.ew2(OP_DIV, A, sqrt_psi), 1.0 / nsqrt)
            var s = Mat(1, B.c)
            var v = Mat(B.c, B.c)
            E.svd(B.p(), B.r, B.c, s.p(), v.p())
            var sv = Mat(0, 0)
            Vfull = Mat(0, 0)
            svd_desc(s, v, sv, Vfull)
            s2 = k.ew1(OP_SQ, sv, 0.0)
        else:
            var Z = k.ew1(OP_SCALE, k.ew2(OP_DIV, A, sqrt_psi), 1.0 / nsqrt)
            var ev = Mat(1, d)
            var V = Mat(d, d)
            k.eigh(k.mm(Z, Z, True, False), ev, V)
            s2 = k.ew1(OP_MAXS, rev_cols(ev), 0.0)
            Vfull = mat_t(rev_cols(V))
        var Vt = mat_rows(Vfull, 0, nc)
        var sk = cols_of(s2, 0, nc)
        var unexp = dsum(s2, nc, d) if nc < d else 0.0
        W = k.ew2(OP_MUL, Vt, mat_vec_t(k.ew1(OP_SQRT, k.ew1(OP_MAXS, k.ew1(OP_ADDS, sk, -1.0), 0.0), 0.0)))
        W = k.ew2(OP_MUL, W, sqrt_psi)
        var lsk = k.ew1(OP_LOGS, sk, FA_LOG_FLOOR)
        var slog = dsum(lsk, 0, lsk.n())
        var lps = k.ew1(OP_LOGS, psi, FA_LOG_FLOOR)
        var plog = dsum(lps, 0, lps.n())
        var ll = (a.llconst + slog + unexp + plog) * (-Float64(n) / 2.0)
        ll_out.append(ll)
        if (ll - old_ll) < a.tol:
            break
        old_ll = ll
        var dfull = Vfull.r
        var keep = fa_mask(nc, dfull, True)
        var drop = fa_mask(nc, dfull, False)
        var wts = k.ew2(OP_ADD, k.ew2(OP_MUL, k.ew1(OP_MINS, s2, 1.0), keep), k.ew2(OP_MUL, s2, drop))
        var share = k.mm(wts, k.ew1(OP_SQ, Vfull, 0.0), False, False)
        psi = k.ew1(OP_MAXS, k.ew2(OP_MUL, k.ew1(OP_SQ, sqrt_psi, 0.0), share), FA_SMALL)
    return it


def _fa_args(p: PythonObject, f: PythonObject) raises -> FaArgs:
    """p = [n, d, nc, max_iter, ar] (ar: A's rows); f = [tol, llconst]."""
    return FaArgs(_n(p, 0), _n(p, 1), _n(p, 2), Int(py=p[3]), Float64(py=f[0]), Float64(py=f[1]))


def fa_em_main_py[E: Exec](
    a: PythonObject, psi: PythonObject, w: PythonObject, ll: PythonObject, p: PythonObject, f: PythonObject
) raises -> PythonObject:
    """The EM loop: a (ar x d), psi (d, in and out), w (nc x d, out), ll
    (max_iter float64, out). Returns it (the first it log-likelihoods set)."""
    var args = _fa_args(p, f)
    var ar = _n(p, 4)
    if args.nc < 1 or args.nc > args.d or ar * args.d > 2147483647:
        raise Error("x_decomp: factor analysis shape out of range")
    var pa = _f(a)
    var pp = _f(psi)
    var pw = _f(w)
    var pl = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(py=ll))
    var it = 0
    with GILReleased(Python()):
        var A = mat_from(pa, ar, args.d)
        var P = mat_from(pp, 1, args.d)
        var W = Mat(args.nc, args.d)
        var lls = List[Float64]()
        it = fa_em[E](A, P, W, lls, args)
        for i in range(args.d):
            pp.unsafe_store(i, P.d[i])
        for i in range(W.n()):
            pw.unsafe_store(i, W.d[i])
        for i in range(len(lls)):
            pl.unsafe_store(i, lls[i])
    return PythonObject(it)


def dsum_f32_py(a: PythonObject, n: PythonObject) raises -> PythonObject:
    """`_dsum` of the n float32 words at a (sequential binary64 adds)."""
    var m = Int(py=n)
    if m <= 0:
        return PythonObject(0.0)
    var pa = _f(a)
    var t = 0.0
    for i in range(m):  # small-loop(m: features): the d float32 words of one log-likelihood term
        t += Float64(pa.unsafe_load(i))
    return PythonObject(t)
