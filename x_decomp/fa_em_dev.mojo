# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""x_decomp/fa_em.mojo's FactorAnalysis EM loop on resident device matrices
(lane py-runtime-b): the same statements on `DKit`. The d x d spectral
step comes home where `_Kit.svd` / `_Kit.eigh` brought it (DevExec's solve
writes host s and V): its descending order, the column reversal and the
transposes are exact copies of d x d words (`svd_desc`, `rev_cols`), and the
log-likelihood's float32 words come home for `_dsum`, as Python read them.
GPU binding only."""
from std.math import inf, sqrt
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from x_decomp.api import _f, _n
from x_decomp.fa_em import FA_LOG_FLOOR, FA_SMALL, FaArgs, _fa_args, dsum, fa_mask, rev_cols, svd_desc
from x_decomp.kit import (
    Mat, OP_ADD, OP_ADDS, OP_DIV, OP_LOGS, OP_MAXS, OP_MINS, OP_MUL, OP_SCALE, OP_SQ, OP_SQRT, mat_from, mat_t,
)
from x_decomp.kit_device import DKit, DMat


def fa_em_dev(mut k: DKit, A: DMat, mut psi: DMat, mut W: DMat, mut ll_out: List[Float64], a: FaArgs) raises -> Int:
    var n = a.n
    var d = a.d
    var nc = a.nc
    var nsqrt = sqrt(Float64(n))
    var old_ll = -inf[DType.float64]()
    var it = 0
    for i in range(1, a.max_iter + 1):
        it = i
        var sqrt_psi = k.ew1(OP_ADDS, k.ew1(OP_SQRT, psi, 0.0), FA_SMALL)
        var s2: DMat
        var Vh: Mat
        if n >= d:
            var B = k.ew1(OP_SCALE, k.ew2(OP_DIV, A, sqrt_psi), 1.0 / nsqrt)
            var s = Mat(0, 0)
            var v = Mat(0, 0)
            k.svd_host(B, s, v)
            var sv = Mat(0, 0)
            Vh = Mat(0, 0)
            svd_desc(s, v, sv, Vh)
            s2 = k.ew1(OP_SQ, k.upload(sv), 0.0)
        else:
            var Z = k.ew1(OP_SCALE, k.ew2(OP_DIV, A, sqrt_psi), 1.0 / nsqrt)
            var e = k.eigh(k.mm(Z, Z, True, False))
            var vh = k.get(e.vd)
            k.sync()
            s2 = k.ew1(OP_MAXS, k.upload(rev_cols(e.wh)), 0.0)
            Vh = mat_t(rev_cols(vh))
        var Vfull = k.upload(Vh)
        var Vt = k.rows(Vfull, 0, nc)
        var s2h = k.get(s2)
        k.sync()
        var sk = k.rows(k.vec_t(k.copy(s2)), 0, nc)
        sk = k.vec_t(sk^)
        var unexp = dsum(s2h, nc, d) if nc < d else 0.0
        W = k.ew2(OP_MUL, Vt, k.vec_t(k.ew1(OP_SQRT, k.ew1(OP_MAXS, k.ew1(OP_ADDS, sk, -1.0), 0.0), 0.0)))
        W = k.ew2(OP_MUL, W, sqrt_psi)
        var lskd = k.ew1(OP_LOGS, sk, FA_LOG_FLOOR)
        var lpsd = k.ew1(OP_LOGS, psi, FA_LOG_FLOOR)
        var lsk = k.get(lskd)
        var lps = k.get(lpsd)
        k.sync()
        _ = lskd^
        _ = lpsd^
        var slog = dsum(lsk, 0, lsk.n())
        var plog = dsum(lps, 0, lps.n())
        var ll = (a.llconst + slog + unexp + plog) * (-Float64(n) / 2.0)
        ll_out.append(ll)
        if (ll - old_ll) < a.tol:
            break
        old_ll = ll
        var dfull = Vfull.r
        var keep = k.upload(fa_mask(nc, dfull, True))
        var drop = k.upload(fa_mask(nc, dfull, False))
        var wts = k.ew2(OP_ADD, k.ew2(OP_MUL, k.ew1(OP_MINS, s2, 1.0), keep), k.ew2(OP_MUL, s2, drop))
        var share = k.mm(wts, k.ew1(OP_SQ, Vfull, 0.0), False, False)
        psi = k.ew1(OP_MAXS, k.ew2(OP_MUL, k.ew1(OP_SQ, sqrt_psi, 0.0), share), FA_SMALL)
    return it


def fa_em_main_dev_py(
    a: PythonObject, psi: PythonObject, w: PythonObject, ll: PythonObject, p: PythonObject, f: PythonObject
) raises -> PythonObject:
    """`fa_em_main_py` on the resident kit."""
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
        var k = DKit()
        var A = k.upload(mat_from(pa, ar, args.d))
        var P = k.upload(mat_from(pp, 1, args.d))
        var W = k.zeros(args.nc, args.d)
        var lls = List[Float64]()
        it = fa_em_dev(k, A, P, W, lls, args)
        var hp = k.get(P)
        var hw = k.get(W)
        k.sync()
        for i in range(args.d):
            pp.unsafe_store(i, hp.d[i])
        for i in range(hw.n()):
            pw.unsafe_store(i, hw.d[i])
        for i in range(len(lls)):
            pl.unsafe_store(i, lls[i])
    return PythonObject(it)
