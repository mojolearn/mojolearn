# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`NMF.fit_transform` / `fit` / `transform` as ONE native operation each
(lane mca2-decomp-a, 2026-10-22): the steps `_expansion_decomp.NMF._validate`
(the negativity count), `_init` (with `_thin_svd`), `_fit_transform`,
`_solve` and `_err` sequenced from Python with the decomp kit, on `RKit`
(x_decomp/decomp_owner_kit.mojo): the same entries' typed bodies, operands,
broadcast modes, float32 scalars, Philox streams and the same binary64
scalar arithmetic CPython performed (xmean / nc, the regularization the
caller passes, the loss sums), the solver loops the binding's
`x_decomp_nmf_solve` / `x_decomp_nmf_nndsvd` run (`DecompDev`)."""
from std.math import sqrt

from x_decomp.cells import F32Ptr
from x_decomp.exec_trait import Exec
from x_decomp.nmf import NmfArgs
from x_decomp.pca_randomized import RDev, RKit
from x_decomp.decomp_owner_kit import (
    DecompDev, OP_ABS, OP_ADDS, OP_DIV, OP_GTS, OP_LOGS, OP_MAXS, OP_MUL, OP_SCALE, OP_SELECT, OP_SQDIFF, OP_SQRT,
    kit_rand, kit_rows,
)

#: the init codes the caller passes
comptime NMF_INIT_RANDOM = 0
comptime NMF_INIT_NNDSVD = 1
comptime NMF_INIT_NNDSVDA = 2
comptime NMF_INIT_NNDSVDAR = 3
comptime NMF_INIT_GIVEN = 4  # W and H given (init='custom')
comptime NMF_TRANSFORM = 5  # H fixed (transform): W from the solver's start
#: `_F32_EPS` and the smallest normal float32 (`_err`'s floors)
comptime _F32_EPS = 1.1920928955078125e-07
comptime _F32_TINY = 1.1754943508222875e-38


def _scaled_total[E: Exec, D: RDev, RES: Bool](mut kit: RKit[E, D, RES], a: Int, scale: Float64) raises -> Float64:
    """`k.ew("scale", k.total(A), s=scale).s[0]`."""
    var total_value = kit.total(a)
    var total = kit.constant(Float32(total_value))
    var scaled = kit.ew(OP_SCALE, total, scalar=scale)
    return kit.word(scaled)


def _any_negative[E: Exec, D: RDev, RES: Bool](mut kit: RKit[E, D, RES], m: Int) raises -> Bool:
    """`_any_negative(M)`: the count of x < 0 (as -x > 0) through the cells."""
    if kit.r(m) * kit.c(m) == 0:
        return False
    var negated = kit.ew(OP_SCALE, m, scalar=-1.0)
    var flags = kit.ew(OP_GTS, negated, scalar=0.0)
    return kit.total(flags) > 0


def _thin_svd[E: Exec, D: RDev, RES: Bool](
    mut kit: RKit[E, D, RES], x: Int, nc: Int, u_based: Bool
) raises -> Tuple[Int, Int, Int]:
    """`_thin_svd(k, X, nc, u_based)`: (U, S, Vt) with svd_flip's signs."""
    var n = kit.r(x)
    var d = kit.c(x)
    var u: Int
    var s: Int
    var vt: Int
    if d <= n:
        var raw = kit.tall_svd(x)
        s = kit.columns(raw[0], 0, nc)
        vt = kit_rows(kit, raw[1], 0, nc)
        var xv = kit.mm(x, vt, tb=True)
        u = kit.ew(OP_DIV, xv, s)
    else:
        var xt = kit.transpose(x)
        var raw = kit.tall_svd(xt)
        s = kit.columns(raw[0], 0, nc)
        var ut = kit_rows(kit, raw[1], 0, nc)
        u = kit.transpose(ut)
        var ux = kit.mm(u, x, ta=True)
        var st = kit.transpose(s)
        vt = kit.ew(OP_DIV, ux, st)
    if u_based:
        var sg = kit.absmax_signs(u, True)
        u = kit.ew(OP_MUL, u, sg)
        var sgt = kit.transpose(sg)
        vt = kit.ew(OP_MUL, vt, sgt)
    else:
        var sg = kit.absmax_signs(vt, False)
        var sgt = kit.transpose(sg)
        u = kit.ew(OP_MUL, u, sgt)
        vt = kit.ew(OP_MUL, vt, sg)
    return (u, s, vt)


def _random_abs[E: Exec, D: RDev, RES: Bool](
    mut kit: RKit[E, D, RES], rows: Int, cols: Int, seed: Int, stream: Int, scale: Float64
) raises -> Int:
    # k.ew("scale", k.ew("abs", k.rand(rows, cols, seed, stream, 1)), s=scale)
    var drawn = kit_rand(kit, rows, cols, seed, stream, 1)
    var absolute = kit.ew(OP_ABS, drawn)
    return kit.ew(OP_SCALE, absolute, scalar=scale)


def _init[E: Exec, D: RDev, RES: Bool, X: DecompDev](
    mut kit: RKit[E, D, RES], m: Int, nc: Int, init: Int, seed: Int
) raises -> Tuple[Int, Int]:
    """`NMF._init(k, M, nc, init)`: (W n x nc, H nc x d)."""
    var n = kit.r(m)
    var d = kit.c(m)
    var xmean = _scaled_total(kit, m, 1.0 / Float64(n * d))
    if init == NMF_INIT_RANDOM:
        var avg = sqrt(xmean / Float64(nc))
        var h = _random_abs(kit, nc, d, seed, 10, avg)
        var w = _random_abs(kit, n, nc, seed, 11, avg)
        return (w, h)
    var svd = _thin_svd(kit, m, nc, n >= d)
    var u = svd[0]
    var s = svd[1]
    var vt = svd[2]
    var w = kit.zeros(n, nc)
    var h = kit.zeros(nc, d)
    var r = kit.r(s) * kit.c(s)
    _ = X.nmf_nndsvd(kit.ptr(u), kit.ptr(s), kit.ptr(vt), kit.ptr(w), kit.ptr(h), n, d, r, nc)
    var z = kit.zeros(1, 1)
    var lo: Float64 = 1e-6
    var hi: Float64 = 1e-13
    var floor = lo - hi
    w = kit.ew(OP_SELECT, w, w, z, scalar=floor)
    h = kit.ew(OP_SELECT, h, h, z, scalar=floor)
    if init == NMF_INIT_NNDSVDA:
        var fill_w = kit.constant(Float32(xmean))
        w = kit.ew(OP_SELECT, w, w, fill_w, scalar=0.0)
        var fill_h = kit.constant(Float32(xmean))
        h = kit.ew(OP_SELECT, h, h, fill_h, scalar=0.0)
    elif init == NMF_INIT_NNDSVDAR:
        var spread = abs(xmean) / 100.0
        var rw = _random_abs(kit, n, nc, seed, 12, spread)
        var rh = _random_abs(kit, nc, d, seed, 13, spread)
        w = kit.ew(OP_SELECT, w, w, rw, scalar=0.0)
        h = kit.ew(OP_SELECT, h, h, rh, scalar=0.0)
    return (w, h)


def _err[E: Exec, D: RDev, RES: Bool](mut kit: RKit[E, D, RES], m: Int, w: Int, h: Int, beta: Float64) raises -> Float64:
    """`NMF._err`: the beta-divergence reconstruction error."""
    if beta == 2.0:
        var wh = kit.mm(w, h)
        var spread = kit.ew(OP_SQDIFF, m, wh)
        var total_value = kit.total(spread)
        var total = kit.constant(Float32(total_value))
        var root = kit.ew(OP_SQRT, total)
        return kit.word(root)
    var product = kit.mm(w, h)
    var wh = kit.ew(OP_MAXS, product, scalar=_F32_EPS)
    var keep = kit.ew(OP_GTS, m, scalar=_F32_EPS)
    var ratio = kit.ew(OP_DIV, m, wh)
    var res: Float64
    if beta == 1.0:
        var cw = kit.sum(w, True)
        var rh = kit.sum(h, False)
        var cross = kit.mm(cw, rh)
        var sum_wh = kit.word(cross)
        var logs = kit.ew(OP_LOGS, ratio, scalar=_F32_TINY)
        var weighted = kit.ew(OP_MUL, m, logs)
        var kept = kit.ew(OP_MUL, weighted, keep)
        var xlog = kit.total(kept)
        var mk = kit.ew(OP_MUL, m, keep)
        var xs = kit.total(mk)
        res = xlog + sum_wh - xs
    else:
        var dk = kit.ew(OP_MUL, ratio, keep)
        var dsum = kit.total(dk)
        var logs = kit.ew(OP_LOGS, ratio, scalar=_F32_TINY)
        var lk = kit.ew(OP_MUL, logs, keep)
        var lsum = kit.total(lk)
        res = dsum - Float64(kit.r(m) * kit.c(m)) - lsum
    return sqrt(2.0 * res) if res > 0 else 0.0


def nmf_run[E: Exec, D: RDev, RES: Bool, X: DecompDev](
    x: F32Ptr, n: Int, d: Int, nc: Int, init: Int, w_in: F32Ptr, h_in: F32Ptr, seed: Int, solver: Int,
    a: NmfArgs, w_out: F32Ptr, h_out: F32Ptr,
) raises -> Tuple[Int, Float64]:
    """One NMF operation on x (n x d): `init` NMF_INIT_* (fit_transform,
    H updated, the reconstruction error returned) or NMF_TRANSFORM (H =
    h_in fixed). W (n x nc) and H (nc x d) written; returns (n_iter_, err)."""
    var kit = RKit[E, D, RES]()
    try:
        var m = kit.add(Int(x), -1, n, d)
        if _any_negative(kit, m):
            raise Error("Negative values in data passed to NMF (input X)")
        var w: Int
        var h: Int
        if init == NMF_INIT_GIVEN:
            w = kit.add(Int(w_in), -1, n, nc)
            h = kit.add(Int(h_in), -1, nc, d)
        elif init == NMF_TRANSFORM:
            h = kit.add(Int(h_in), -1, nc, d)
            if solver == 1:
                var xmean = _scaled_total(kit, m, 1.0 / Float64(n * d))
                # k.const(sqrt(xmean / nc), n, nc): every word the float32
                # start (0 + start is exact), filled by one cell
                var start = Float64(Float32(sqrt(xmean / Float64(nc))))
                var blank = kit.zeros(n, nc)
                w = kit.ew(OP_ADDS, blank, scalar=start)
            else:
                w = kit.zeros(n, nc)
        else:
            var started = _init[E, D, RES, X](kit, m, nc, init, seed)
            w = started[0]
            h = started[1]
        # `_solve`: host copies of W and H solved in place on X's host words
        var ws = kit.copy_host(w)
        var hs = kit.copy_host(h)
        var it = X.nmf_solve(kit.ptr(m), kit.ptr(ws), kit.ptr(hs), n, d, nc, solver, a)
        if it < 0:
            raise Error("ZeroDivisionError: float division by zero")
        var err = Float64(0.0)
        if init != NMF_TRANSFORM:
            err = _err(kit, m, ws, hs, a.beta)
        kit.copy_out(ws, w_out, n * nc)
        kit.copy_out(hs, h_out, nc * d)
        kit.close()
        return (it, err)
    except error:
        kit.close()
        raise error^


def nmf_inverse_run[E: Exec, D: RDev, RES: Bool](
    w: F32Ptr, n: Int, nc: Int, h: F32Ptr, d: Int, out_buf: F32Ptr
) raises:
    """`inverse_transform`: W components_ (n x d)."""
    var kit = RKit[E, D, RES]()
    try:
        var wm = kit.add(Int(w), -1, n, nc)
        var hm = kit.add(Int(h), -1, nc, d)
        var y = kit.mm(wm, hm)
        kit.copy_out(y, out_buf, n * d)
        kit.close()
    except error:
        kit.close()
        raise error^
