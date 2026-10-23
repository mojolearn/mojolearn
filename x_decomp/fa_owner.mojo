# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`FactorAnalysis.fit` and its fitted queries (transform, get_covariance,
get_precision, score_samples, score) as ONE native operation each (lane
mca2-decomp-a, 2026-10-22): the steps `_expansion_decomp.FactorAnalysis.fit`,
`_fit_fast`, `_fit_store`, `transform`, `_cov`, `_ss` and `score`
sequenced from Python with the decomp kit, on `RKit`
(x_decomp/decomp_owner_kit.mojo): the same entries' typed bodies, operands,
broadcast modes, float32 scalars and binary64 scalar arithmetic (products
feeding a sum pinned, as CPython rounds each), the routes the binding's
switches chose (`DecompDev.fa_flags`: the FAST Apple Gram and device EM,
the fused transform), the EM loop of `x_decomp_fa_em_main`, the QR, LU and
rotation entries where the Python kit reached them."""
from std.math import sqrt
from std.memory import bitcast

from checks.numerics import pinned_mul_f64
from x_decomp.api import qr_r_run
from x_decomp.cells import F32Ptr
from x_decomp.exec_trait import Exec
from x_decomp.fa_em import FaArgs, dsum_f32_run, fa_em_main_check
from x_decomp.pca_randomized import RDev, RKit
from x_decomp.decomp_owner_kit import (
    DecompDev, OP_ADD, OP_ADDS, OP_DIV, OP_LOGS, OP_MAXS, OP_MINS, OP_MUL, OP_SCALE, OP_SQ, OP_SQRT, OP_SUB,
    kit_colmean, kit_diag_add, kit_eigh, kit_eye, kit_inv, kit_logdet, kit_reversed_columns, kit_rows,
)

comptime LOG_2PI = 1.8378770664093453
#: x_decomp/fa_fast.mojo FA_MAX_D, FA_TR_MAXK, FA_TR_FLOATS (`_FA_MAX_D`, ...)
comptime _FA_MAX_D = 256
comptime _FA_TR_MAXK = 16
comptime _FA_TR_FLOATS = 4096
comptime _SMALL = 1e-12
comptime _F32_TINY = 1.1754943508222875e-38
#: `_ortho_rotation`'s defaults
comptime _ROT_TOL = 1e-6
comptime _ROT_MAX_ITER = 100
#: the fitted queries
comptime FA_TRANSFORM = 0
comptime FA_COVARIANCE = 1
comptime FA_PRECISION = 2
comptime FA_SCORE_SAMPLES = 3
comptime FA_SCORE = 4


def fa_llconst(d: Int, nc: Int) -> Float64:
    """`d * _LOG_2PI + nc`, the product rounded before the sum."""
    return pinned_mul_f64(Float64(d), LOG_2PI) + Float64(nc)


def _dsum[E: Exec, D: RDev, RES: Bool](mut kit: RKit[E, D, RES], a: Int) raises -> Float64:
    """`_dsum(k, M)`: the sequential binary64 sum of M's host words."""
    var count = kit.r(a) * kit.c(a)
    if count == 0:
        return 0.0
    var p = kit.ptr(a)
    return dsum_f32_run(Int(p), count)


def _gram_loop[E: Exec, D: RDev, RES: Bool, X: DecompDev](
    mut kit: RKit[E, D, RES], gram: Int, psi_in: Int, n: Int, d: Int, nc: Int, max_iter: Int, tol: Float64,
    llconst: Float64, ll_out: MutPointer[Float64, MutAnyOrigin],
) raises -> Tuple[Int, Int, Int]:
    """`_fit_fast`'s Python EM loop (GRAM_ONCE without ITER_DEVICE): (W,
    psi, iterations), the log-likelihoods written to ll_out."""
    var psi = psi_in
    var old_ll = bitcast[DType.float64](UInt64(0xFFF0000000000000))  # -inf
    var keep = kit.zeros(1, d)
    var drop = kit.zeros(1, d)
    var pk = kit.ptr(keep)
    var pd = kit.ptr(drop)
    for j in range(d):  # driver-loop: over the features' keep / drop flags (model structure)
        if j < nc:
            pk[j] = 1.0
        else:
            pd[j] = 1.0
    var it = 0
    var w = -1
    for step in range(1, max_iter + 1):  # driver-loop: over the EM iterations
        it = step
        var root = kit.ew(OP_SQRT, psi)
        var sqrt_psi = kit.ew(OP_ADDS, root, scalar=_SMALL)
        var left = kit.ew(OP_DIV, gram, sqrt_psi)
        var sqrt_psi_t = kit.transpose(sqrt_psi)
        var both = kit.ew(OP_DIV, left, sqrt_psi_t)
        var scaled = kit.ew(OP_SCALE, both, scalar=1.0 / Float64(n))
        var eig = kit_eigh[E, D, RES, X](kit, scaled)
        var ev = kit_reversed_columns(kit, eig[0])
        var s2 = kit.ew(OP_MAXS, ev, scalar=0.0)
        var v_rev = kit_reversed_columns(kit, eig[1])
        var vfull = kit.transpose(v_rev)
        var lead = kit_rows(kit, vfull, 0, nc)
        var signs = kit.absmax_signs(lead, False)
        var vt = kit.ew(OP_MUL, lead, signs)
        var sk = kit.columns(s2, 0, nc)
        var unexp = Float64(0.0)
        if nc < d:
            var tail = kit.columns(s2, nc, d)
            unexp = _dsum(kit, tail)
        var shifted = kit.ew(OP_ADDS, sk, scalar=-1.0)
        var floored = kit.ew(OP_MAXS, shifted, scalar=0.0)
        var spread = kit.ew(OP_SQRT, floored)
        var spread_t = kit.transpose(spread)
        w = kit.ew(OP_MUL, vt, spread_t)
        w = kit.ew(OP_MUL, w, sqrt_psi)
        var slogs = kit.ew(OP_LOGS, sk, scalar=_F32_TINY)
        var slog = _dsum(kit, slogs)
        var plogs = kit.ew(OP_LOGS, psi, scalar=_F32_TINY)
        var plog = _dsum(kit, plogs)
        var ll = (llconst + slog + unexp + plog) * (-Float64(n) / 2.0)
        ll_out[step - 1] = ll
        if (ll - old_ll) < tol:
            break
        old_ll = ll
        # main's cancellation-free psi update
        var capped = kit.ew(OP_MINS, s2, scalar=1.0)
        var kept = kit.ew(OP_MUL, capped, keep)
        var dropped = kit.ew(OP_MUL, s2, drop)
        var wts = kit.ew(OP_ADD, kept, dropped)
        var vsq = kit.ew(OP_SQ, vfull)
        var share = kit.mm(wts, vsq)
        var psq = kit.ew(OP_SQ, sqrt_psi)
        var product = kit.ew(OP_MUL, psq, share)
        psi = kit.ew(OP_MAXS, product, scalar=_SMALL)
    return (w, psi, it)


def fa_fit_run[E: Exec, D: RDev, RES: Bool, X: DecompDev](
    x: F32Ptr, n: Int, d: Int, nc: Int, max_iter: Int, tol: Float64, psi0: F32Ptr, rotation: Int,
    components_out: F32Ptr, psi_out: F32Ptr, mean_out: F32Ptr, ll_out: MutPointer[Float64, MutAnyOrigin],
) raises -> Int:
    """`FactorAnalysis.fit`: rotation 0 None, 1 varimax, 2 quartimax; psi0
    the initial noise variances (d). Writes components (nc x d), noise
    variances (d), mean (d) and the log-likelihoods (ll_out, at least
    max(max_iter, 1) words). Returns n_iter_ (0: no component was fitted)."""
    var llconst = fa_llconst(d, nc)
    var kit = RKit[E, D, RES]()
    try:
        var m = kit.add(Int(x), -1, n, d)
        # the two-pass mean
        var first = kit_colmean(kit, m)
        var residual = kit.ew(OP_SUB, m, first)
        var correction = kit_colmean(kit, residual)
        var mean = kit.ew(OP_ADD, first, correction)
        var psi = kit.add(Int(psi0), -1, 1, d)
        var flags = X.fa_flags()
        var w = -1
        var it = 0
        if (flags & 3) != 0 and d <= _FA_MAX_D and nc >= 1 and nc <= d and max_iter >= 1:
            # `_fit_fast`: the Gram once on the device
            var gram = kit.device_output(2 * d if (flags & 8) != 0 else d, d)
            var variance = kit.device_output(1, d)
            var dm = kit.device(m)
            var dmean = kit.device(mean)
            X.fa_gram(dm, dmean, kit.mats[gram].dev, kit.mats[variance].dev, n, d)
            if (flags & 2) != 0:
                var wa = kit.zeros(nc, d)
                var pa = kit.zeros(1, d)
                var p0 = kit.copy_host(psi)
                var pw = kit.ptr(wa)
                var pp = kit.ptr(pa)
                var src = kit.ptr(p0)
                _ = p0  # keep the Python object alive through the native call above
                it = X.fa_em(kit.mats[gram].dev, src, pw, pp, ll_out, d, nc, n, max_iter, tol)
                _ = p0  # keep the Python object alive through the native call above
                w = wa
                psi = pa
            else:
                var looped = _gram_loop[E, D, RES, X](kit, gram, psi, n, d, nc, max_iter, tol, llconst, ll_out)
                w = looped[0]
                psi = looped[1]
                it = looped[2]
        else:
            var xc = kit.ew(OP_SUB, m, mean)
            var a = xc
            if n >= d:
                # Xc = Q R once (consumed: never read again)
                var r = kit.zeros(d, d)
                var pr = kit.ptr(r)
                var resident = False
                comptime if RES:
                    resident = X.has_qr_r() and d >= 1 and kit.mats[xc].dev >= 0
                if resident:
                    _ = X.qr_r(kit.mats[xc].dev, pr, n, d, True)
                else:
                    qr_r_run[E](kit.ptr(xc), pr, n, d)
                a = r
            var wm = kit.zeros(nc, d)
            var ps = kit.copy_host(psi)
            var rows = kit.r(a)
            var args = FaArgs(n, d, nc, max_iter, tol, llconst)
            fa_em_main_check(args, rows)
            var ll = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(ll_out))
            var pa = kit.ptr(a)
            var pp = kit.ptr(ps)
            var pw = kit.ptr(wm)
            it = X.fa_em_main(pa, pp, pw, ll, rows, args)
            w = wm
            psi = ps
        if it == 0:
            kit.close()
            return 0
        if rotation != 0:
            # `_ortho_rotation(k, W.T, rotation).rows(0, nc)`
            var c = kit.transpose(w)
            var rotated = kit.zeros(kit.c(c), kit.r(c))
            var pc = kit.ptr(c)
            var po = kit.ptr(rotated)
            X.ortho_rotation(pc, po, kit.r(c), kit.c(c), rotation == 1, _ROT_MAX_ITER, _ROT_TOL)
            w = kit_rows(kit, rotated, 0, nc)
        kit.copy_out(w, components_out, nc * d)
        kit.copy_out(psi, psi_out, d)
        kit.copy_out(mean, mean_out, d)
        kit.close()
        return it
    except error:
        kit.close()
        raise error^


def _cov[E: Exec, D: RDev, RES: Bool](mut kit: RKit[E, D, RES], w: Int, psi: Int) raises -> Int:
    """`FactorAnalysis._cov`: W^T W with the noise variances on its diagonal."""
    var gram = kit.mm(w, w, ta=True)
    return kit_diag_add(kit, gram, psi)


def fa_query_run[E: Exec, D: RDev, RES: Bool, X: DecompDev](
    op: Int, x: F32Ptr, n: Int, d: Int, nc: Int, components: F32Ptr, psi: F32Ptr, mean: F32Ptr, out_buf: F32Ptr,
) raises -> Float64:
    """A fitted FactorAnalysis query on X (n x d; unread for the covariance
    and precision): FA_TRANSFORM (n x nc), FA_COVARIANCE / FA_PRECISION
    (d x d), FA_SCORE_SAMPLES (n) written to out_buf; FA_SCORE returned."""
    var kit = RKit[E, D, RES]()
    try:
        var w = kit.add(Int(components), -1, nc, d)
        var noise = kit.add(Int(psi), -1, 1, d)
        var mu = kit.add(Int(mean), -1, 1, d)
        var result = Float64(0.0)
        if op == FA_TRANSFORM:
            var flags = X.fa_flags()
            var xm = kit.add(Int(x), -1, n, d)
            var fused = (flags & 4) != 0 and nc <= _FA_TR_MAXK and d * (nc + 1) <= _FA_TR_FLOATS
            var y: Int
            var wpsi: Int
            if fused:
                wpsi = kit.ew(OP_DIV, w, noise)
            else:
                xm = kit.ew(OP_SUB, xm, mu)
                wpsi = kit.ew(OP_DIV, w, noise)
            var eye = kit_eye(kit, nc)
            var inner = kit.mm(wpsi, w, tb=True)
            var summed = kit.ew(OP_ADD, eye, inner)
            var cov_z = kit_inv[E, D, RES, X](kit, summed)
            if fused:
                var p = kit.mm(wpsi, cov_z, ta=True)
                y = kit.device_output(n, nc)
                var dx = kit.device(xm)
                var dmu = kit.device(mu)
                var dp = kit.device(p)
                X.fa_transform(dx, dmu, dp, kit.mats[y].dev, n, d, nc)
            else:
                var projected = kit.mm(xm, wpsi, tb=True)
                y = kit.mm(projected, cov_z)
            kit.copy_out(y, out_buf, n * nc)
        elif op == FA_COVARIANCE:
            var c = _cov(kit, w, noise)
            kit.copy_out(c, out_buf, d * d)
        elif op == FA_PRECISION:
            var c = _cov(kit, w, noise)
            var p = kit_inv[E, D, RES, X](kit, c)
            kit.copy_out(p, out_buf, d * d)
        else:
            # `_ss`
            var xm = kit.add(Int(x), -1, n, d)
            var xr = kit.ew(OP_SUB, xm, mu)
            var c = _cov(kit, w, noise)
            var p = kit_inv[E, D, RES, X](kit, c)
            var ld = kit_logdet[E, D, RES, X](kit, p)
            var xp = kit.mm(xr, p)
            var prod = kit.ew(OP_MUL, xr, xp)
            var q = kit.sum(prod, False)
            var half = kit.ew(OP_SCALE, q, scalar=-0.5)
            var shift = -0.5 * (pinned_mul_f64(Float64(d), LOG_2PI) - ld)
            var v = kit.ew(OP_ADDS, half, scalar=shift)
            if op == FA_SCORE_SAMPLES:
                kit.copy_out(v, out_buf, n)
            else:
                var total_value = kit.total(v)
                var total = kit.constant(Float32(total_value))
                var mean_ll = kit.ew(OP_SCALE, total, scalar=1.0 / Float64(n))
                result = kit.word(mean_ll)
        kit.close()
        return result
    except error:
        kit.close()
        raise error^
