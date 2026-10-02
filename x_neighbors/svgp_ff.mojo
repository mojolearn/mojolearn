# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""SVGP in float-float (lane/neural-pass106, 2026-10-01), the fallback when the
float32 factors fail (taxi: Sigma = Kuu + jitter I + B / noise has
eigenvalues from 1e-6 to 9e5, and B's float32 accumulation error is larger
than the smallest). B = Kuf Kfu and b = Kuf y are accumulated in float-float
from the float32 Kfu values (`matmul_tn_acc_ff_item`: every product exact
(two-product), the sums double-double), so B is a Gram matrix to ~1e-14
and Sigma is positive definite as it should be; then `svgp_item`'s solve in
float-float (x_linear/ff.mojo): the two Cholesky factors, alpha, C = Kuu^-1 -
Sigma^-1, q_mu, q_sqrt and the bound, each rounded to float32 at the store.
Every step a fixed chain of IDENTICAL operations: the same words on the host
and every device. Choice (Andrew's standing order, best accuracy): the
jitter escalation needed jitter 1 to 10 on taxi (held-out R2 0.13 / -0.04)."""
from x_neighbors.items import FP
from x_linear.ff import (
    FF, ff_of, two_prod, ff_add, ff_add_f, ff_sub, ff_mul, ff_div, ff_f32, ff_ld, ff_st, ff_cholesky, ff_chol_solve,
)
from checks.numerics import ftz, identical_log
from core.host_parallel import host_parallelize


def matmul_tn_acc_ff_item(t: Int, a: FP, b: FP, rh: FP, rl: FP, rows: Int, n: Int, m: Int):
    """`matmul_tn_acc_item` in float-float: (rh, rl)[t] continued by
    sum_p a[p, i] b[p, j] over `rows` rows, p ascending, t = i*m + j."""
    var i = t // m
    var j = t - i * m
    var acc = FF(rh.unsafe_load(t), rl.unsafe_load(t))
    for p in range(rows):
        acc = ff_add(acc, two_prod(ftz(a.unsafe_load(p * n + i)), ftz(b.unsafe_load(p * m + j))))
    rh.unsafe_store(t, acc.hi)
    rl.unsafe_store(t, acc.lo)


@always_inline
def _kj(kuu: FP, m: Int, i: Int, k: Int, jitter: Float32) -> FF:
    """(Kuu + jitter I)[i, k] as float-float."""
    var v = ff_of(kuu.unsafe_load(i * m + k))
    if i == k:
        v = ff_add_f(v, jitter)
    return v


def svgp_ff_solve(
    kuu: FP, bh: FP, bl: FP, bvh: FP, bvl: FP, y: FP, alpha: FP, cmat: FP, qmu: FP, qsqrt: FP, info: FP,
    m: Int, n: Int, noise: Float32, jitter: Float32, kdiag: Float32,
):
    """`svgp_item`'s outputs from the float-float statistics (bh/bl, bvh/bvl);
    info = [elbo, ok]."""
    var mm = m * m
    var w = List[Float32](length=4 * mm + 4 * m, fill=Float32(0))
    var luh = FP(unsafe_from_address=Int(w.unsafe_ptr()))
    var lul = luh + mm
    var lsh = lul + mm
    var lsl = lsh + mm
    var xah = lsl + mm
    var xal = xah + m
    var nz = ff_of(noise)
    for i in range(m):
        for k in range(m):
            var v = _kj(kuu, m, i, k, jitter)
            ff_st(luh, lul, i * m + k, v)
            ff_st(lsh, lsl, i * m + k, ff_add(v, ff_div(ff_ld(bh, bl, i * m + k), nz)))
    var ok1 = ff_cholesky(luh, lul, m)
    var ok2 = ff_cholesky(lsh, lsl, m)
    if not (ok1 and ok2):
        info.unsafe_store(0, Float32(0))
        info.unsafe_store(1, Float32(0))
        _ = w^
        return
    # alpha = Sigma^-1 b / noise
    for i in range(m):
        ff_st(xah, xal, i, ff_ld(bvh, bvl, i))
    ff_chol_solve(lsh, lsl, m, xah, xal)
    for i in range(m):
        var a = ff_div(ff_ld(xah, xal, i), nz)
        ff_st(xah, xal, i, a)
        alpha.unsafe_store(i, ff_f32(a))
    # q_mu = (Kuu + jitter I) alpha
    for i in range(m):
        var s = ff_of(Float32(0))
        for k in range(m):
            s = ff_add(s, ff_mul(_kj(kuu, m, i, k, jitter), ff_ld(xah, xal, k)))
        qmu.unsafe_store(i, ff_f32(s))
    # C = Kuu^-1 - Sigma^-1 and S = Kuu' Sigma^-1 Kuu', one column a task
    var sw = List[Float32](length=2 * mm, fill=Float32(0))
    var sh = FP(unsafe_from_address=Int(sw.unsafe_ptr()))
    var sl = sh + mm
    var cols = List[Float32](length=6 * m * m, fill=Float32(0))
    var cp = FP(unsafe_from_address=Int(cols.unsafe_ptr()))

    def column(j: Int) {imm kuu, imm m, imm jitter, imm luh, imm lul, imm lsh, imm lsl, imm cmat, imm sh, imm sl, imm cp}:
        var uh = cp + j * 6 * m
        var ul = uh + m
        var vh = ul + m
        var vl = vh + m
        var kh = vl + m
        var kl = kh + m
        for i in range(m):
            var e = ff_of(Float32(1) if i == j else Float32(0))
            ff_st(uh, ul, i, e)
            ff_st(vh, vl, i, e)
            ff_st(kh, kl, i, _kj(kuu, m, i, j, jitter))
        ff_chol_solve(luh, lul, m, uh, ul)
        ff_chol_solve(lsh, lsl, m, vh, vl)
        for i in range(m):
            cmat.unsafe_store(i * m + j, ff_f32(ff_sub(ff_ld(uh, ul, i), ff_ld(vh, vl, i))))
        ff_chol_solve(lsh, lsl, m, kh, kl)
        for i in range(m):
            var s = ff_of(Float32(0))
            for k in range(m):
                s = ff_add(s, ff_mul(_kj(kuu, m, i, k, jitter), ff_ld(kh, kl, k)))
            ff_st(sh, sl, i * m + j, s)

    host_parallelize(column, m)
    var ok3 = ff_cholesky(sh, sl, m)
    for i in range(m):
        for k in range(m):
            qsqrt.unsafe_store(i * m + k, ff_f32(ff_ld(sh, sl, i * m + k)) if k <= i else Float32(0))
    # the collapsed bound, its cancelling terms in float-float
    var yty = ff_of(Float32(0))
    for i in range(n):
        var yv = y.unsafe_load(i)
        yty = ff_add(yty, two_prod(yv, yv))
    for i in range(m):
        ff_st(xah, xal, i, ff_ld(bvh, bvl, i))
    ff_chol_solve(lsh, lsl, m, xah, xal)
    var bsb = ff_of(Float32(0))
    for i in range(m):
        bsb = ff_add(bsb, ff_mul(ff_ld(bvh, bvl, i), ff_ld(xah, xal, i)))
    var quad = ff_sub(ff_div(yty, nz), ff_div(bsb, ff_mul(nz, nz)))
    var lds = Float32(0)
    var ldu = Float32(0)
    for i in range(m):
        lds = ftz(lds + ftz(identical_log(ff_f32(ff_ld(lsh, lsl, i * m + i)))))
        ldu = ftz(ldu + ftz(identical_log(ff_f32(ff_ld(luh, lul, i * m + i)))))
    var logdet = ff_add(ff_mul(ff_of(Float32(2)), ff_sub(ff_of(lds), ff_of(ldu))),
                        ff_mul(ff_of(Float32(n)), ff_of(ftz(identical_log(noise)))))
    # tr(Kuu^-1 B): column solves against B, one column a task
    var trs = List[Float32](length=2 * m, fill=Float32(0))
    var tp = FP(unsafe_from_address=Int(trs.unsafe_ptr()))

    def tcol(j: Int) {imm bh, imm bl, imm m, imm luh, imm lul, imm cp, imm tp}:
        var uh = cp + j * 6 * m
        var ul = uh + m
        for i in range(m):
            ff_st(uh, ul, i, ff_ld(bh, bl, i * m + j))
        ff_chol_solve(luh, lul, m, uh, ul)
        ff_st(tp, tp + m, j, ff_ld(uh, ul, j))

    host_parallelize(tcol, m)
    var trq = ff_of(Float32(0))
    for j in range(m):
        trq = ff_add(trq, ff_ld(tp, tp + m, j))
    var trace_term = ff_div(ff_sub(ff_mul(ff_of(Float32(n)), ff_of(kdiag)), trq), nz)
    var total = ff_add(ff_add(ff_mul(ff_of(Float32(n)), ff_of(Float32(1.8378770664093453))), logdet), ff_add(quad, trace_term))
    info.unsafe_store(0, ff_f32(ff_mul(ff_of(Float32(-0.5)), total)))
    info.unsafe_store(1, Float32(1) if ok3 else Float32(0))
    _ = w^
    _ = sw^
    _ = cols^
    _ = trs^
