# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""x_decomp/lda_fit.mojo's fit loop on resident device matrices (lane
py-runtime-b): the driver text of lda_fit.mojo on `DKit`. An online epoch
is `lda_online_dev` (x_decomp/kit_device.mojo), as Python's `_online_pass`
called the GPU binding (the components go down and come back up around it);
the E-step takes the FAST Apple fused kernel (`launch_lda_fused_ss`) where
Python's `_fused_estep_ss` did. GPU binding only."""
from std.math import inf
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from x_decomp.api import _f, _n
from x_decomp.kit import (
    Mat, OP_ADD, OP_ADDS, OP_DIGAMMA, OP_DIV, OP_EXP, OP_LOGS, OP_MUL, OP_SCALE, OP_SUB, mat_from,
)
from x_decomp.kit_device import DKit, DMat, lda_online_dev
from x_decomp.lda_fit import LDA_F64_EPS, LDA_FLT_MIN, LdaArgs, _OP_LGAMMA
from x_decomp.lda_fast import LDA_FUSED_SS, LFS_BLOCKS, LFS_DPB, LFS_K_CAP, LFS_V_CAP, launch_lda_fused_ss
from x_decomp.resident import _ptr, pool_alloc, pool_free


def _online_epoch_dev(mut k: DKit, X: DMat, Xh: Mat, mut C: DMat, mut ED: DMat, a: LdaArgs,
                      mut draw: Int, mut nbi: Int) raises:
    var hc = k.get(C)
    var he = k.get(ED)
    k.sync()
    lda_online_dev(Xh, hc, he, a.bs, a.mdi, a.seed, draw, nbi, a.doc_prior, a.topic_prior, a.offset, a.decay,
                   a.tol, a.total)
    C = k.upload(hc)
    ED = k.upload(he)


def _e_ss_dev(mut k: DKit, X: DMat, ED: DMat, mut Dt: DMat, mut Et: DMat, a: LdaArgs, mut ss: DMat) raises -> Bool:
    """`_fused_estep_ss`: the FAST Apple fused E-step and statistics, or
    False (main's chain) past its caps or in any other build."""
    comptime if LDA_FUSED_SS:
        var n = X.r
        var nc = ED.r
        var v = ED.c
        if n < 1 or nc < 1 or v < 1 or nc > LFS_K_CAP or v > LFS_V_CAP:
            return False
        var groups = min(LFS_BLOCKS, (n + LFS_DPB - 1) // LFS_DPB)
        var kv = nc * v
        if n * v > 2147483647 or groups * kv > 2147483647:
            return False
        ss = DMat(nc, v)
        var sid = pool_alloc(groups * kv)
        launch_lda_fused_ss(k.ctx, X.p(), ED.p(), Dt.p(), Et.p(), _ptr(sid, groups * kv), ss.p(), n, nc, v,
                            Float32(a.doc_prior), a.mdi, Float32(a.tol), groups)
        k.ctx.synchronize()
        pool_free(sid)
        return True
    else:
        return False


def dirichlet2_dev(mut k: DKit, A: DMat) raises -> DMat:
    return k.ew2(OP_SUB, k.ew1(OP_DIGAMMA, A, 0.0), k.ew1(OP_DIGAMMA, k.rowsum(A), 0.0))


def e_step_dev(mut k: DKit, X: DMat, ED: DMat, nc: Int, a: LdaArgs, cal: Bool, rnd: Bool, mut draw: Int,
                    mut ss: DMat) raises -> DMat:
    """`_e_step`: Dt returned; ss set when cal."""
    var n = X.r
    var Dt: DMat
    if rnd:
        draw += 1
        Dt = k.ew1(OP_SCALE, k.rand_gamma(n, nc, a.seed, 61 + draw, 100.0), 0.01)
    else:
        Dt = k.const(1.0, n, nc)
    var Et = k.ew1(OP_EXP, dirichlet2_dev(k, Dt), 0.0)
    var fused = False
    if cal:
        fused = _e_ss_dev(k, X, ED, Dt, Et, a, ss)
    if not fused:
        k.lda_rows(X, ED, Dt, Et, a.doc_prior, a.mdi, a.tol)
    if cal and not fused:
        var norm_phi = k.ew1(OP_ADDS, k.mm(Et, ED, False, False), LDA_F64_EPS)
        var R = k.ew2(OP_DIV, X, norm_phi)
        ss = k.ew2(OP_MUL, k.mm(Et, R, True, False), ED)
    return Dt^


def loglik_dev(mut k: DKit, prior: Float64, distr: DMat, dirich: DMat, size: Int) raises -> Float64:
    """`_loglik`."""
    var s1 = k.word(k.total(k.ew2(OP_MUL, k.ew1(OP_SCALE, k.ew1(OP_ADDS, distr, -prior), -1.0), dirich)))
    var lgp = k.word(k.ew1(_OP_LGAMMA, k.const(prior, 1, 1), 0.0))
    var s2 = k.word(k.total(k.ew1(OP_ADDS, k.ew1(_OP_LGAMMA, distr, 0.0), -lgp)))
    var lg_ps = k.word(k.ew1(_OP_LGAMMA, k.const(prior * Float64(size), 1, 1), 0.0))
    var s3 = k.word(k.total(k.ew1(OP_SCALE, k.ew1(OP_ADDS, k.ew1(_OP_LGAMMA, k.rowsum(distr), 0.0), -lg_ps), -1.0)))
    return s1 + s2 + s3


def perplexity_dev(mut k: DKit, X: DMat, Dt: DMat, C: DMat, a: LdaArgs) raises -> Float64:
    """`_perplexity(k, M, Dt)` (no sub-sampling) with `_approx_bound`."""
    var nc = C.r
    var v = X.c
    var ddt = dirichlet2_dev(k, Dt)
    var dcomp = dirichlet2_dev(k, C)
    var score = k.word(k.total(k.lda_bound(X, ddt, dcomp, LDA_FLT_MIN)))
    score += loglik_dev(k, a.doc_prior, Dt, ddt, nc)
    score += loglik_dev(k, a.topic_prior, C, dcomp, v)
    var word_cnt = k.word(k.total(X))
    if word_cnt == 0.0:
        return inf[DType.float64]()
    return k.word(k.ew1(OP_EXP, k.const(-score / word_cnt, 1, 1), 0.0))


def lda_fit_loop_dev(mut k: DKit, X: DMat, Xh: Mat, mut C: DMat, mut ED: DMat, a: LdaArgs, mut draw: Int,
                          mut nbi: Int) raises -> Int:
    """The fit loop; returns n_iter_."""
    var nc = C.r
    var last = 0.0
    var have_last = False
    var it = 0
    for i in range(1, a.max_iter + 1):
        it = i
        if a.online:
            _online_epoch_dev(k, X, Xh, C, ED, a, draw, nbi)
        else:
            var ss = k.zeros(0, 0)
            _ = e_step_dev(k, X, ED, nc, a, True, True, draw, ss)
            C = k.ew1(OP_ADDS, ss, a.topic_prior)
            ED = k.ew1(OP_EXP, dirichlet2_dev(k, C), 0.0)
            nbi += 1
        if a.evaluate_every > 0 and i % a.evaluate_every == 0:
            var none = k.zeros(0, 0)
            var Dt = e_step_dev(k, X, ED, nc, a, False, False, draw, none)
            var bound = perplexity_dev(k, X, Dt, C, a)
            if have_last and abs(last - bound) < a.perp_tol:
                break
            last = bound
            have_last = True
    return it


def lda_fit_dev_py(
    x: PythonObject, comps: PythonObject, exp_dir: PythonObject, p: PythonObject, f: PythonObject
) raises -> PythonObject:
    """`lda_fit_py` on the resident kit."""
    var n = _n(p, 0)
    var v = _n(p, 1)
    var nc = _n(p, 2)
    var draw = Int(py=p[8])
    var nbi = Int(py=p[9])
    if v < 1 or nc < 1 or n * v > 2147483647 or nc * v > 2147483647:
        raise Error("x_decomp: lda fit shape out of range")
    var a = LdaArgs(p, f)
    if a.online and a.bs < 1:
        raise Error("x_decomp: lda batch size")
    var px = _f(x)
    var pc = _f(comps)
    var pe = _f(exp_dir)
    var it = 0
    with GILReleased(Python()):
        var k = DKit()
        var Xh = mat_from(px, n, v)
        var X = k.upload(Xh)
        var C = k.upload(mat_from(pc, nc, v))
        var ED = k.upload(mat_from(pe, nc, v))
        it = lda_fit_loop_dev(k, X, Xh, C, ED, a, draw, nbi)
        var hc = k.get(C)
        var he = k.get(ED)
        k.sync()
        for i in range(nc * v):
            pc.unsafe_store(i, hc.d[i])
            pe.unsafe_store(i, he.d[i])
    return Python.tuple(it, draw, nbi)
