# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""LatentDirichletAllocation's fit loop in Mojo (lane py-runtime-b,
2026-10-05): `_expansion_decomp.LatentDirichletAllocation.fit`'s
`for it in range(1, max_iter + 1)` with `_em_step` (batch), `_e_step`,
`_perplexity`, `_approx_bound` and `_loglik`; an online epoch is
x_decomp/lda_online.mojo's pass, as Python's `_online_pass` called it.
Statement for statement on `Kit[E]`: the same cells, draws (61 + draw),
broadcast modes and float32 scalars, and Python's float64 bound arithmetic
and stopping test in the same order, so the IDENTICAL words are the Python
driver's. x_decomp/lda_fit_dev.mojo is the same driver text on DKit."""
from std.math import inf
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from x_decomp.api import _f, _n
from x_decomp.exec_trait import Exec
from x_decomp.kit import (
    Kit, Mat, OP_ADD, OP_ADDS, OP_DIGAMMA, OP_DIV, OP_EXP, OP_LOGS, OP_MUL, OP_SCALE, OP_SUB, mat_from,
)
from x_decomp.lda_online import lda_online_pass

comptime LDA_F64_EPS: Float64 = 2.220446049250313e-16
comptime LDA_FLT_MIN: Float64 = 1.1754943508222875e-38
comptime _OP_LGAMMA = 37


struct LdaArgs(Copyable, Movable):
    var max_iter: Int
    var online: Bool
    var bs: Int
    var mdi: Int
    var seed: Int
    var evaluate_every: Int
    var doc_prior: Float64
    var topic_prior: Float64
    var offset: Float64
    var decay: Float64
    var tol: Float64
    var total: Float64
    var perp_tol: Float64

    def __init__(out self, p: PythonObject, f: PythonObject) raises:
        """p = [n, v, nc, max_iter, online, bs, mdi, seed, draw, nbi,
        evaluate_every]; f = [doc_prior, topic_prior, offset, decay,
        mean_change_tol, total_samples, perp_tol]."""
        self.max_iter = Int(py=p[3])
        self.online = Int(py=p[4]) != 0
        self.bs = Int(py=p[5])
        self.mdi = Int(py=p[6])
        self.seed = Int(py=p[7])
        self.evaluate_every = Int(py=p[10])
        self.doc_prior = Float64(py=f[0])
        self.topic_prior = Float64(py=f[1])
        self.offset = Float64(py=f[2])
        self.decay = Float64(py=f[3])
        self.tol = Float64(py=f[4])
        self.total = Float64(py=f[5])
        self.perp_tol = Float64(py=f[6])


def _online_epoch[E: Exec](mut k: Kit[E], X: Mat, Xh: Mat, mut C: Mat, mut ED: Mat, a: LdaArgs,
                           mut draw: Int, mut nbi: Int) raises:
    lda_online_pass[E](Xh, C, ED, a.bs, a.mdi, a.seed, draw, nbi, a.doc_prior, a.topic_prior, a.offset, a.decay,
                       a.tol, a.total)


def _e_ss[E: Exec](mut k: Kit[E], X: Mat, ED: Mat, mut Dt: Mat, mut Et: Mat, a: LdaArgs, mut ss: Mat) raises -> Bool:
    """The fused E-step (FAST Apple GPU only): never on the host column."""
    return False


# ---- driver (the same text on DKit in x_decomp/lda_fit_dev.mojo)
def dirichlet2[E: Exec](mut k: Kit[E], A: Mat) raises -> Mat:
    return k.ew2(OP_SUB, k.ew1(OP_DIGAMMA, A, 0.0), k.ew1(OP_DIGAMMA, k.rowsum(A), 0.0))


def e_step[E: Exec](mut k: Kit[E], X: Mat, ED: Mat, nc: Int, a: LdaArgs, cal: Bool, rnd: Bool, mut draw: Int,
                    mut ss: Mat) raises -> Mat:
    """`_e_step`: Dt returned; ss set when cal."""
    var n = X.r
    var Dt: Mat
    if rnd:
        draw += 1
        Dt = k.ew1(OP_SCALE, k.rand_gamma(n, nc, a.seed, 61 + draw, 100.0), 0.01)
    else:
        Dt = k.const(1.0, n, nc)
    var Et = k.ew1(OP_EXP, dirichlet2(k, Dt), 0.0)
    var fused = False
    if cal:
        fused = _e_ss(k, X, ED, Dt, Et, a, ss)
    if not fused:
        k.lda_rows(X, ED, Dt, Et, a.doc_prior, a.mdi, a.tol)
    if cal and not fused:
        var norm_phi = k.ew1(OP_ADDS, k.mm(Et, ED, False, False), LDA_F64_EPS)
        var R = k.ew2(OP_DIV, X, norm_phi)
        ss = k.ew2(OP_MUL, k.mm(Et, R, True, False), ED)
    return Dt^


def loglik[E: Exec](mut k: Kit[E], prior: Float64, distr: Mat, dirich: Mat, size: Int) raises -> Float64:
    """`_loglik`."""
    var s1 = k.word(k.total(k.ew2(OP_MUL, k.ew1(OP_SCALE, k.ew1(OP_ADDS, distr, -prior), -1.0), dirich)))
    var lgp = k.word(k.ew1(_OP_LGAMMA, k.const(prior, 1, 1), 0.0))
    var s2 = k.word(k.total(k.ew1(OP_ADDS, k.ew1(_OP_LGAMMA, distr, 0.0), -lgp)))
    var lg_ps = k.word(k.ew1(_OP_LGAMMA, k.const(prior * Float64(size), 1, 1), 0.0))
    var s3 = k.word(k.total(k.ew1(OP_SCALE, k.ew1(OP_ADDS, k.ew1(_OP_LGAMMA, k.rowsum(distr), 0.0), -lg_ps), -1.0)))
    return s1 + s2 + s3


def perplexity[E: Exec](mut k: Kit[E], X: Mat, Dt: Mat, C: Mat, a: LdaArgs) raises -> Float64:
    """`_perplexity(k, M, Dt)` (no sub-sampling) with `_approx_bound`."""
    var nc = C.r
    var v = X.c
    var ddt = dirichlet2(k, Dt)
    var dcomp = dirichlet2(k, C)
    var score = k.word(k.total(k.lda_bound(X, ddt, dcomp, LDA_FLT_MIN)))
    score += loglik(k, a.doc_prior, Dt, ddt, nc)
    score += loglik(k, a.topic_prior, C, dcomp, v)
    var word_cnt = k.word(k.total(X))
    if word_cnt == 0.0:
        return inf[DType.float64]()
    return k.word(k.ew1(OP_EXP, k.const(-score / word_cnt, 1, 1), 0.0))


def lda_fit_loop[E: Exec](mut k: Kit[E], X: Mat, Xh: Mat, mut C: Mat, mut ED: Mat, a: LdaArgs, mut draw: Int,
                          mut nbi: Int) raises -> Int:
    """The fit loop; returns n_iter_."""
    var nc = C.r
    var last = 0.0
    var have_last = False
    var it = 0
    for i in range(1, a.max_iter + 1):
        it = i
        if a.online:
            _online_epoch(k, X, Xh, C, ED, a, draw, nbi)
        else:
            var ss = k.zeros(0, 0)
            _ = e_step(k, X, ED, nc, a, True, True, draw, ss)
            C = k.ew1(OP_ADDS, ss, a.topic_prior)
            ED = k.ew1(OP_EXP, dirichlet2(k, C), 0.0)
            nbi += 1
        if a.evaluate_every > 0 and i % a.evaluate_every == 0:
            var none = k.zeros(0, 0)
            var Dt = e_step(k, X, ED, nc, a, False, False, draw, none)
            var bound = perplexity(k, X, Dt, C, a)
            if have_last and abs(last - bound) < a.perp_tol:
                break
            last = bound
            have_last = True
    return it
# ---- end driver


def lda_fit_py[E: Exec](
    x: PythonObject, comps: PythonObject, exp_dir: PythonObject, p: PythonObject, f: PythonObject
) raises -> PythonObject:
    """x (n x v); comps and exp_dir (nc x v) replaced. Returns (n_iter,
    draw, n_batch_iter)."""
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
        var k = Kit[E]()
        var X = mat_from(px, n, v)
        var C = mat_from(pc, nc, v)
        var ED = mat_from(pe, nc, v)
        it = lda_fit_loop(k, X, X, C, ED, a, draw, nbi)
        for i in range(nc * v):
            pc.unsafe_store(i, C.d[i])
            pe.unsafe_store(i, ED.d[i])
    return Python.tuple(it, draw, nbi)
