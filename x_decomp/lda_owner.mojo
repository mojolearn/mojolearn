# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`LatentDirichletAllocation`'s fit, partial_fit, transform, score and
perplexity as ONE native operation each (lane mca2-decomp-a, 2026-10-22):
the steps `_expansion_decomp.LatentDirichletAllocation._check_X`, `_init`,
`fit` (`x_decomp_lda_fit`), `_online_pass` (`x_decomp_lda_online`),
`_e_step` (no sufficient statistics: every public path), `_approx_bound`,
`_loglik` and `_perplexity` sequenced from Python with the decomp kit, on
`RKit` (x_decomp/decomp_owner_kit.mojo): the same entries' typed bodies,
operands, Philox streams and draw counters, float32 scalars and binary64
bound arithmetic, resident where the Python kit was (`DecompDev.lda_rows`,
`lda_bound` on a GPU binding)."""
from std.memory import bitcast

from x_decomp.api import lda_rows_run, rand_gamma_run
from x_decomp.cells import F32Ptr
from x_decomp.chi2 import lda_bound_host_run
from x_decomp.exec_trait import Exec
from x_decomp.lda_fit import LdaArgs
from x_decomp.pca_randomized import RDev, RKit
from x_decomp.decomp_owner_kit import DecompDev, OP_ADDS, OP_DIGAMMA, OP_DIV, OP_EXP, OP_MUL, OP_SCALE, OP_SUB

comptime _OP_LGAMMA = 37
comptime _OP_GTS = 23
comptime _F32_TINY = 1.1754943508222875e-38
#: the operations
comptime LDA_FIT = 0
comptime LDA_PARTIAL = 1
comptime LDA_TRANSFORM = 2
comptime LDA_SCORE = 3
comptime LDA_PERPLEXITY = 4


def _dirichlet[E: Exec, D: RDev, RES: Bool](mut kit: RKit[E, D, RES], a: Int) raises -> Int:
    """`_dirichlet_expectation_2d(k, A)`: digamma(A) - digamma(rowsum(A))."""
    var dg = kit.ew(OP_DIGAMMA, a)
    var rows = kit.sum(a, False)
    var dr = kit.ew(OP_DIGAMMA, rows)
    return kit.ew(OP_SUB, dg, dr)


def _negative[E: Exec, D: RDev, RES: Bool](mut kit: RKit[E, D, RES], m: Int) raises -> Bool:
    """`_any_negative(M)`."""
    if kit.r(m) * kit.c(m) == 0:
        return False
    var negated = kit.ew(OP_SCALE, m, scalar=-1.0)
    var flags = kit.ew(_OP_GTS, negated, scalar=0.0)
    return kit.total(flags) > 0


def _e_step[E: Exec, D: RDev, RES: Bool, X: DecompDev](
    mut kit: RKit[E, D, RES], m: Int, exp_dir: Int, nc: Int, prior: Float64, max_iter: Int, tol: Float64
) raises -> Int:
    """`_e_step(k, X, random_init=False, cal_sstats=False)`: Dt (n x nc)."""
    var n = kit.r(m)
    var v = kit.c(m)
    # k.const(1.0, n, nc): host ones (0 + 1 by one cell)
    var blank = kit.zeros(n, nc)
    var dt = kit.ew(OP_ADDS, blank, scalar=1.0)
    var dd = _dirichlet(kit, dt)
    var et = kit.ew(OP_EXP, dd)
    # `_Kit.lda_rows`: in place on the device for a GPU binding, else the host rows
    var resident = False
    comptime if RES:
        resident = n > 0
    if resident:
        var dx = kit.device(m)
        var dw = kit.device(exp_dir)
        var ddt = kit.device(dt)
        var de = kit.device(et)
        X.lda_rows(dx, dw, ddt, de, n, nc, v, max_iter, prior, tol)
        return dt
    var its = kit.zeros(n, 1)
    lda_rows_run[E](kit.ptr(m), kit.ptr(exp_dir), kit.ptr(dt), kit.ptr(et), kit.ptr(its), n, nc, v, max_iter,
                    Float32(prior), Float32(tol))
    return dt


def _scalar_lgamma[E: Exec, D: RDev, RES: Bool](mut kit: RKit[E, D, RES], value: Float64) raises -> Float64:
    """`k.ew("lgamma", k.const(value)).s[0]`."""
    var c = kit.constant(Float32(value))
    var g = kit.ew(_OP_LGAMMA, c)
    return kit.word(g)


def _loglik[E: Exec, D: RDev, RES: Bool](
    mut kit: RKit[E, D, RES], prior: Float64, distr: Int, dirichlet: Int, size: Int
) raises -> Float64:
    """`_loglik(k, prior, distr, dirichlet, size)`."""
    var shifted = kit.ew(OP_ADDS, distr, scalar=-prior)
    var negated = kit.ew(OP_SCALE, shifted, scalar=-1.0)
    var weighted = kit.ew(OP_MUL, negated, dirichlet)
    var s1 = kit.total(weighted)
    var lg = kit.ew(_OP_LGAMMA, distr)
    var lg_prior = _scalar_lgamma(kit, prior)
    var lg_shift = kit.ew(OP_ADDS, lg, scalar=-lg_prior)
    var s2 = kit.total(lg_shift)
    var lg_ps = _scalar_lgamma(kit, prior * Float64(size))
    var rows = kit.sum(distr, False)
    var lg_rows = kit.ew(_OP_LGAMMA, rows)
    var lg_rows_shift = kit.ew(OP_ADDS, lg_rows, scalar=-lg_ps)
    var lg_rows_neg = kit.ew(OP_SCALE, lg_rows_shift, scalar=-1.0)
    var s3 = kit.total(lg_rows_neg)
    return s1 + s2 + s3


def _approx_bound[E: Exec, D: RDev, RES: Bool, X: DecompDev](
    mut kit: RKit[E, D, RES], m: Int, dt: Int, components: Int, doc_prior: Float64, topic_prior: Float64,
    sub_sampling: Bool, total_samples: Float64,
) raises -> Float64:
    """`_approx_bound(k, M, Dt, sub_sampling)`."""
    var n = kit.r(m)
    var v = kit.c(m)
    var nc = kit.r(components)
    var ddt = _dirichlet(kit, dt)
    var dcomp = _dirichlet(kit, components)
    var score: Float64
    var resident = False
    comptime if RES:
        resident = n * v > 0
    if resident:
        var p = kit.device_output(n, v)
        var dx = kit.device(m)
        var dd = kit.device(ddt)
        var dc = kit.device(dcomp)
        X.lda_bound(dx, dd, dc, kit.mats[p].dev, n, nc, v, _F32_TINY)
        score = kit.total(p)
    else:
        var p = kit.zeros(n, v)
        if n * v:
            _ = lda_bound_host_run(kit.ptr(m), kit.ptr(ddt), kit.ptr(dcomp), kit.ptr(p), n, nc, v, _F32_TINY)
        score = kit.total(p)
    score += _loglik(kit, doc_prior, dt, ddt, nc)
    if sub_sampling:
        score *= total_samples / Float64(n)
    score += _loglik(kit, topic_prior, components, dcomp, v)
    return score


def _init[E: Exec, D: RDev, RES: Bool](
    mut kit: RKit[E, D, RES], nc: Int, v: Int, seed: Int
) raises -> Tuple[Int, Int]:
    """`_init(k, d)`: (components, exp_dir) from the stream-60 gamma draws."""
    var drawn = kit.zeros(nc, v)
    if nc * v:
        rand_gamma_run[E](kit.ptr(drawn), nc * v, seed, 60, Float32(100.0))
    var components = kit.ew(OP_SCALE, drawn, scalar=0.01)
    var dd = _dirichlet(kit, components)
    var exp_dir = kit.ew(OP_EXP, dd)
    return (components, exp_dir)


def lda_run[E: Exec, D: RDev, RES: Bool, X: DecompDev](
    op: Int, x: F32Ptr, n: Int, v: Int, nc: Int, fresh: Bool, a: LdaArgs, mut draw: Int, mut nbi: Int,
    flag: Bool, comps_in: F32Ptr, exp_in: F32Ptr, comps_out: F32Ptr, exp_out: F32Ptr, out: F32Ptr,
    negative_message: String,
) raises -> Tuple[Int, Float64]:
    """One LatentDirichletAllocation operation on X (n x v): LDA_FIT,
    LDA_PARTIAL (`fresh`: initialize first), LDA_TRANSFORM (`flag`:
    normalize; n x nc to out), LDA_SCORE, LDA_PERPLEXITY (`flag`:
    sub_sampling). The fitted state is comps_in / exp_in (nc x v) and is
    written to comps_out / exp_out by the fits. Returns (n_iter_, value):
    the fit's bound, the score or the perplexity."""
    var kit = RKit[E, D, RES]()
    try:
        var m = kit.add(Int(x), -1, n, v)
        if _negative(kit, m):
            raise Error(negative_message)
        var components: Int
        var exp_dir: Int
        if op == LDA_FIT or (op == LDA_PARTIAL and fresh):
            var started = _init(kit, nc, v, a.seed)
            components = started[0]
            exp_dir = started[1]
        else:
            components = kit.add(Int(comps_in), -1, nc, v)
            exp_dir = kit.add(Int(exp_in), -1, nc, v)
        var it = 0
        var value = Float64(0.0)
        if op == LDA_FIT:
            # the fit loop on host copies (`C, E = ....copy()`)
            var c = kit.copy_host(components)
            var e = kit.copy_host(exp_dir)
            it = X.lda_fit(kit.ptr(m), kit.ptr(c), kit.ptr(e), n, v, nc, a, draw, nbi)
            var dt = _e_step[E, D, RES, X](kit, m, e, nc, a.doc_prior, a.mdi, a.tol)
            # `_perplexity(k, M, Dt)` (no sub-sampling)
            var bound = _approx_bound[E, D, RES, X](kit, m, dt, c, a.doc_prior, a.topic_prior, False, a.total)
            var word_cnt = kit.total(m)
            if word_cnt == 0:
                value = bitcast_inf()
            else:
                var ratio = kit.constant(Float32(-bound / word_cnt))
                var perp = kit.ew(OP_EXP, ratio)
                value = kit.word(perp)
            kit.copy_out(c, comps_out, nc * v)
            kit.copy_out(e, exp_out, nc * v)
        elif op == LDA_PARTIAL:
            # `_online_pass`: the pass on the state's host words, in place
            var fv = List[Float64]()
            fv.append(a.doc_prior)
            fv.append(a.topic_prior)
            fv.append(a.offset)
            fv.append(a.decay)
            fv.append(a.tol)
            fv.append(a.total)
            var pm = kit.ptr(m)
            var pc = kit.ptr(components)
            var pe = kit.ptr(exp_dir)
            X.lda_online(pm, pc, pe, n, v, nc, a.bs, a.mdi, a.seed, draw, nbi, fv)
            kit.copy_out(components, comps_out, nc * v)
            kit.copy_out(exp_dir, exp_out, nc * v)
        else:
            var dt = _e_step[E, D, RES, X](kit, m, exp_dir, nc, a.doc_prior, a.mdi, a.tol)
            if op == LDA_TRANSFORM:
                if flag:
                    var rows = kit.sum(dt, False)
                    dt = kit.ew(OP_DIV, dt, rows)
                kit.copy_out(dt, out, n * nc)
            elif op == LDA_SCORE:
                value = _approx_bound[E, D, RES, X](kit, m, dt, components, a.doc_prior, a.topic_prior, False,
                                                    a.total)
            else:
                var bound = _approx_bound[E, D, RES, X](kit, m, dt, components, a.doc_prior, a.topic_prior, flag,
                                                        a.total)
                var word_cnt = kit.total(m)
                if flag:
                    word_cnt *= a.total / Float64(n)
                if word_cnt == 0:
                    value = bitcast_inf()
                else:
                    var ratio = kit.constant(Float32(-bound / word_cnt))
                    var perp = kit.ew(OP_EXP, ratio)
                    value = kit.word(perp)
        kit.close()
        return (it, value)
    except error:
        kit.close()
        raise error^


def bitcast_inf() -> Float64:
    """`math.inf`."""
    return bitcast[DType.float64](UInt64(0x7FF0000000000000))
