# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""LatentDirichletAllocation's online pass in Mojo (lane/py-decomp-nbrs,
2026-09-28).

Before this file, `_expansion_decomp.LatentDirichletAllocation` called
`_em_step` from Python once per mini-batch (about 15 kit calls each, 23 s
per epoch at 1M rows). `lda_online_pass` is that loop, statement for
statement: `_e_step(random_init=True)`, the (offset + n_batch_iter)^-decay
weight through the cells' logs and exp, the blend of `components_` and the
new `exp_dirichlet_component_`, with the same cells, broadcast modes, draw
streams (61 + draw) and float32 scalars (each Python double rounded once,
as the binding boundary rounds it). The executors are `x_decomp/kit.mojo`'s.
"""
from x_decomp.exec_trait import Exec
from x_decomp.kit import (
    Kit, Mat, OP_ADD, OP_ADDS, OP_DIGAMMA, OP_DIV, OP_EXP, OP_LOGS, OP_MUL, OP_SCALE, OP_SUB, mat_const, mat_rows,
)

#: `_expansion_decomp._F64_EPS`, the `adds` of `norm_phi`
comptime _F64_EPS: Float64 = 2.220446049250313e-16


def dirichlet_expectation_2d[E: Exec, S: Exec](k: Kit[E, S], A: Mat) raises -> Mat:
    """psi(A) - psi(rowsum(A))."""
    return k.ew2(OP_SUB, k.ew1(OP_DIGAMMA, A, 0.0), k.ew1(OP_DIGAMMA, k.rowsum(A), 0.0))


def lda_online_pass[E: Exec, S: Exec](
    X: Mat, mut comps: Mat, mut exp_dir: Mat, bs: Int, max_doc_iter: Int, seed: Int, mut draw: Int,
    mut n_batch_iter: Int, doc_prior: Float64, topic_prior: Float64, offset: Float64, decay: Float64,
    tol: Float64, total_samples: Float64, dev: Int,
) raises:
    """One `for a in range(0, n, bs): self._em_step(k, M.rows(a, b), total, False)`
    pass. `comps` (k x v) and `exp_dir` are replaced; `draw` and
    `n_batch_iter` advance as the Python attributes did."""
    var k = Kit[E, S](dev)
    var n = X.r
    var nc = comps.r
    var a = 0
    while a < n:
        var b = min(a + bs, n)
        var Xb = mat_rows(X, a, b)
        # _e_step(k, Xb, cal_sstats=True, random_init=True)
        draw += 1
        var Dt = k.ew1(OP_SCALE, k.rand_gamma(Xb.r, nc, seed, 61 + draw, 100.0), 0.01)
        var Et = k.ew1(OP_EXP, dirichlet_expectation_2d(k, Dt), 0.0)
        k.lda_rows(Xb, exp_dir, Dt, Et, doc_prior, max_doc_iter, tol)
        var norm_phi = k.ew1(OP_ADDS, k.mm(Et, exp_dir, False, False), _F64_EPS)
        var R = k.ew2(OP_DIV, Xb, norm_phi)
        var ss = k.ew2(OP_MUL, k.mm(Et, R, True, False), exp_dir)
        # the online update (batch_update=False)
        var wt = k.ew1(
            OP_EXP,
            k.ew1(OP_SCALE, k.ew1(OP_LOGS, mat_const(offset + Float64(n_batch_iter), 1, 1), 1e-30), -decay),
            0.0,
        )
        var weight = Float64(wt.d[0])
        var doc_ratio = total_samples / Float64(Xb.r)
        var upd = k.ew1(OP_ADDS, k.ew1(OP_SCALE, ss, doc_ratio), topic_prior)
        comps = k.ew2(OP_ADD, k.ew1(OP_SCALE, comps, 1.0 - weight), k.ew1(OP_SCALE, upd, weight))
        exp_dir = k.ew1(OP_EXP, dirichlet_expectation_2d(k, comps), 0.0)
        n_batch_iter += 1
        a = b
