# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The CPU column of glm/impl/gram_solve.mojo (MOJOLEARN_CLASSICAL_LINEAR_GRAM_SOLVE):
the same items in the same order. Exact column sums and binary64 means
(glm/host/center_host.mojo, as `HostExec.ols_tsqr_factor`); the centered
Gram and cross by the reference cells the device's leaf kernels reproduce
(`centered_gram_v1_cell`, `centered_cross_v1_cell`, leaves of
`contract_leaf_size(n)` rows, the binary-counter fold), one task per cell;
the same equilibration, `chol_serial`, the same trust gate and the same
triangular-solve statements. No GPU import."""
from std.memory import bitcast
from checks.numerics import ftz, identical_mul, identical_mul_add
from core.classical_centered import centered_cross_v1_cell, centered_gram_v1_cell
from core.host_parallel import host_parallelize
from glm.host.center_host import col_sums_on_cpu
from glm.host.glm_oracle import host_equilibration_scale
from glm.impl.gram_solve_cells import gs_equilibrated_cell, gs_pivot_trusted, gs_regularized_diag, gs_scaled_rhs
from x_decomp.cells import F32Ptr, chol_serial, div0
from experiments.classical_identical_ideas.fg_linear_controls import IDN_GRAM_FF_FALLBACK
from glm.impl.gram_ff_cells import (
    GFF_LEAVES,
    gff_backward_cell,
    gff_cells,
    gff_chol_cell,
    gff_coef,
    gff_equilibrated_cell,
    gff_fold_cell,
    gff_forward_cell,
    gff_leaf_cell,
    gff_mean_split,
    gff_pivot_trusted,
    gff_regularized_diag_f32,
)
from x_linear.ff import ff_ld, ff_mul_f, ff_st

comptime _AP = MutPointer[Float32, MutAnyOrigin]


def host_linear_gram_fit(
    x: List[Float32],
    y: List[Float32],
    n_rows: Int,
    n_features: Int,
    alpha: Float32,
    center: Bool,
    coef_ptr: MutPointer[Float32, MutUntrackedOrigin],
    mu_ptr: MutPointer[Float32, MutUntrackedOrigin],
    ymean_ptr: MutPointer[Float64, MutUntrackedOrigin],
) raises -> Int:
    """`linear_gram_fit_host`'s words on the host: 0 and coef (and, with
    `center`, mu and the y mean) written, or 1 with nothing written."""
    if n_features <= 0 or n_rows <= 0:
        raise Error("linear_gram_fit: n_rows and n_features must be positive")
    if alpha < Float32(0.0):
        raise Error("linear_gram_fit: alpha must be non-negative")
    var d = n_features
    var xp = _AP(unsafe_from_address=Int(x.unsafe_ptr()))
    var yp = _AP(unsafe_from_address=Int(y.unsafe_ptr()))
    var means = List[Float32](length=d, fill=Float32(0.0))
    var ymean32 = Float32(0.0)
    # the binary64 mean bits (X then y), zero when not centering: lane
    # fg-linear L3's float-float means (the device keeps d_m64 the same way)
    var m64 = List[UInt64](length=d + 1, fill=UInt64(0))
    if center:
        var sums = List[UInt64](length=d + 1, fill=UInt64(0))
        var sp = Int(sums.unsafe_ptr())
        col_sums_on_cpu(Int(x.unsafe_ptr()), sp, n_rows, d)
        col_sums_on_cpu(Int(y.unsafe_ptr()), sp + 8 * d, n_rows, 1)
        for j in range(d + 1):  # small-loop(d + 1: feature count): the means' narrowing, the fit's outputs
            var mean = bitcast[DType.float64](sums[j]) / Float64(n_rows)
            m64[j] = bitcast[DType.uint64](mean)
            if j < d:
                means[j] = mean.cast[DType.float32]()
                mu_ptr.unsafe_store(j, means[j])
            else:
                ymean32 = mean.cast[DType.float32]()
                ymean_ptr.unsafe_store(0, mean)
        _ = sums^
    var mp = _AP(unsafe_from_address=Int(means.unsafe_ptr()))
    var g = List[Float32](length=d * d, fill=Float32(0.0))
    var c = List[Float32](length=d, fill=Float32(0.0))
    var gp = _AP(unsafe_from_address=Int(g.unsafe_ptr()))
    var cp = _AP(unsafe_from_address=Int(c.unsafe_ptr()))

    def _gram_cell(q: Int) {imm gp, imm xp, imm mp, imm n_rows, imm d}:
        var i = q // d
        var j = q - i * d
        gp.unsafe_store(q, centered_gram_v1_cell(xp, mp, n_rows, d, i, j))

    def _cross_cell(j: Int) {imm cp, imm xp, imm mp, imm yp, imm ymean32, imm n_rows, imm d}:
        cp.unsafe_store(j, centered_cross_v1_cell(xp, mp, yp, ymean32, n_rows, d, j))

    host_parallelize(_gram_cell, d * d)
    host_parallelize(_cross_cell, d)
    var a = List[Float32](length=d * d, fill=Float32(0.0))
    var adiag = List[Float32](length=d, fill=Float32(0.0))
    var b = List[Float32](length=d, fill=Float32(0.0))
    var sv = List[Float32](length=d, fill=Float32(0.0))
    var fg = F32Ptr(unsafe_from_address=Int(g.unsafe_ptr()))
    var fc = F32Ptr(unsafe_from_address=Int(c.unsafe_ptr()))
    for i in range(d):
        sv[i] = host_equilibration_scale(gs_regularized_diag(g[i * d + i], alpha))
    for q in range(d * d):
        var i = q // d
        var j = q - i * d
        var v = gs_equilibrated_cell(fg, i, j, d, alpha, sv[i], sv[j])
        a[q] = v
        if i == j:
            adiag[i] = v
            b[i] = gs_scaled_rhs(fc, i, sv[i])
    var info = List[Float32](length=1, fill=Float32(0.0))
    var fa = F32Ptr(unsafe_from_address=Int(a.unsafe_ptr()))
    chol_serial(fa, d, F32Ptr(unsafe_from_address=Int(info.unsafe_ptr())))
    var ok = info[0] == Float32(0)
    for j in range(d):
        if not gs_pivot_trusted(a[j * d + j], adiag[j]):
            ok = False
    if not ok:
        comptime if IDN_GRAM_FF_FALLBACK:
            # lane fg-linear L3: the device's float-float second chance for
            # a rejected Ridge Gram, the same cells in the same order
            if alpha > Float32(0.0):
                return _host_gff_fit(x, y, m64, n_rows, d, alpha, coef_ptr)
        return 1
    # forward L z = b (column steps, rows below in order), then L^T w = z:
    # the device kernels' statements, z and w in their own vectors
    var z = List[Float32](length=d, fill=Float32(0.0))
    var w = List[Float32](length=d, fill=Float32(0.0))
    for j in range(d):
        var zj = div0(ftz(b[j]), a[j * d + j])
        for i in range(j + 1, d):
            b[i] = ftz(identical_mul_add(-ftz(a[i * d + j]), zj, ftz(b[i])))
        z[j] = zj
    for jj in range(d):
        var j = d - 1 - jj
        var wj = div0(ftz(z[j]), a[j * d + j])
        for i in range(j):
            z[i] = ftz(identical_mul_add(-ftz(a[j * d + i]), wj, ftz(z[i])))
        w[j] = wj
    for i in range(d):
        coef_ptr.unsafe_store(i, ftz(identical_mul(ftz(sv[i]), ftz(w[i]))))
    return 0


def _host_gff_fit(
    x: List[Float32],
    y: List[Float32],
    m64: List[UInt64],
    n_rows: Int,
    d: Int,
    alpha: Float32,
    coef_ptr: MutPointer[Float32, MutUntrackedOrigin],
) raises -> Int:
    """`_gs_ff_fit` (glm/impl/gram_solve.mojo) on the host: 0 with coef
    written when the float-float factor is trusted, else 1. The leaf cells
    run in parallel (each its own chain), every other step in the device's
    column order."""
    var t = gff_cells(d)
    var mh = List[Float32](length=d + 1, fill=Float32(0.0))
    var ml = List[Float32](length=d + 1, fill=Float32(0.0))
    for j in range(d + 1):  # small-loop(d + 1: feature count): the means' float-float split
        var f = gff_mean_split(m64[j])
        mh[j] = f.hi
        ml[j] = f.lo
    var ph = List[Float32](length=GFF_LEAVES * t, fill=Float32(0.0))
    var pl = List[Float32](length=GFF_LEAVES * t, fill=Float32(0.0))
    var xp = _AP(unsafe_from_address=Int(x.unsafe_ptr()))
    var yp = _AP(unsafe_from_address=Int(y.unsafe_ptr()))
    var mhp = _AP(unsafe_from_address=Int(mh.unsafe_ptr()))
    var mlp = _AP(unsafe_from_address=Int(ml.unsafe_ptr()))
    var php = _AP(unsafe_from_address=Int(ph.unsafe_ptr()))
    var plp = _AP(unsafe_from_address=Int(pl.unsafe_ptr()))

    def _leaf_task(u: Int) {imm xp, imm yp, imm mhp, imm mlp, imm php, imm plp, imm n_rows, imm d, imm t}:
        var leaf = u // t
        var q = u - leaf * t
        var v = gff_leaf_cell(xp, yp, mhp, mlp, n_rows, d, leaf, q)
        php.unsafe_store(u, v.hi)
        plp.unsafe_store(u, v.lo)

    host_parallelize(_leaf_task, GFF_LEAVES * t)
    var gh = List[Float32](length=d * d, fill=Float32(0.0))
    var gl = List[Float32](length=d * d, fill=Float32(0.0))
    var ch = List[Float32](length=d, fill=Float32(0.0))
    var cl = List[Float32](length=d, fill=Float32(0.0))
    var ghp = _AP(unsafe_from_address=Int(gh.unsafe_ptr()))
    var glp = _AP(unsafe_from_address=Int(gl.unsafe_ptr()))
    var chp = _AP(unsafe_from_address=Int(ch.unsafe_ptr()))
    var clp = _AP(unsafe_from_address=Int(cl.unsafe_ptr()))
    for q in range(t):
        gff_fold_cell(php, plp, ghp, glp, chp, clp, d, q)
    var ah = List[Float32](length=d * d, fill=Float32(0.0))
    var al = List[Float32](length=d * d, fill=Float32(0.0))
    var adh = List[Float32](length=d, fill=Float32(0.0))
    var adl = List[Float32](length=d, fill=Float32(0.0))
    var bh = List[Float32](length=d, fill=Float32(0.0))
    var bl = List[Float32](length=d, fill=Float32(0.0))
    var sv = List[Float32](length=d, fill=Float32(0.0))
    var ahp = _AP(unsafe_from_address=Int(ah.unsafe_ptr()))
    var alp = _AP(unsafe_from_address=Int(al.unsafe_ptr()))
    var adhp = _AP(unsafe_from_address=Int(adh.unsafe_ptr()))
    var adlp = _AP(unsafe_from_address=Int(adl.unsafe_ptr()))
    var bhp = _AP(unsafe_from_address=Int(bh.unsafe_ptr()))
    var blp = _AP(unsafe_from_address=Int(bl.unsafe_ptr()))
    var svp = _AP(unsafe_from_address=Int(sv.unsafe_ptr()))
    for i in range(d):
        sv[i] = host_equilibration_scale(gff_regularized_diag_f32(ghp, glp, i, d, alpha))
    for q in range(d * d):
        var i = q // d
        var j = q - i * d
        var v = gff_equilibrated_cell(ghp, glp, i, j, d, alpha, sv[i], sv[j])
        ff_st(ahp, alp, q, v)
        if i == j:
            ff_st(adhp, adlp, i, v)
            ff_st(bhp, blp, i, ff_mul_f(ff_ld(chp, clp, i), sv[i]))
    var info = List[Float32](length=1, fill=Float32(0.0))
    var infop = _AP(unsafe_from_address=Int(info.unsafe_ptr()))
    for j in range(d):
        for i in range(j, d):
            gff_chol_cell(ahp, alp, adhp, adlp, infop, i, j, d)
    var ok = info[0] == Float32(0)
    for j in range(d):
        if not gff_pivot_trusted(ah[j * d + j], adh[j]):
            ok = False
    if not ok:
        return 1
    var zh = List[Float32](length=d, fill=Float32(0.0))
    var zl = List[Float32](length=d, fill=Float32(0.0))
    var wh = List[Float32](length=d, fill=Float32(0.0))
    var wl = List[Float32](length=d, fill=Float32(0.0))
    var zhp = _AP(unsafe_from_address=Int(zh.unsafe_ptr()))
    var zlp = _AP(unsafe_from_address=Int(zl.unsafe_ptr()))
    var whp = _AP(unsafe_from_address=Int(wh.unsafe_ptr()))
    var wlp = _AP(unsafe_from_address=Int(wl.unsafe_ptr()))
    for j in range(d):
        for i in range(j, d):
            gff_forward_cell(ahp, alp, bhp, blp, zhp, zlp, i, j, d)
    for jj in range(d):
        var j = d - 1 - jj
        for i in range(j + 1):
            gff_backward_cell(ahp, alp, zhp, zlp, whp, wlp, i, j, d)
    for i in range(d):
        coef_ptr.unsafe_store(i, gff_coef(whp, wlp, svp, i))
    return 0
