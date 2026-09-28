# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""FAST ON APPLE: THE LINEAR FITS PAST ONE BLOCK (lane/linear-apple3, 2026-09-28).

x_linear/device.mojo `fit_device` launches `fit_kernel` with grid_dim=1: a
fit is one program on ONE block of LINEAR_TPB = 256 threads, and its row
passes are one thread's chain per output over all n rows
(docs/lanes/progress/py-bugs.md, item 5). A block cannot wait on another
block inside a launch, so a fit that uses the whole GPU cannot be one
launch. Here (FAST, Apple, n >= XB_MIN_ROWS) the fit's control and its
small dense algebra (m x m) run on the host, and every pass over the rows is
ONE launch of n / XB_ROWS blocks:

  * block k owns rows [k * XB_ROWS, (k + 1) * XB_ROWS);
  * its threads first map the block's rows (the linear predictor and the
    per-row terms, into device row buffers), a device-memory barrier, then
    one thread per output folds the block's rows ascending into
    part[k * cells + c];
  * the host sums the n / XB_ROWS partials of each output in float64, block
    ascending, and rounds once to float32.

The per-row arithmetic is the fit's own (x_linear/glm.mojo `_unit`, the
Huber, logistic and pinball expressions), the algorithms are the fits' own
(Newton-Cholesky, L-BFGS, ADMM, the same constants and stopping rules); only
the grouping of each sum over the rows differs, so FAST words change. The
partition depends on n alone, never on the machine: every Apple GPU gives
the same words. IDENTICAL never reaches this file (the hook in
x_linear/device.mojo is comptime-gated), so its bits cannot move.
`-D MOJOLEARN_X_LINEAR_BLOCKS_OFF=1` returns FAST on Apple to the one-block
team fit.
"""
from std.gpu import block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from x_linear.ops import (
    FP, IP, fa, fs, fm, fd, fmad, fexp, flog, fsqrt, fabs, fmax, ld, st, i2f,
    fill, copy, row_dot, cholesky, chol_solve, mean_of,
)
from x_linear.team import team_barrier, team_at
from x_linear.isotonic import isotonic_predict
from x_linear.tops import fold_fa, fold_sq, chain_fmad, chain_fmad_scaled, fold_one_fmad
from x_linear.glm import _unit, GLM_LINK_LOG
from x_linear.lbfgs import LBFGS_M, lbfgs_work, _dot
from x_linear.logcv import _predict_code
from x_linear.quantile import _soft
from checks.numerics import identical_sigmoid, identical_softplus, ftz
from std.sys.compile import is_defined

#: WIP, opt-in `-D MOJOLEARN_X_LINEAR_QUANTILE_STEP=1`: Quantile's ADMM
#: iteration without a read back. Job 1 measured the host-stepped fit bound
#: by one read back and synchronize per iteration (1.34 s at 20k rows and
#: at 100k). Here the iteration's m x m algebra runs in a one-block launch
#: on device state, the host queues XQ_BATCH iterations (two launches each)
#: and reads the state once per batch; a launch after the stop is a no-op.
comptime XB_QUANTILE_STEP = is_defined["MOJOLEARN_X_LINEAR_QUANTILE_STEP"]()
comptime XQ_BATCH = 32
#: scalars of the device state, after M m*m | beta m | rhs m | z d | v d
comptime XQ_RHO = 0
comptime XQ_Q = 1
comptime XQ_ALPHA = 2
comptime XQ_EPS_ABS = 3
comptime XQ_EPS_REL = 4
comptime XQ_YNORM = 5
comptime XQ_DEN = 6
comptime XQ_KQ = 7
comptime XQ_INV = 8
comptime XQ_IT = 9
comptime XQ_DONE = 10
comptime XQ_ITERS = 11
comptime XQ_MAX_ITER = 12
comptime XQ_SCALARS = 16

#: Threads per block (the M2 Pro drops a dispatch above its pipeline limit
#: with no error; 256 is LINEAR_TPB, which every Apple GPU runs).
comptime XB_TPB = 256
#: Rows per block. SCHEDULING AND GROUPING: it fixes where each sum over the
#: rows is cut, so it is one constant for every machine.
comptime XB_ROWS = 1024
#: Below this many rows the one-block team fit is kept (a pass here costs a
#: launch, a read back and a synchronize whatever n is).
comptime XB_MIN_ROWS = 8192

comptime XB_GLM = 2
comptime XB_HUBER = 3
comptime XB_QUANTILE = 7
comptime XB_LOGCV = 10
comptime XB_ISOTONIC_PREDICT = 12


#: Opt-in `-D MOJOLEARN_X_LINEAR_BLOCKS_PLAIN=1`: the block folds as plain
#: loops instead of x_linear/tops.mojo's chains (which hold 32, 64 or 96
#: loaded words in registers). On a GPU without Dynamic Caching (the M2
#: Pro) a pipeline's thread limit falls with its register use, and a
#: dispatch above the limit is dropped with no error; the plain form is the
#: fallback if a block kernel's limit falls under XB_TPB there. The words
#: are the same either way (the same operations in the same order).
comptime XB_PLAIN = is_defined["MOJOLEARN_X_LINEAR_BLOCKS_PLAIN"]()


@always_inline
def xb_dot(a: FP, aoff: Int, astep: Int, b: FP, boff: Int, bstep: Int, n: Int) -> Float32:
    comptime if XB_PLAIN:
        var acc = Float32(0)
        for i in range(n):
            acc = fmad(ld(a, aoff + i * astep), ld(b, boff + i * bstep), acc)
        return acc
    return chain_fmad(a, aoff, astep, b, boff, bstep, n)


@always_inline
def xb_sum(v: FP, off: Int, n: Int) -> Float32:
    comptime if XB_PLAIN:
        var acc = Float32(0)
        for i in range(n):
            acc = fa(acc, ld(v, off + i))
        return acc
    return fold_fa(v, off, 1, n)


@always_inline
def xb_sumsq(v: FP, off: Int, n: Int) -> Float32:
    comptime if XB_PLAIN:
        var acc = Float32(0)
        for i in range(n):
            var r = ld(v, off + i)
            acc = fmad(r, r, acc)
        return acc
    return fold_sq(v, off, n)


@always_inline
def xb_one(v: FP, off: Int, n: Int) -> Float32:
    comptime if XB_PLAIN:
        var acc = Float32(0)
        for i in range(n):
            acc = fmad(Float32(1), ld(v, off + i), acc)
        return acc
    return fold_one_fmad(v, off, n)


@always_inline
def xb_dot_scaled(h: FP, x: FP, j: Int, k: Int, d: Int, n: Int) -> Float32:
    comptime if XB_PLAIN:
        var acc = Float32(0)
        for i in range(n):
            acc = fmad(fm(ld(h, i), ld(x, i * d + j)), ld(x, i * d + k), acc)
        return acc
    return chain_fmad_scaled(h, x, j, k, d, n)


def blocks_handles(algo: Int, n: Int) -> Bool:
    """The fits this file runs (the algo numbers of x_linear/dispatch.mojo)."""
    if n < XB_MIN_ROWS:
        return False
    return (algo == XB_GLM or algo == XB_HUBER or algo == XB_QUANTILE or algo == XB_LOGCV
            or algo == XB_ISOTONIC_PREDICT)


# ---------------------------------------------------------------- kernels

def xb_glm_kernel(
    x: FP, y: FP, th: FP, rw: FP, part: FP, n_in: Int32, d_in: Int32, flags_in: Int32, what_in: Int32,
):
    """th: theta d + 1 | power. flags: bit 0 fit_intercept, bit 1
    sample_weight, bits 8.. the link. what 0: the loss terms, one output
    (their sum); what 1: d/deta and d2/deta2, m gradient cells then the
    Hessian's lower triangle (x_linear/glm.mojo's cell order)."""
    var n = Int(n_in)
    var d = Int(d_in)
    var flags = Int(flags_in)
    var what = Int(what_in)
    var fi = (flags & 1) != 0
    var sw = (flags & 2) != 0
    var link = flags >> 8
    var m = d + 1 if fi else d
    var power = ld(th, d + 1)
    var b = ld(th, d) if fi else Float32(0)
    var tid = Int(thread_idx.x)
    var blk = Int(block_idx.x)
    var r0 = blk * XB_ROWS
    var r1 = r0 + XB_ROWS
    if r1 > n:
        r1 = n
    var rows = r1 - r0
    var i = r0 + tid
    while i < r1:
        var e = fa(row_dot(x, i, d, th, 0), b)
        var yi = ld(y, i)
        if what == 0:
            var l = _unit(power, link, yi, e, 0)
            if sw:
                l = fm(ld(y, n + i), l)
            st(rw, i, l)
        else:
            var gi = _unit(power, link, yi, e, 1)
            var hi = fmax(Float32(0), _unit(power, link, yi, e, 2))
            if sw:
                gi = fm(ld(y, n + i), gi)
                hi = fm(ld(y, n + i), hi)
            st(rw, i, gi)
            st(rw, n + i, hi)
        i += XB_TPB
    team_barrier()
    if what == 0:
        if tid == 0:
            st(part, blk, xb_sum(rw, r0, rows))
        return
    var cells = m + m * (m + 1) // 2
    var c = tid
    while c < cells:
        var acc: Float32
        if c < m:
            if c < d:
                acc = xb_dot(rw, r0, 1, x, r0 * d + c, d, rows)
            else:
                acc = xb_sum(rw, r0, rows)
        else:
            var q = c - m
            var j = 0
            while (j + 1) * (j + 2) // 2 <= q:
                j += 1
            var k = q - j * (j + 1) // 2
            if j < d:
                acc = xb_dot_scaled(rw + (n + r0), x + r0 * d, j, k, d, rows)
            elif k < d:
                acc = xb_dot(rw, n + r0, 1, x, r0 * d + k, d, rows)
            else:
                acc = xb_sum(rw, n + r0, rows)
        st(part, blk * cells + c, acc)
        c += XB_TPB


def xb_huber_kernel(
    x: FP, y: FP, th: FP, rw: FP, part: FP, n_in: Int32, d_in: Int32, flags_in: Int32,
):
    """th: theta (w d, b if fit_intercept, s) | at d + 2: eps * sigma,
    2 / sigma, 2 eps. Row buffers: 0 the gradient coefficients, 1 the
    residuals. Outputs: the gradient cells (d, the intercept's when fitted),
    then the inlier squares, the outlier |r| sum, the outlier count, the
    outlier weight (x_linear/huber.mojo's expressions)."""
    var n = Int(n_in)
    var d = Int(d_in)
    var flags = Int(flags_in)
    var fi = (flags & 1) != 0
    var sw = (flags & 2) != 0
    var b = ld(th, d) if fi else Float32(0)
    var thr = ld(th, d + 2)
    var two_over_sigma = ld(th, d + 3)
    var two_eps = ld(th, d + 4)
    var tid = Int(thread_idx.x)
    var blk = Int(block_idx.x)
    var r0 = blk * XB_ROWS
    var r1 = r0 + XB_ROWS
    if r1 > n:
        r1 = n
    var rows = r1 - r0
    var i = r0 + tid
    while i < r1:
        var r = fs(fs(ld(y, i), row_dot(x, i, d, th, 0)), b)
        var ar = fabs(r)
        var coefv: Float32
        if sw:
            var wi = ld(y, n + i)
            if ar > thr:
                coefv = fm(wi, -two_eps if r > 0 else two_eps)
            else:
                coefv = fm(-two_over_sigma, fm(wi, r))
        elif ar > thr:
            coefv = -two_eps if r > 0 else two_eps
        else:
            coefv = fm(-two_over_sigma, r)
        st(rw, i, coefv)
        st(rw, n + i, r)
        i += XB_TPB
    team_barrier()
    var gcells = d + 1 if fi else d
    var cells = gcells + 4
    var c = tid
    while c < cells:
        var acc = Float32(0)
        if c < d:
            acc = xb_dot(rw, r0, 1, x, r0 * d + c, d, rows)
        elif c < gcells:
            acc = xb_sum(rw, r0, rows)
        else:
            var role = c - gcells
            for q in range(r0, r1):
                var r = ld(rw, n + q)
                var ar = fabs(r)
                if role == 0:
                    if not (ar > thr):
                        if sw:
                            acc = fmad(fm(ld(y, n + q), r), r, acc)
                        else:
                            acc = fmad(r, r, acc)
                elif ar > thr:
                    if role == 1:
                        if sw:
                            acc = fmad(ld(y, n + q), ar, acc)
                        else:
                            acc = fa(acc, ar)
                    elif role == 2:
                        acc = fa(acc, Float32(1))
                    elif sw:
                        acc = fa(acc, ld(y, n + q))
        st(part, blk * cells + c, acc)
        c += XB_TPB


def xb_logistic_kernel(
    x: FP, y: FP, th: FP, rw: FP, part: FP, n_in: Int32, d_in: Int32, kp_in: Int32, fold_in: Int32,
    flags_in: Int32, what_in: Int32,
):
    """th: theta K' * (d + 1). y: labels | fold ids | fit weights | score
    weights. what 0: the objective's rows (a held-out row of `fold` stores
    zeros, so it adds nothing to any sum): row buffers 0..K'-1 the
    residuals, K' the loss terms; outputs the K' * (d + 1) gradient cells,
    then the loss sum. what 1: the held-out rows' hits (weighted by the
    score weight with sample_weight), one output (x_linear/logcv.mojo's
    expressions)."""
    var n = Int(n_in)
    var d = Int(d_in)
    var kp = Int(kp_in)
    var fold = Int(fold_in)
    var flags = Int(flags_in)
    var what = Int(what_in)
    var fi = (flags & 1) != 0
    var sw = (flags & 2) != 0
    var stride = d + 1
    var p = kp * stride
    var tid = Int(thread_idx.x)
    var blk = Int(block_idx.x)
    var r0 = blk * XB_ROWS
    var r1 = r0 + XB_ROWS
    if r1 > n:
        r1 = n
    var rows = r1 - r0
    var i = r0 + tid
    while i < r1:
        var held = fold >= 0 and Int(ld(y, n + i)) == fold
        var label = Int(ld(y, i))
        if what == 1:
            var v = Float32(0)
            if held:
                if _predict_code(x, i, d, kp, fi, th, 0) == label:
                    v = ld(y, 3 * n + i) if sw else Float32(1)
            st(rw, i, v)
        elif held:
            for k in range(kp):
                st(rw, k * n + i, Float32(0))
            st(rw, kp * n + i, Float32(0))
        else:
            var wi = Float32(1)
            if sw:
                wi = ld(y, 2 * n + i)
            if kp == 1:
                var z = fa(row_dot(x, i, d, th, 0), ld(th, d) if fi else Float32(0))
                var yi = Float32(1) if label == 1 else Float32(0)
                var li = fs(ftz(identical_softplus(z)), fm(yi, z))
                var r = fs(ftz(identical_sigmoid(z)), yi)
                if sw:
                    li = fm(wi, li)
                    r = fm(wi, r)
                st(rw, i, r)
                st(rw, n + i, li)
            else:
                var zmax = Float32(-3.0e38)
                for k in range(kp):
                    var z = fa(row_dot(x, i, d, th, k * stride), ld(th, k * stride + d) if fi else Float32(0))
                    zmax = fmax(zmax, z)
                var se = Float32(0)
                var zy = Float32(0)
                for k in range(kp):
                    var z = fa(row_dot(x, i, d, th, k * stride), ld(th, k * stride + d) if fi else Float32(0))
                    se = fa(se, fexp(fs(z, zmax)))
                    if k == label:
                        zy = z
                var lse = fa(zmax, flog(se))
                st(rw, kp * n + i, fm(wi, fs(lse, zy)) if sw else fs(lse, zy))
                for k in range(kp):
                    var z = fa(row_dot(x, i, d, th, k * stride), ld(th, k * stride + d) if fi else Float32(0))
                    var r = fexp(fs(z, lse))
                    if k == label:
                        r = fs(r, Float32(1))
                    if sw:
                        r = fm(wi, r)
                    st(rw, k * n + i, r)
        i += XB_TPB
    team_barrier()
    if what == 1:
        if tid == 0:
            st(part, blk, xb_sum(rw, r0, rows))
        return
    var cells = p + 1
    var c = tid
    while c < cells:
        var acc = Float32(0)
        if c < p:
            var k = c // stride
            var j = c - k * stride
            if j < d:
                acc = xb_dot(rw, k * n + r0, 1, x, r0 * d + j, d, rows)
            elif fi:
                acc = xb_sum(rw, k * n + r0, rows)
        else:
            acc = xb_sum(rw, kp * n + r0, rows)
        st(part, blk * cells + c, acc)
        c += XB_TPB


def xb_quantile_kernel(
    x: FP, y: FP, th: FP, rw: FP, part: FP, n_in: Int32, d_in: Int32, flags_in: Int32, what_in: Int32,
):
    """th: beta d + 1 | q | 1 / (den rho) | the u rescale. Row buffers:
    0 A beta, 1 the change of r, 2 the primal residuals, 3 y - r - u, 4 r,
    5 u. what 0: the lower triangle of A'A (A = [X, 1]). what 1: one ADMM
    row update; outputs |A beta|^2, |r|^2, the primal residuals' and |u|^2's
    row parts, the m sums of A' dr, the m sums of the next A'(y - r - u).
    what 2: u rescaled, then the m sums of A'(y - r - u)
    (x_linear/quantile.mojo's expressions)."""
    var n = Int(n_in)
    var d = Int(d_in)
    var flags = Int(flags_in)
    var what = Int(what_in)
    var fi = (flags & 1) != 0
    var sw = (flags & 2) != 0
    var m = d + 1 if fi else d
    var tid = Int(thread_idx.x)
    var blk = Int(block_idx.x)
    var r0 = blk * XB_ROWS
    var r1 = r0 + XB_ROWS
    if r1 > n:
        r1 = n
    var rows = r1 - r0
    if what == 0:
        var gcells = m * (m + 1) // 2
        var c = tid
        while c < gcells:
            var j = 0
            var qq = c
            while qq >= j + 1:
                qq -= j + 1
                j += 1
            var k = qq
            var acc = Float32(0)
            for i in range(r0, r1):
                var aj = ld(x, i * d + j) if j < d else Float32(1)
                var ak = ld(x, i * d + k) if k < d else Float32(1)
                acc = fmad(aj, ak, acc)
            st(part, blk * gcells + c, acc)
            c += XB_TPB
        return
    var b = ld(th, d) if fi else Float32(0)
    var q = ld(th, d + 1)
    var kq = ld(th, d + 2)
    var inv = ld(th, d + 3)
    var i = r0 + tid
    while i < r1:
        var yi = ld(y, i)
        if what == 1:
            var up = fm(q, kq)
            var lo = fm(fs(Float32(1), q), kq)
            var abi = fa(row_dot(x, i, d, th, 0), b)
            var ui = ld(rw, 5 * n + i)
            var vv = fs(fs(yi, abi), ui)
            if sw:
                var ki = fm(ld(y, n + i), kq)
                up = fm(q, ki)
                lo = fm(fs(Float32(1), q), ki)
            var nr: Float32
            if vv > up:
                nr = fs(vv, up)
            elif vv < -lo:
                nr = fa(vv, lo)
            else:
                nr = Float32(0)
            st(rw, n + i, fs(nr, ld(rw, 4 * n + i)))
            st(rw, 4 * n + i, nr)
            var pr = fs(fa(abi, nr), yi)
            var un = fa(ui, pr)
            st(rw, i, abi)
            st(rw, 2 * n + i, pr)
            st(rw, 5 * n + i, un)
            st(rw, 3 * n + i, fs(fs(yi, nr), un))
        else:
            var un = fm(ld(rw, 5 * n + i), inv)
            st(rw, 5 * n + i, un)
            st(rw, 3 * n + i, fs(fs(yi, ld(rw, 4 * n + i)), un))
        i += XB_TPB
    team_barrier()
    var lead = 0
    var cells = m
    if what == 1:
        lead = 4
        cells = 4 + 2 * m
    var c = tid
    while c < cells:
        var acc: Float32
        if c < lead:
            if c == 0:
                acc = xb_sumsq(rw, r0, rows)
            elif c == 1:
                acc = xb_sumsq(rw, 4 * n + r0, rows)
            elif c == 2:
                acc = xb_sumsq(rw, 2 * n + r0, rows)
            else:
                acc = xb_sumsq(rw, 5 * n + r0, rows)
        else:
            var o = c - lead
            var src = 3 * n
            if what == 1:
                src = n
            if o >= m:
                o -= m
                src = 3 * n
            if o < d:
                acc = xb_dot(x, r0 * d + o, d, rw, src + r0, 1, rows)
            else:
                acc = xb_one(rw, src + r0, rows)
        st(part, blk * cells + c, acc)
        c += XB_TPB


def xb_isotonic_predict_kernel(
    x: FP, y: FP, ip: IP, fp: FP, res: FP, n_in: Int32, nt_in: Int32,
):
    """x_linear/isotonic.mojo `isotonic_predict` on a team that spans the
    whole grid. Every query is its own output and the function has no
    barrier and no broadcast, so the words are the one-block schedule's
    (this schedule is bitwise identical; it is FAST and Apple only here
    because nothing else in this round may touch IDENTICAL)."""
    var tid = Int(block_idx.x) * XB_TPB + Int(thread_idx.x)
    var t = team_at(tid, Int(nt_in), res, Int(n_in), 3, 0)
    isotonic_predict(t, x, y, Int(n_in), 1, ip, fp, res, res, ip)


def xq_row_kernel(
    x: FP, y: FP, sd: FP, rw: FP, part: FP, n_in: Int32, d_in: Int32, flags_in: Int32,
):
    """`xb_quantile_kernel` what 1 on the device state sd (M | beta | rhs |
    z | v | scalars | totals): u is first rescaled by the previous
    iteration's factor. Outputs: the four norms, the m sums of A' dr, the m
    sums of A'(y - r - u), and on every tenth iteration the same m sums
    with u rescaled by 1/2 and by 2 (the residual balancing may pick
    either, and the step launch takes the one it picked)."""
    var n = Int(n_in)
    var d = Int(d_in)
    var flags = Int(flags_in)
    var fi = (flags & 1) != 0
    var sw = (flags & 2) != 0
    var m = d + 1 if fi else d
    var beta = m * m
    var sc = beta + 2 * m + 2 * d
    if ld(sd, sc + XQ_DONE) != 0:
        return
    var it = Int(ld(sd, sc + XQ_IT))
    var tenth = (it + 1) % 10 == 0
    var b = ld(sd, beta + d) if fi else Float32(0)
    var q = ld(sd, sc + XQ_Q)
    var kq = ld(sd, sc + XQ_KQ)
    var inv = ld(sd, sc + XQ_INV)
    var tid = Int(thread_idx.x)
    var blk = Int(block_idx.x)
    var r0 = blk * XB_ROWS
    var r1 = r0 + XB_ROWS
    if r1 > n:
        r1 = n
    var rows = r1 - r0
    var i = r0 + tid
    while i < r1:
        var yi = ld(y, i)
        var up = fm(q, kq)
        var lo = fm(fs(Float32(1), q), kq)
        var abi = fa(row_dot(x, i, d, sd, beta), b)
        var ui = fm(ld(rw, 5 * n + i), inv)
        var vv = fs(fs(yi, abi), ui)
        if sw:
            var ki = fm(ld(y, n + i), kq)
            up = fm(q, ki)
            lo = fm(fs(Float32(1), q), ki)
        var nr: Float32
        if vv > up:
            nr = fs(vv, up)
        elif vv < -lo:
            nr = fa(vv, lo)
        else:
            nr = Float32(0)
        st(rw, n + i, fs(nr, ld(rw, 4 * n + i)))
        st(rw, 4 * n + i, nr)
        var pr = fs(fa(abi, nr), yi)
        var un = fa(ui, pr)
        st(rw, i, abi)
        st(rw, 2 * n + i, pr)
        st(rw, 5 * n + i, un)
        st(rw, 3 * n + i, fs(fs(yi, nr), un))
        i += XB_TPB
    team_barrier()
    var cells = 4 + 4 * m
    var c = tid
    while c < cells:
        var acc = Float32(0)
        if c == 0:
            acc = xb_sumsq(rw, r0, rows)
        elif c == 1:
            acc = xb_sumsq(rw, 4 * n + r0, rows)
        elif c == 2:
            acc = xb_sumsq(rw, 2 * n + r0, rows)
        elif c == 3:
            acc = xb_sumsq(rw, 5 * n + r0, rows)
        elif c < 4 + 2 * m:
            var o = c - 4
            var src = n
            if o >= m:
                o -= m
                src = 3 * n
            if o < d:
                acc = xb_dot(x, r0 * d + o, d, rw, src + r0, 1, rows)
            else:
                acc = xb_one(rw, src + r0, rows)
        elif tenth:
            var o = c - 4 - 2 * m
            var f = Float32(0.5)
            if o >= m:
                o -= m
                f = Float32(2)
            for k in range(r0, r1):
                var tv = fs(fs(ld(y, k), ld(rw, 4 * n + k)), fm(ld(rw, 5 * n + k), f))
                var a = ld(x, k * d + o) if o < d else Float32(1)
                acc = fmad(a, tv, acc)
        st(part, blk * cells + c, acc)
        c += XB_TPB


def xq_step_kernel(
    sd: FP, part: FP, n_in: Int32, d_in: Int32, nb_in: Int32, flags_in: Int32,
):
    """One block. Thread c sums output c of `xq_row_kernel` over the blocks
    (32 partials at a time, then the groups); then thread 0 runs the rest
    of the ADMM iteration of x_linear/quantile.mojo (the z and v updates,
    the residuals, the stopping rule, the residual balancing) and the next
    iteration's beta, on the device state."""
    var n = Int(n_in)
    var d = Int(d_in)
    var nb = Int(nb_in)
    var flags = Int(flags_in)
    var fi = (flags & 1) != 0
    var m = d + 1 if fi else d
    var mm = 0
    var beta = m * m
    var rhs = beta + m
    var z = rhs + m
    var v = z + d
    var sc = v + d
    var tot = sc + XQ_SCALARS
    var cells = 4 + 4 * m
    var done = ld(sd, sc + XQ_DONE)
    var tid = Int(thread_idx.x)
    if done == 0:
        var c = tid
        while c < cells:
            var s = Float32(0)
            var k = 0
            while k < nb:
                var e = k + 32
                if e > nb:
                    e = nb
                var g = Float32(0)
                for kk in range(k, e):
                    g = fa(g, ld(part, kk * cells + c))
                s = fa(s, g)
                k = e
            st(sd, tot + c, s)
            c += XB_TPB
    team_barrier()
    if tid != 0 or done != 0:
        return
    var it = Int(ld(sd, sc + XQ_IT))
    var rho = ld(sd, sc + XQ_RHO)
    var alpha = ld(sd, sc + XQ_ALPHA)
    var eps_abs = ld(sd, sc + XQ_EPS_ABS)
    var eps_rel = ld(sd, sc + XQ_EPS_REL)
    var ynorm = ld(sd, sc + XQ_YNORM)
    var den = ld(sd, sc + XQ_DEN)
    var max_iter = Int(ld(sd, sc + XQ_MAX_ITER))
    var abn = ld(sd, tot)
    var rn = ld(sd, tot + 1)
    var prim = ld(sd, tot + 2)
    var un = ld(sd, tot + 3)
    # z-update
    var zdiff = Float32(0)
    var wn = Float32(0)
    var zn = Float32(0)
    var t = fd(alpha, rho)
    for j in range(d):
        var wj = ld(sd, beta + j)
        wn = fmad(wj, wj, wn)
        var nz = _soft(fa(wj, ld(sd, v + j)), t)
        var dz = fs(nz, ld(sd, z + j))
        zdiff = fmad(dz, dz, zdiff)
        st(sd, z + j, nz)
        zn = fmad(nz, nz, zn)
    for j in range(d):
        var pj = fs(ld(sd, beta + j), ld(sd, z + j))
        prim = fmad(pj, pj, prim)
        st(sd, v + j, fa(ld(sd, v + j), pj))
    var dual = zdiff
    for j in range(m):
        var acc = ld(sd, tot + 4 + j)
        dual = fmad(acc, acc, dual)
    var prim_n = fsqrt(prim)
    var dual_n = fm(rho, fsqrt(dual))
    var scale_p = fmax(fmax(fsqrt(abn), fsqrt(rn)), fmax(ynorm, fmax(fsqrt(wn), fsqrt(zn))))
    var eps_p = fa(fm(eps_abs, fsqrt(i2f(n + d))), fm(eps_rel, scale_p))
    for j in range(d):
        un = fmad(ld(sd, v + j), ld(sd, v + j), un)
    var eps_d = fa(fm(eps_abs, fsqrt(i2f(m))), fm(fm(eps_rel, rho), fsqrt(un)))
    st(sd, sc + XQ_ITERS, i2f(it + 1))
    if prim_n <= eps_p and dual_n <= eps_d:
        st(sd, sc + XQ_DONE, Float32(1))
        return
    var pick = 0
    var inv = Float32(1)
    if (it + 1) % 10 == 0:
        var factor = Float32(0)
        if prim_n > fm(Float32(10), dual_n):
            factor = Float32(2)
            pick = 1
        elif dual_n > fm(Float32(10), prim_n):
            factor = Float32(0.5)
            pick = 2
        if factor != 0:
            rho = fm(rho, factor)
            inv = fd(Float32(1), factor)
            for j in range(d):
                st(sd, v + j, fm(ld(sd, v + j), inv))
    if it + 1 >= max_iter:
        st(sd, sc + XQ_DONE, Float32(2))
        return
    # the next iteration's beta
    for j in range(m):
        var a = ld(sd, tot + 4 + m + pick * m + j)
        if j < d:
            a = fa(a, fs(ld(sd, z + j), ld(sd, v + j)))
        st(sd, rhs + j, a)
    chol_solve(sd, mm, m, sd, rhs)
    for j in range(m):
        st(sd, beta + j, ld(sd, rhs + j))
    st(sd, sc + XQ_RHO, rho)
    st(sd, sc + XQ_KQ, fd(Float32(1), fm(den, rho)))
    st(sd, sc + XQ_INV, inv)
    st(sd, sc + XQ_IT, i2f(it + 1))


# ------------------------------------------------------------ host state

struct Part(Movable):
    """One pass's outputs: part[k * cells + c] on the device and at home."""

    var dev: DeviceBuffer[DType.float32]
    var home: HostBuffer[DType.float32]
    var nb: Int
    var cells: Int

    def __init__(out self, ctx: DeviceContext, nb: Int, cells: Int) raises:
        self.dev = ctx.enqueue_create_buffer[DType.float32](nb * cells)
        self.home = ctx.enqueue_create_host_buffer[DType.float32](nb * cells)
        self.nb = nb
        self.cells = cells

    def fetch(mut self, ctx: DeviceContext) raises:
        """The launch's partials, read back behind ONE synchronize."""
        ctx.enqueue_copy(dst_buf=self.home, src_buf=self.dev)
        ctx.synchronize()

    def total64(mut self, c: Int) -> Float64:
        """Output c: its block partials summed in float64, block ascending."""
        var acc = Float64(0)
        var hp = self.home.unsafe_ptr()
        for k in range(self.nb):
            acc += Float64(hp.unsafe_load(k * self.cells + c))
        return acc

    def total(mut self, c: Int) -> Float32:
        return Float32(self.total64(c))


struct XB(Movable):
    """A fit's device buffers (fields, so they outlive every launch that
    takes their pointers) and the objective's parameters."""

    var dx: DeviceBuffer[DType.float32]
    var dy: DeviceBuffer[DType.float32]
    var dth: DeviceBuffer[DType.float32]
    var hth: HostBuffer[DType.float32]
    var drw: DeviceBuffer[DType.float32]
    var n: Int
    var d: Int
    var nb: Int
    var which: Int
    var fi: Bool
    var sw: Bool
    var kp: Int
    var fold: Int
    var c: Float32
    var cntf: Float32
    var eps: Float32
    var alpha: Float32
    var w_all: Float32

    def __init__(
        out self, ctx: DeviceContext, x: FP, n_x: Int, y: FP, n_y: Int, n: Int, d: Int,
        n_th: Int, row_bufs: Int,
    ) raises:
        self.dx = ctx.enqueue_create_buffer[DType.float32](max(n_x, 1))
        self.dy = ctx.enqueue_create_buffer[DType.float32](max(n_y, 1))
        self.dth = ctx.enqueue_create_buffer[DType.float32](n_th)
        self.hth = ctx.enqueue_create_host_buffer[DType.float32](n_th)
        self.drw = ctx.enqueue_create_buffer[DType.float32](row_bufs * n)
        self.n = n
        self.d = d
        self.nb = (n + XB_ROWS - 1) // XB_ROWS
        self.which = 0
        self.fi = False
        self.sw = False
        self.kp = 1
        self.fold = -1
        self.c = Float32(1)
        self.cntf = Float32(1)
        self.eps = Float32(0)
        self.alpha = Float32(0)
        self.w_all = Float32(0)
        if n_x > 0:
            ctx.enqueue_copy(dst_buf=self.dx, src_ptr=x)
        if n_y > 0:
            ctx.enqueue_copy(dst_buf=self.dy, src_ptr=y)
        self.drw.enqueue_fill(Float32(0))
        var hp = self.hth.unsafe_ptr()
        for j in range(n_th):
            hp.unsafe_store(j, Float32(0))

    def flags(self) -> Int:
        var f = 0
        if self.fi:
            f |= 1
        if self.sw:
            f |= 2
        return f

    def upload(mut self, ctx: DeviceContext, th: FP, toff: Int, count: Int) raises:
        """theta (count words) and whatever the caller stored behind it."""
        var hp = self.hth.unsafe_ptr()
        for j in range(count):
            hp.unsafe_store(j, ld(th, toff + j))
        ctx.enqueue_copy(dst_buf=self.dth, src_buf=self.hth)

    def param(mut self, k: Int, v: Float32):
        self.hth.unsafe_ptr().unsafe_store(k, v)

    def push(mut self, ctx: DeviceContext) raises:
        """The parameters stored with `param`, to the device."""
        ctx.enqueue_copy(dst_buf=self.dth, src_buf=self.hth)


def _host_fp(mut w: List[Float32]) -> FP:
    return FP(unsafe_from_address=Int(w.unsafe_ptr()))


def _zeros(count: Int) -> List[Float32]:
    var w = List[Float32](capacity=max(count, 1))
    for _ in range(max(count, 1)):
        w.append(Float32(0))
    return w^


# ------------------------------------------------------------------- GLM

def _glm_objective(
    mut b: XB, mut pa: Part, ctx: DeviceContext, th: FP, toff: Int, m: Int, power: Float32,
    link: Int, alpha: Float32, den: Float32,
) raises -> Float64:
    """The objective in float64: the block partials of the loss terms summed
    in float64 and NOT rounded back, so the line search can see a decrease
    below float32's resolution of the objective."""
    b.param(b.d + 1, power)
    b.upload(ctx, th, toff, m)
    ctx.enqueue_function[xb_glm_kernel](
        b.dx.unsafe_ptr(), b.dy.unsafe_ptr(), b.dth.unsafe_ptr(), b.drw.unsafe_ptr(), pa.dev.unsafe_ptr(),
        Int32(b.n), Int32(b.d), Int32(b.flags() | (link << 8)), Int32(0),
        grid_dim=b.nb, block_dim=XB_TPB,
    )
    pa.fetch(ctx)
    var acc = pa.total64(0)
    var reg = Float64(0)
    for j in range(b.d):
        var w = Float64(ld(th, toff + j))
        reg += w * w
    return acc / Float64(den) + 0.5 * Float64(alpha) * reg


def glm_fit_blocks(
    ctx: DeviceContext, x: FP, n_x: Int, y: FP, n_y: Int, n: Int, d: Int,
    ip: List[Int32], fp: List[Float32], res: FP,
) raises:
    """x_linear/glm.mojo `glm_fit` (Newton-Cholesky, the Armijo halving),
    its row passes in blocks. res: coef d, intercept, n_iter, converged.

    Two differences from the one-block fit, both toward the minimizer:
    the Armijo test compares float64 objectives (the one-block fit's
    float32 objective cannot show a decrease under 6e-8 of its value, and
    its fit then stops or wanders on rounding), and when the gradient meets
    `tol` ONE more Newton step is taken if it lowers the objective (Newton
    converges quadratically, so that step lands on the minimizer at
    float32's resolution of the coefficients)."""
    var max_iter = Int(ip[0])
    var fi = Int(ip[1]) != 0
    var link = Int(ip[2])
    var sw = Int(ip[3]) != 0
    var power = fp[0]
    var alpha = fp[1]
    var tol = fp[2]
    var m = d + 1 if fi else d
    var cells = m + m * (m + 1) // 2
    var b = XB(ctx, x, n_x, y, n_y, n, d, d + 2, 2)
    b.fi = fi
    b.sw = sw
    var pa = Part(ctx, b.nb, 1)
    var pb = Part(ctx, b.nb, cells)
    var den = i2f(n)
    if sw:
        den = Float32(0)
        for i in range(n):
            den = fa(den, ld(y, n + i))
    var hw = _zeros(3 * m + m * m + 1)
    var fw = _host_fp(hw)
    var g = 0
    var h = g + m
    var step = h + m * m
    var trial = step + m
    fill(res, 0, d + 3, Float32(0))
    if fi:
        var ym = mean_of(y, n)
        if sw:
            var acc = Float32(0)
            for i in range(n):
                acc = fmad(ld(y, n + i), ld(y, i), acc)
            ym = fd(acc, den)
        st(res, d, flog(ym) if link == GLM_LINK_LOG else ym)
    var iters = 0
    var converged = False
    var f = _glm_objective(b, pa, ctx, res, 0, m, power, link, alpha, den)
    for it in range(max_iter):
        b.param(d + 1, power)
        b.upload(ctx, res, 0, m)
        ctx.enqueue_function[xb_glm_kernel](
            b.dx.unsafe_ptr(), b.dy.unsafe_ptr(), b.dth.unsafe_ptr(), b.drw.unsafe_ptr(), pb.dev.unsafe_ptr(),
            Int32(n), Int32(d), Int32(b.flags() | (link << 8)), Int32(1),
            grid_dim=b.nb, block_dim=XB_TPB,
        )
        pb.fetch(ctx)
        for c in range(m):
            st(fw, g + c, pb.total(c))
        for j in range(m):
            for k in range(j + 1):
                st(fw, h + j * m + k, pb.total(m + j * (j + 1) // 2 + k))
        var slope = Float32(0)
        var inv_n = fd(Float32(1), den)
        var gmax = Float32(0)
        for j in range(m):
            var gj = fm(ld(fw, g + j), inv_n)
            if j < d:
                gj = fmad(alpha, ld(res, j), gj)
            st(fw, g + j, gj)
            gmax = fmax(gmax, fabs(gj))
        # the tolerance is met: one more step, then stop
        var met = gmax <= tol
        for j in range(m):
            for k in range(j + 1):
                var v = fm(ld(fw, h + j * m + k), inv_n)
                if j == k and j < d:
                    v = fa(v, alpha)
                st(fw, h + j * m + k, v)
                st(fw, h + k * m + j, v)
        for j in range(m):
            st(fw, step + j, -ld(fw, g + j))
        var ok = cholesky(fw, h, m)
        if ok:
            chol_solve(fw, h, m, fw, step)
        for j in range(m):
            slope = fmad(ld(fw, g + j), ld(fw, step + j), slope)
        if not (slope < 0):
            converged = met
            if not met:
                iters = it + 1
            break
        var tt = Float32(1)
        var accepted = False
        for _ in range(40):
            for j in range(m):
                st(fw, trial + j, fmad(tt, ld(fw, step + j), ld(res, j)))
            var ft = _glm_objective(b, pa, ctx, fw, trial, m, power, link, alpha, den)
            if ft == ft and ft <= f + 1.0e-4 * Float64(tt) * Float64(slope):
                copy(res, 0, fw, trial, m)
                f = ft
                accepted = True
                break
            tt = fm(tt, Float32(0.5))
        if met:
            converged = True
            if accepted:
                iters = it + 1
            break
        iters = it + 1
        if not accepted:
            break
    if not fi:
        st(res, d, Float32(0))
    st(res, d + 1, i2f(iters))
    st(res, d + 2, Float32(1) if converged else Float32(0))
    ctx.synchronize()
    _ = hw^
    _ = pa^
    _ = pb^
    _ = b^


# ----------------------------------------------- L-BFGS fits (Huber, LogCV)

def _huber_objective(
    mut b: XB, mut pa: Part, ctx: DeviceContext, th: FP, toff: Int, g: FP, goff: Int,
) raises -> Float32:
    """x_linear/huber.mojo `_huber_objective_host`, its row pass in blocks."""
    var d = b.d
    var p = d + 2 if b.fi else d + 1
    var sigma = fexp(ld(th, toff + p - 1))
    var thr = fm(b.eps, sigma)
    var two_over_sigma = fd(Float32(2), sigma)
    var two_eps = fm(Float32(2), b.eps)
    b.param(d + 2, thr)
    b.param(d + 3, two_over_sigma)
    b.param(d + 4, two_eps)
    b.upload(ctx, th, toff, d + 1 if b.fi else d)
    ctx.enqueue_function[xb_huber_kernel](
        b.dx.unsafe_ptr(), b.dy.unsafe_ptr(), b.dth.unsafe_ptr(), b.drw.unsafe_ptr(), pa.dev.unsafe_ptr(),
        Int32(b.n), Int32(d), Int32(b.flags()),
        grid_dim=b.nb, block_dim=XB_TPB,
    )
    pa.fetch(ctx)
    var gcells = d + 1 if b.fi else d
    fill(g, goff, p, Float32(0))
    for c in range(gcells):
        st(g, goff + c, pa.total(c))
    var sq = pa.total(gcells)
    var out_abs = pa.total(gcells + 1)
    var n_out = pa.total(gcells + 2)
    var w_out = pa.total(gcells + 3)
    var wn = Float32(0)
    for j in range(d):
        var w = ld(th, toff + j)
        wn = fmad(w, w, wn)
        st(g, goff + j, fmad(fm(Float32(2), b.alpha), w, ld(g, goff + j)))
    var squared_loss = fd(sq, sigma)
    var eps2 = fm(b.eps, b.eps)
    var cnt_out = w_out if b.sw else n_out
    var cnt = b.w_all if b.sw else i2f(b.n)
    var outlier_loss = fs(fm(two_eps, out_abs), fm(fm(sigma, cnt_out), eps2))
    var gsigma = fs(fs(cnt, fm(cnt_out, eps2)), fd(squared_loss, sigma))
    st(g, goff + p - 1, fm(gsigma, sigma))
    return fa(fa(fa(fm(cnt, sigma), squared_loss), outlier_loss), fm(b.alpha, wn))


def _logistic_objective(
    mut b: XB, mut pa: Part, ctx: DeviceContext, th: FP, toff: Int, g: FP, goff: Int,
) raises -> Float32:
    """x_linear/logcv.mojo `_logistic_objective_host`, its row pass in
    blocks (b.fold's held-out rows add nothing; b.cntf is the training rows'
    count, or their weight with sample_weight)."""
    var d = b.d
    var stride = d + 1
    var p = b.kp * stride
    b.upload(ctx, th, toff, p)
    ctx.enqueue_function[xb_logistic_kernel](
        b.dx.unsafe_ptr(), b.dy.unsafe_ptr(), b.dth.unsafe_ptr(), b.drw.unsafe_ptr(), pa.dev.unsafe_ptr(),
        Int32(b.n), Int32(d), Int32(b.kp), Int32(b.fold), Int32(b.flags()), Int32(0),
        grid_dim=b.nb, block_dim=XB_TPB,
    )
    pa.fetch(ctx)
    for o in range(p):
        st(g, goff + o, pa.total(o))
    var acc = pa.total(p)
    var inv_n = fd(Float32(1), b.cntf)
    var lam = fd(Float32(1), fm(b.c, b.cntf))
    var reg = Float32(0)
    for k in range(b.kp):
        for j in range(stride):
            var o = k * stride + j
            var gv = fm(ld(g, goff + o), inv_n)
            if j < d:
                var w = ld(th, toff + o)
                reg = fmad(w, w, reg)
                gv = fmad(lam, w, gv)
            st(g, goff + o, gv)
    return fa(fm(acc, inv_n), fm(fm(Float32(0.5), lam), reg))


def _objective(
    mut b: XB, mut pa: Part, ctx: DeviceContext, th: FP, toff: Int, g: FP, goff: Int,
) raises -> Float32:
    if b.which == XB_HUBER:
        return _huber_objective(b, pa, ctx, th, toff, g, goff)
    return _logistic_objective(b, pa, ctx, th, toff, g, goff)


def _lbfgs(
    mut b: XB, mut pa: Part, ctx: DeviceContext, theta: FP, toff: Int, p: Int, max_iter: Int,
    tol: Float32, fw: FP, woff: Int,
) raises -> Int:
    """x_linear/lbfgs.mojo `lbfgs`, statement for statement, on the host
    (a team of one), the objective `_objective`.
    Work layout at fw[woff:]: tn P | g P | gn P | dir P | S m*P | Y m*P | rho m | al m."""
    var tn = woff
    var g = tn + p
    var gn = g + p
    var dr = gn + p
    var sS = dr + p
    var sY = sS + LBFGS_M * p
    var rho = sY + LBFGS_M * p
    var al = rho + LBFGS_M
    var f = _objective(b, pa, ctx, theta, toff, fw, g)
    var count = 0
    var head = 0
    var it = 0
    while it < max_iter:
        var flag = 0
        var slope = Float32(0)
        var gmax = Float32(0)
        for j in range(p):
            gmax = fmax(gmax, fabs(ld(fw, g + j)))
        if gmax <= tol:
            flag = 1
        else:
            for j in range(p):
                st(fw, dr + j, ld(fw, g + j))
            for kk in range(count):
                var k = (head - 1 - kk + 2 * LBFGS_M) % LBFGS_M
                var a = fm(ld(fw, rho + k), _dot(fw, sS + k * p, fw, dr, p))
                st(fw, al + k, a)
                for j in range(p):
                    st(fw, dr + j, fs(ld(fw, dr + j), fm(a, ld(fw, sY + k * p + j))))
            var gamma: Float32
            if count > 0:
                var k = (head - 1 + LBFGS_M) % LBFGS_M
                gamma = fd(_dot(fw, sS + k * p, fw, sY + k * p, p), _dot(fw, sY + k * p, fw, sY + k * p, p))
            else:
                gamma = fd(Float32(1), fmax(Float32(1), fsqrt(_dot(fw, g, fw, g, p))))
            for j in range(p):
                st(fw, dr + j, fm(gamma, ld(fw, dr + j)))
            for kk in range(count):
                var k = (head - count + kk + 2 * LBFGS_M) % LBFGS_M
                var bb = fm(ld(fw, rho + k), _dot(fw, sY + k * p, fw, dr, p))
                var cc = fs(ld(fw, al + k), bb)
                for j in range(p):
                    st(fw, dr + j, fmad(cc, ld(fw, sS + k * p + j), ld(fw, dr + j)))
            for j in range(p):
                st(fw, dr + j, -ld(fw, dr + j))
            slope = _dot(fw, g, fw, dr, p)
            if not (slope < 0):
                count = 0
                head = 0
                for j in range(p):
                    st(fw, dr + j, -ld(fw, g + j))
                slope = _dot(fw, g, fw, dr, p)
                if not (slope < 0):
                    flag = 1
        if flag == 1:
            return it
        var tt = Float32(1)
        var accepted = False
        var fnew = f
        for _ in range(40):
            for j in range(p):
                st(fw, tn + j, fmad(tt, ld(fw, dr + j), ld(theta, toff + j)))
            fnew = _objective(b, pa, ctx, fw, tn, fw, gn)
            var ok = 0
            if fnew == fnew and fnew <= fa(f, fm(fm(Float32(1e-4), tt), slope)):
                ok = 1
            elif fnew == fnew and fnew <= fa(f, fa(fm(Float32(1e-6), fabs(f)), Float32(1e-30))):
                var dg = _dot(fw, gn, fw, dr, p)
                if fabs(dg) <= fm(Float32(0.9), fabs(slope)):
                    ok = 1
            if ok == 1:
                accepted = True
                break
            tt = fm(tt, Float32(0.5))
        if not accepted:
            return it
        it += 1
        var k = head
        for j in range(p):
            st(fw, sS + k * p + j, fs(ld(fw, tn + j), ld(theta, toff + j)))
            st(fw, sY + k * p + j, fs(ld(fw, gn + j), ld(fw, g + j)))
        var sy = _dot(fw, sS + k * p, fw, sY + k * p, p)
        var ss = _dot(fw, sS + k * p, fw, sS + k * p, p)
        var yy = _dot(fw, sY + k * p, fw, sY + k * p, p)
        if sy > fm(Float32(1e-10), fsqrt(fm(ss, yy))) and sy > 0:
            st(fw, rho + k, fd(Float32(1), sy))
            head = (head + 1) % LBFGS_M
            if count < LBFGS_M:
                count += 1
        copy(theta, toff, fw, tn, p)
        copy(fw, g, fw, gn, p)
        f = fnew
    return -it


def huber_fit_blocks(
    ctx: DeviceContext, x: FP, n_x: Int, y: FP, n_y: Int, n: Int, d: Int,
    ip: List[Int32], fp: List[Float32], res: FP,
) raises:
    """x_linear/huber.mojo `huber_fit`. res: coef d, intercept, scale,
    n_iter | theta scratch (P) at d + 4."""
    var fi = Int(ip[1]) != 0
    var sw = Int(ip[2]) != 0
    var p = d + 2 if fi else d + 1
    var th = d + 4
    var gcells = d + 1 if fi else d
    var b = XB(ctx, x, n_x, y, n_y, n, d, d + 5, 2)
    b.which = XB_HUBER
    b.fi = fi
    b.sw = sw
    b.eps = fp[0]
    b.alpha = fp[1]
    if sw:
        var w_all = Float32(0)
        for i in range(n):
            w_all = fa(w_all, ld(y, n + i))
        b.w_all = w_all
    var pa = Part(ctx, b.nb, gcells + 4)
    var hw = _zeros(lbfgs_work(p))
    var fw = _host_fp(hw)
    fill(res, th, p, Float32(0))
    var it = _lbfgs(b, pa, ctx, res, th, p, Int(ip[0]), fp[2], fw, 0)
    for j in range(d):
        st(res, j, ld(res, th + j))
    st(res, d, ld(res, th + d) if fi else Float32(0))
    st(res, d + 1, fexp(ld(res, th + p - 1)))
    st(res, d + 2, i2f(it if it >= 0 else -it))
    ctx.synchronize()
    _ = hw^
    _ = pa^
    _ = b^


def logcv_fit_blocks(
    ctx: DeviceContext, x: FP, n_x: Int, y: FP, n_y: Int, n: Int, d: Int,
    ip: List[Int32], fp: List[Float32], res: FP,
) raises:
    """x_linear/logcv.mojo `logcv_fit`: the Cs path with warm starts on each
    fold, the held-out accuracy, the first best mean, the refit from zero.
    res: coef K'*d | intercept K' | C_ | n_iter | scores F*nC."""
    var max_iter = Int(ip[0])
    var fi = Int(ip[1]) != 0
    var kp = Int(ip[2])
    var nc = Int(ip[3])
    var nf = Int(ip[4])
    var sw = Int(ip[5]) != 0
    var tol = fp[0]
    var stride = d + 1
    var p = kp * stride
    var sc = kp * d + kp + 2
    var b = XB(ctx, x, n_x, y, n_y, n, d, p, kp + 1)
    b.which = XB_LOGCV
    b.fi = fi
    b.sw = sw
    b.kp = kp
    var pa = Part(ctx, b.nb, p + 1)
    var ph = Part(ctx, b.nb, 1)
    var hw = _zeros(p + lbfgs_work(p))
    var fw = _host_fp(hw)
    var th = 0
    var work = p
    for f in range(nf):
        var cnt = 0
        var wrows = Float32(0)
        var wt = Float32(0)
        for i in range(n):
            if Int(ld(y, n + i)) != f:
                cnt += 1
                if sw:
                    wrows = fa(wrows, ld(y, 2 * n + i))
            elif sw:
                wt = fa(wt, ld(y, 3 * n + i))
        b.fold = f
        b.cntf = wrows if sw else i2f(cnt)
        fill(fw, th, p, Float32(0))
        for ci in range(nc):
            b.c = fp[1 + ci]
            _ = _lbfgs(b, pa, ctx, fw, th, p, max_iter, tol, fw, work)
            b.upload(ctx, fw, th, p)
            ctx.enqueue_function[xb_logistic_kernel](
                b.dx.unsafe_ptr(), b.dy.unsafe_ptr(), b.dth.unsafe_ptr(), b.drw.unsafe_ptr(),
                ph.dev.unsafe_ptr(),
                Int32(n), Int32(d), Int32(kp), Int32(f), Int32(b.flags()), Int32(1),
                grid_dim=b.nb, block_dim=XB_TPB,
            )
            ph.fetch(ctx)
            if sw:
                st(res, sc + f * nc + ci, fd(ph.total(0), wt) if wt > 0 else Float32(0))
            else:
                var held = n - cnt
                var hit = Int(ph.total64(0))
                st(res, sc + f * nc + ci, fd(i2f(hit), i2f(held)) if held > 0 else Float32(0))
    var best = 0
    var bs = Float32(0)
    for ci in range(nc):
        var acc = Float32(0)
        for f in range(nf):
            acc = fa(acc, ld(res, sc + f * nc + ci))
        var mean = fd(acc, i2f(nf))
        if ci == 0 or mean > bs:  # DEVIATION 5005: the first best C
            best = ci
            bs = mean
    b.fold = -1
    b.c = fp[1 + best]
    b.cntf = i2f(n)
    if sw:
        var wrows = Float32(0)
        for i in range(n):
            wrows = fa(wrows, ld(y, 2 * n + i))
        b.cntf = wrows
    fill(fw, th, p, Float32(0))
    var it = _lbfgs(b, pa, ctx, fw, th, p, max_iter, tol, fw, work)
    for k in range(kp):
        for j in range(d):
            st(res, k * d + j, ld(fw, th + k * stride + j))
        st(res, kp * d + k, ld(fw, th + k * stride + d) if fi else Float32(0))
    st(res, kp * d + kp, fp[1 + best])
    st(res, kp * d + kp + 1, i2f(it if it >= 0 else -it))
    ctx.synchronize()
    _ = hw^
    _ = pa^
    _ = ph^
    _ = b^


# -------------------------------------------------------------- Quantile

def quantile_fit_blocks(
    ctx: DeviceContext, x: FP, n_x: Int, y: FP, n_y: Int, n: Int, d: Int,
    ip: List[Int32], fp: List[Float32], res: FP,
) raises:
    """x_linear/quantile.mojo `_quantile_fit_host` (the scaled-form ADMM),
    its row passes in blocks. res: coef d, intercept, n_iter, converged.
    Host work: M m*m | beta m | rhs m | z d | v d | next rhs m."""
    var max_iter = Int(ip[0])
    var fi = Int(ip[1]) != 0
    var sw = Int(ip[2]) != 0
    var q = fp[0]
    var alpha = fp[1]
    var eps_abs = fp[2]
    var eps_rel = fp[3]
    var m = d + 1 if fi else d
    var gcells = m * (m + 1) // 2
    var b = XB(ctx, x, n_x, y, n_y, n, d, d + 4, 6)
    b.fi = fi
    b.sw = sw
    var pg = Part(ctx, b.nb, gcells)
    var pi = Part(ctx, b.nb, 4 + 2 * m)
    var pr = Part(ctx, b.nb, m)
    var den = i2f(n)
    if sw:
        den = Float32(0)
        for i in range(n):
            den = fa(den, ld(y, n + i))
    var hw = _zeros(m * m + 3 * m + 2 * d + 1)
    var fw = _host_fp(hw)
    var mm = 0
    var beta = m * m
    var rhs = beta + m
    var z = rhs + m
    var v = z + d
    var nrhs = v + d
    ctx.enqueue_function[xb_quantile_kernel](
        b.dx.unsafe_ptr(), b.dy.unsafe_ptr(), b.dth.unsafe_ptr(), b.drw.unsafe_ptr(), pg.dev.unsafe_ptr(),
        Int32(n), Int32(d), Int32(b.flags()), Int32(0),
        grid_dim=b.nb, block_dim=XB_TPB,
    )
    pg.fetch(ctx)
    for j in range(m):
        for k in range(j + 1):
            var acc = pg.total(j * (j + 1) // 2 + k)
            if j == k and j < d:
                acc = fa(acc, Float32(1))
            st(fw, mm + j * m + k, acc)
            st(fw, mm + k * m + j, acc)
    _ = cholesky(fw, mm, m)
    var ym = mean_of(y, n)
    var spread = Float32(0)
    for i in range(n):
        spread = fa(spread, fabs(fs(ld(y, i), ym)))
    spread = fmax(fd(spread, i2f(n)), Float32(1e-6))
    var rho = fd(Float32(1), fm(i2f(n), spread))
    var ynorm = Float32(0)
    for i in range(n):
        ynorm = fmad(ld(y, i), ld(y, i), ynorm)
    ynorm = fsqrt(ynorm)
    comptime if XB_QUANTILE_STEP:
        if 4 + 4 * m <= XB_TPB and max_iter > 0:
            # beta of iteration 0: rhs = A'y (r = u = 0), z = v = 0
            b.param(d + 3, Float32(1))
            b.push(ctx)
            ctx.enqueue_function[xb_quantile_kernel](
                b.dx.unsafe_ptr(), b.dy.unsafe_ptr(), b.dth.unsafe_ptr(), b.drw.unsafe_ptr(),
                pr.dev.unsafe_ptr(),
                Int32(n), Int32(d), Int32(b.flags()), Int32(2),
                grid_dim=b.nb, block_dim=XB_TPB,
            )
            pr.fetch(ctx)
            for j in range(m):
                st(fw, rhs + j, pr.total(j))
            chol_solve(fw, mm, m, fw, rhs)
            var sc = m * m + 2 * m + 2 * d
            var n_sd = sc + XQ_SCALARS + 4 + 4 * m
            var dsd = ctx.enqueue_create_buffer[DType.float32](n_sd)
            var hsd = ctx.enqueue_create_host_buffer[DType.float32](n_sd)
            var pq = Part(ctx, b.nb, 4 + 4 * m)
            var hp = hsd.unsafe_ptr()
            for j in range(n_sd):
                hp.unsafe_store(j, Float32(0))
            for j in range(m * m):
                hp.unsafe_store(j, ld(fw, mm + j))
            for j in range(m):
                hp.unsafe_store(m * m + j, ld(fw, rhs + j))
            hp.unsafe_store(sc + XQ_RHO, rho)
            hp.unsafe_store(sc + XQ_Q, q)
            hp.unsafe_store(sc + XQ_ALPHA, alpha)
            hp.unsafe_store(sc + XQ_EPS_ABS, eps_abs)
            hp.unsafe_store(sc + XQ_EPS_REL, eps_rel)
            hp.unsafe_store(sc + XQ_YNORM, ynorm)
            hp.unsafe_store(sc + XQ_DEN, den)
            hp.unsafe_store(sc + XQ_KQ, fd(Float32(1), fm(den, rho)))
            hp.unsafe_store(sc + XQ_INV, Float32(1))
            hp.unsafe_store(sc + XQ_MAX_ITER, i2f(max_iter))
            ctx.enqueue_copy(dst_buf=dsd, src_buf=hsd)
            var queued = 0
            while True:
                for _ in range(XQ_BATCH):
                    ctx.enqueue_function[xq_row_kernel](
                        b.dx.unsafe_ptr(), b.dy.unsafe_ptr(), dsd.unsafe_ptr(), b.drw.unsafe_ptr(),
                        pq.dev.unsafe_ptr(),
                        Int32(n), Int32(d), Int32(b.flags()),
                        grid_dim=b.nb, block_dim=XB_TPB,
                    )
                    ctx.enqueue_function[xq_step_kernel](
                        dsd.unsafe_ptr(), pq.dev.unsafe_ptr(),
                        Int32(n), Int32(d), Int32(b.nb), Int32(b.flags()),
                        grid_dim=1, block_dim=XB_TPB,
                    )
                queued += XQ_BATCH
                ctx.enqueue_copy(dst_buf=hsd, src_buf=dsd)
                ctx.synchronize()
                if hp.unsafe_load(sc + XQ_DONE) != 0 or queued >= max_iter + XQ_BATCH:
                    break
            var zo = m * m + 2 * m
            for j in range(d):
                st(res, j, hp.unsafe_load(zo + j))
            st(res, d, hp.unsafe_load(m * m + d) if fi else Float32(0))
            st(res, d + 1, hp.unsafe_load(sc + XQ_ITERS))
            st(res, d + 2, Float32(1) if hp.unsafe_load(sc + XQ_DONE) == Float32(1) else Float32(0))
            ctx.synchronize()
            _ = dsd^
            _ = hsd^
            _ = pq^
            _ = hw^
            _ = pg^
            _ = pi^
            _ = pr^
            _ = b^
            return
    var iters = 0
    var converged = False
    var rhs_ready = False
    var scale_u = Float32(1)
    for it in range(max_iter):
        iters = it + 1
        if rhs_ready:
            copy(fw, rhs, fw, nrhs, m)
        else:
            # u rescaled (by 1 at the start), then A'(y - r - u)
            b.param(d + 3, scale_u)
            b.upload(ctx, fw, beta, m)
            scale_u = Float32(1)
            ctx.enqueue_function[xb_quantile_kernel](
                b.dx.unsafe_ptr(), b.dy.unsafe_ptr(), b.dth.unsafe_ptr(), b.drw.unsafe_ptr(),
                pr.dev.unsafe_ptr(),
                Int32(n), Int32(d), Int32(b.flags()), Int32(2),
                grid_dim=b.nb, block_dim=XB_TPB,
            )
            pr.fetch(ctx)
            for j in range(m):
                st(fw, rhs + j, pr.total(j))
        for j in range(d):
            st(fw, rhs + j, fa(ld(fw, rhs + j), fs(ld(fw, z + j), ld(fw, v + j))))
        chol_solve(fw, mm, m, fw, rhs)
        copy(fw, beta, fw, rhs, m)
        var kq = fd(Float32(1), fm(den, rho))
        b.param(d + 1, q)
        b.param(d + 2, kq)
        b.upload(ctx, fw, beta, m)
        ctx.enqueue_function[xb_quantile_kernel](
            b.dx.unsafe_ptr(), b.dy.unsafe_ptr(), b.dth.unsafe_ptr(), b.drw.unsafe_ptr(), pi.dev.unsafe_ptr(),
            Int32(n), Int32(d), Int32(b.flags()), Int32(1),
            grid_dim=b.nb, block_dim=XB_TPB,
        )
        pi.fetch(ctx)
        var abn = pi.total(0)
        var rn = pi.total(1)
        var prim = pi.total(2)
        var un = pi.total(3)
        for j in range(m):
            st(fw, rhs + j, pi.total(4 + j))
            st(fw, nrhs + j, pi.total(4 + m + j))
        rhs_ready = True
        # z-update
        var zdiff = Float32(0)
        var wn = Float32(0)
        var zn = Float32(0)
        var t = fd(alpha, rho)
        for j in range(d):
            var wj = ld(fw, beta + j)
            wn = fmad(wj, wj, wn)
            var nz = _soft(fa(wj, ld(fw, v + j)), t)
            var dz = fs(nz, ld(fw, z + j))
            zdiff = fmad(dz, dz, zdiff)
            st(fw, z + j, nz)
            zn = fmad(nz, nz, zn)
        for j in range(d):
            var pj = fs(ld(fw, beta + j), ld(fw, z + j))
            prim = fmad(pj, pj, prim)
            st(fw, v + j, fa(ld(fw, v + j), pj))
        var dual = zdiff
        for j in range(m):
            var acc = ld(fw, rhs + j)
            dual = fmad(acc, acc, dual)
        var prim_n = fsqrt(prim)
        var dual_n = fm(rho, fsqrt(dual))
        var scale_p = fmax(fmax(fsqrt(abn), fsqrt(rn)), fmax(ynorm, fmax(fsqrt(wn), fsqrt(zn))))
        var eps_p = fa(fm(eps_abs, fsqrt(i2f(n + d))), fm(eps_rel, scale_p))
        for j in range(d):
            un = fmad(ld(fw, v + j), ld(fw, v + j), un)
        var eps_d = fa(fm(eps_abs, fsqrt(i2f(m))), fm(fm(eps_rel, rho), fsqrt(un)))
        if prim_n <= eps_p and dual_n <= eps_d:
            converged = True
            break
        if (it + 1) % 10 == 0:
            var factor = Float32(0)
            if prim_n > fm(Float32(10), dual_n):
                factor = Float32(2)
            elif dual_n > fm(Float32(10), prim_n):
                factor = Float32(0.5)
            if factor != 0:
                rho = fm(rho, factor)
                var inv = fd(Float32(1), factor)
                scale_u = inv
                rhs_ready = False
                for j in range(d):
                    st(fw, v + j, fm(ld(fw, v + j), inv))
    copy(res, 0, fw, z, d)
    st(res, d, ld(fw, beta + d) if fi else Float32(0))
    st(res, d + 1, i2f(iters))
    st(res, d + 2, Float32(1) if converged else Float32(0))
    ctx.synchronize()
    _ = hw^
    _ = pg^
    _ = pi^
    _ = pr^
    _ = b^


def isotonic_predict_blocks(
    ctx: DeviceContext, x: FP, n_x: Int, y: FP, n_y: Int, n: Int,
    ip: List[Int32], fp: List[Float32], n_out: Int, res: FP,
) raises:
    """ALGO_ISOTONIC_PREDICT: about four queries a thread over the grid."""
    var dx = ctx.enqueue_create_buffer[DType.float32](max(n_x, 1))
    var dy = ctx.enqueue_create_buffer[DType.float32](max(n_y, 1))
    var dip = ctx.enqueue_create_buffer[DType.int32](max(len(ip), 1))
    var dfp = ctx.enqueue_create_buffer[DType.float32](max(len(fp), 1))
    var dout = ctx.enqueue_create_buffer[DType.float32](max(n_out, 1))
    var hip = ip.copy()
    var hfp = fp.copy()
    ctx.enqueue_copy(dst_buf=dx, src_ptr=x)
    if n_y > 0:
        ctx.enqueue_copy(dst_buf=dy, src_ptr=y)
    if len(hip) > 0:
        ctx.enqueue_copy(dst_buf=dip, src_ptr=hip.unsafe_ptr())
    if len(hfp) > 0:
        ctx.enqueue_copy(dst_buf=dfp, src_ptr=hfp.unsafe_ptr())
    dout.enqueue_fill(Float32(0))
    var grid = (n + 4 * XB_TPB - 1) // (4 * XB_TPB)
    ctx.enqueue_function[xb_isotonic_predict_kernel](
        dx.unsafe_ptr(), dy.unsafe_ptr(), dip.unsafe_ptr(), dfp.unsafe_ptr(), dout.unsafe_ptr(),
        Int32(n), Int32(grid * XB_TPB),
        grid_dim=grid, block_dim=XB_TPB,
    )
    ctx.enqueue_copy(dst_ptr=res, src_buf=dout)
    ctx.synchronize()
    _ = hip^
    _ = hfp^
    _ = dx^
    _ = dy^
    _ = dip^
    _ = dfp^
    _ = dout^


# ------------------------------------------------------------------ entry

def blocks_fit(
    ctx: DeviceContext, algo: Int, x: FP, n_x: Int, y: FP, n_y: Int, n: Int, d: Int,
    ip: List[Int32], fp: List[Float32], n_out: Int, res: FP,
) raises:
    """The fit `algo` (one `blocks_handles` names), its result in res."""
    if algo == XB_ISOTONIC_PREDICT:
        isotonic_predict_blocks(ctx, x, n_x, y, n_y, n, ip, fp, n_out, res)
        return
    fill(res, 0, n_out, Float32(0))
    if algo == XB_GLM:
        glm_fit_blocks(ctx, x, n_x, y, n_y, n, d, ip, fp, res)
    elif algo == XB_HUBER:
        huber_fit_blocks(ctx, x, n_x, y, n_y, n, d, ip, fp, res)
    elif algo == XB_QUANTILE:
        quantile_fit_blocks(ctx, x, n_x, y, n_y, n, d, ip, fp, res)
    elif algo == XB_LOGCV:
        logcv_fit_blocks(ctx, x, n_x, y, n_y, n, d, ip, fp, res)
