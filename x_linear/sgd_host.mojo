# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""SGDClassifier's and SGDRegressor's fit on the host column (the CPU-only install and the host binding,
bindings/_mojolearn_x_linear_host.mojo, through x_linear/dispatch_host.mojo
`fit_dispatch`).

lane debt-host-call (2026-10-21): these functions moved here verbatim out of
x_linear/sgd.mojo, which the device binding imports, because their row passes
run on the host thread pool (`par_rows`, x_linear/ops.mojo) and no
GPU-reachable module calls host threads (the x_linear/isotonic_host.mojo
precedent). The device binding runs this fit on its grid driver
(x_linear/device.mojo `fit_device`, `_sgd_mb_grid` and `_sgd_ps_grid`); the statements here are unchanged,
so the host column's bits are unchanged.
"""
from x_linear.ops_host import par_rows
from x_linear.ops import FP, IP, fill, i2f, ld, ldi, shuffle, st
from x_linear.team import Team
from x_linear.sgd import (
    mb_sub_size,
    mb_subs,
    sgd_batch,
    sgd_mb_on,
    sgd_mb_one,
    sgd_one,
    sgd_perc_avg_from,
    sgd_perc_avg_on,
)


def sgd_fit(t: Team, x: FP, y: FP, n: Int, d: Int, ip: IP, fp: FP, res: FP, fw: FP, iw: IP):
    """ip: [n_classes, loss, penalty, lr, fit_intercept, max_iter, n_iter_no_change,
    shuffle, seed_lo, seed_hi, sample_weight, class_weight]; with
    sample_weight y = labels n | weights n; with class_weight fp carries
    [.., pos weight per problem (P), neg weight per problem (P)]; n_classes 0 = regression, 1 = one-class,
    2 = binary (positive class = label 1), K > 2 = one-vs-rest.
    fp: [alpha, l1_ratio, eta0, power_t, epsilon, tol].
    y: labels as 0..K-1 (classification) or targets. res: coef (P*d),
    intercept (P; for k == 1 the one-class offset_ = 1 - intercept,
    lane/neural-pass139), n_iter (1), status (1: 0 ok, -1 non-finite).
    The host binding's fit (cpu-gpu-cleanup w2-linear: the device's warp
    and thread forms, which ran a problem on one warp / one thread, are
    deleted; the device runs `_sgd_mb_grid` / `_sgd_ps_grid`).
    Host form (lane linear-cpu): fw: per problem n (targets) + d (q), then P
    (epochs run); iw: per problem n (order); the problems run as independent
    units (par_rows)."""
    var k = ldi(ip, 0)
    var loss = ldi(ip, 1)
    var penalty = ldi(ip, 2)
    var lr = ldi(ip, 3)
    var fit_intercept = ldi(ip, 4) != 0
    var max_iter = ldi(ip, 5)
    var nic = ldi(ip, 6)
    var do_shuffle = ldi(ip, 7) != 0
    var seed = (UInt64(UInt32(ip.unsafe_load(9))) << 32) | UInt64(UInt32(ip.unsafe_load(8)))
    var alpha = ld(fp, 0)
    var l1r = ld(fp, 1)
    var eta0 = ld(fp, 2)
    var power_t = ld(fp, 3)
    var eps = ld(fp, 4)
    var tol = ld(fp, 5)
    var has_sw = ldi(ip, 10) != 0
    var has_cw = ldi(ip, 11) != 0
    # ip[12] (lane/neural-pass103): the minibatch size, 0 the per-sample fit
    # (fix-l1-linear: negative is "auto", `sgd_batch`)
    var batch = sgd_batch(Int(ldi(ip, 12)), Int(k), n)
    # ip[13] (lane/neural-pass132): the per-sample-equivalent batch step (`mb_step`)
    var bsum = ldi(ip, 13) != 0
    var swp = y + n
    var problems = k if k > 2 else 1
    var epr = fw + problems * (n + d)  # the host's epochs run per problem
    # one-vs-rest problems are independent (their own targets, q, order and
    # result slots): the host may run them at once (lane linear-cpu)

    def run_problems(lo: Int, hi: Int) {imm x, imm y, imm n, imm d, imm k, imm fw, imm iw, imm res, imm problems,
                                        imm loss, imm penalty, imm alpha, imm l1r, imm lr, imm eta0, imm power_t,
                                        imm eps, imm fit_intercept, imm max_iter, imm tol, imm nic, imm do_shuffle,
                                        imm seed, imm swp, imm has_sw, imm has_cw, imm fp, imm epr, imm batch, imm bsum}:
        for c in range(lo, hi):
            var ys = fw + c * (n + d)
            var q = ys + n
            var idx = iw + c * n
            for i in range(n):
                var v = ld(y, i)
                if k == 0:
                    st(ys, i, v)
                elif k == 1:
                    st(ys, i, Float32(1))
                elif k == 2:
                    st(ys, i, Float32(1) if v == Float32(1) else Float32(-1))
                else:
                    st(ys, i, Float32(1) if v == i2f(c) else Float32(-1))
            var ep: Int
            if sgd_mb_on(batch, k, lr):
                # lane/neural-pass103: the minibatch form (x_linear/sgd_mb.mojo)
                var avg_from = sgd_perc_avg_from(max_iter) if sgd_perc_avg_on(loss, lr, k, max_iter) else -1
                var sl = List[Float32](length=2 * batch + (d + 2) * mb_subs(batch, mb_sub_size(batch)) + d + 1,
                                       fill=Float32(0))
                ep = sgd_mb_one(
                    x, ys, n, d, loss, penalty, alpha, l1r, lr, eta0, power_t, eps,
                    fit_intercept, max_iter, tol, nic, do_shuffle,
                    seed + UInt64(1000003) * UInt64(c), res, c * d, res, problems * d + c, idx,
                    swp, has_sw, ld(fp, 6 + c) if has_cw else Float32(1),
                    ld(fp, 6 + problems + c) if has_cw else Float32(1), has_cw, batch,
                    FP(unsafe_from_address=Int(sl.unsafe_ptr())), k == 1, bsum, avg_from,
                )
                _ = sl^
            else:
                ep = sgd_one(
                    x, ys, n, d, loss, penalty, alpha, l1r, lr, eta0, power_t, eps,
                    fit_intercept, max_iter, tol, nic, do_shuffle,
                    seed + UInt64(1000003) * UInt64(c), k == 1,
                    res, c * d, res, problems * d + c, q, idx,
                    swp, has_sw, ld(fp, 6 + c) if has_cw else Float32(1),
                    ld(fp, 6 + problems + c) if has_cw else Float32(1), has_cw,
                )
            st(epr, c, i2f(ep))

    par_rows(run_problems, problems, 1)
    if not t.lead():
        return
    var max_epochs = 0
    var status = 0
    for c in range(problems):
        var ep = Int(ld(epr, c))
        if ep < 0:
            status = -1
        elif ep > max_epochs:
            max_epochs = ep
    st(res, problems * d + problems, i2f(max_epochs))
    st(res, problems * d + problems + 1, i2f(status))
