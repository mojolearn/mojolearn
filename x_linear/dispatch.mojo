# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE LINEAR LANE'S ONE ENTRY POINT per call kind (lane/algos-linear).

`fit_dispatch` runs the fit named by `algo`; the host binding calls it on the
CPU and the GPU binding calls it from a one-thread kernel, so both columns run
this same source. `decision_one` is the shared scoring of one (row, output)
pair: link(b_c + sum_j x_ij w_cj), j ascending, the intercept added last.
"""
from x_linear.ops import FP, IP, fa, fmad, fexp, ld, st, row_dot
from checks.numerics import identical_sigmoid, ftz
from x_linear.team import Team, TEAM_ROW_BUFS
from x_linear.sgd import sgd_fit
from x_linear.glm import glm_fit
from x_linear.huber import huber_fit
from x_linear.bayes import bayes_ridge_fit, ard_fit
from x_linear.lars import lars_fit
from x_linear.quantile import quantile_fit
from x_linear.ridge import ridge_fit
from x_linear.cd import enetcv_fit
from x_linear.logcv import logcv_fit, logcv_team_rows
from x_linear.isotonic import isotonic_fit, isotonic_predict

comptime ALGO_SGD = 1
comptime ALGO_GLM = 2
comptime ALGO_HUBER = 3
comptime ALGO_BAYES = 4
comptime ALGO_ARD = 5
comptime ALGO_LARS = 6
comptime ALGO_QUANTILE = 7
comptime ALGO_RIDGE = 8
comptime ALGO_ENETCV = 9
comptime ALGO_LOGCV = 10
comptime ALGO_ISOTONIC = 11
comptime ALGO_ISOTONIC_PREDICT = 12

comptime LINK_IDENTITY = 0
comptime LINK_EXP = 1
comptime LINK_SIGMOID = 2


def team_fit(algo: Int) -> Bool:
    """The fits that run on a whole team (x_linear/team.mojo); the device
    runs every other fit on thread 0 alone."""
    return algo == ALGO_GLM or algo == ALGO_HUBER or algo == ALGO_LOGCV


def team_rows(algo: Int, ip: IP) -> Int:
    """Row buffers of n words the fit's team scratch holds."""
    if algo == ALGO_LOGCV:
        return logcv_team_rows(ip)
    return TEAM_ROW_BUFS


def fit_dispatch(t: Team, algo: Int, x: FP, y: FP, n: Int, d: Int, ip: IP, fp: FP, res: FP, fw: FP, iw: IP):
    """`t` is the team the fit runs on (a team of one for a fit that
    `team_fit` does not name)."""
    if algo == ALGO_SGD:
        sgd_fit(x, y, n, d, ip, fp, res, fw, iw)
    elif algo == ALGO_GLM:
        glm_fit(t, x, y, n, d, ip, fp, res, fw, iw)
    elif algo == ALGO_HUBER:
        huber_fit(t, x, y, n, d, ip, fp, res, fw, iw)
    elif algo == ALGO_BAYES:
        bayes_ridge_fit(x, y, n, d, ip, fp, res, fw, iw)
    elif algo == ALGO_ARD:
        ard_fit(x, y, n, d, ip, fp, res, fw, iw)
    elif algo == ALGO_LARS:
        lars_fit(x, y, n, d, ip, fp, res, fw, iw)
    elif algo == ALGO_QUANTILE:
        quantile_fit(x, y, n, d, ip, fp, res, fw, iw)
    elif algo == ALGO_RIDGE:
        ridge_fit(x, y, n, d, ip, fp, res, fw, iw)
    elif algo == ALGO_ENETCV:
        enetcv_fit(x, y, n, d, ip, fp, res, fw, iw)
    elif algo == ALGO_LOGCV:
        logcv_fit(t, x, y, n, d, ip, fp, res, fw, iw)
    elif algo == ALGO_ISOTONIC:
        isotonic_fit(x, y, n, d, ip, fp, res, fw, iw)
    elif algo == ALGO_ISOTONIC_PREDICT:
        isotonic_predict(x, y, n, d, ip, fp, res, fw, iw)


def decision_one(x: FP, i: Int, d: Int, wb: FP, c: Int, link: Int) -> Float32:
    var woff = c * (d + 1)
    # DEVIATION 5007 (IDENTITY_PATHS row 107): the fold first, the intercept last
    var z = fa(row_dot(x, i, d, wb, woff), ld(wb, woff + d))
    if link == LINK_EXP:
        return fexp(z)
    if link == LINK_SIGMOID:
        return ftz(identical_sigmoid(z))
    return z
